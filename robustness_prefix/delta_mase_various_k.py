import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

import sys 
sys.path.append("../")
#from dataset import GiftEvalPretrain
from configs import PatchFMConfig, TrainConfig, EvalConfig
from model.inference import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-5

PATCH_LEN = PatchFMConfig().patch_len
SEQ_PATCHES = 32
CTX_PATCHES = SEQ_PATCHES - 1     # full context; the last patch is predicted
J_GRID = list(range(1, CTX_PATCHES + 1))   # patches used for the stats
BATCH_SIZE = 1024
NS_BINS = 5
SEED = 0

MODELS = [
    ("vanilla", True,  None),
    ("vanilla", False, None),
    ("prefix",  True,  4),
    ("prefix",  False, 4),
]

LABEL_STYLE = {
    "vanilla":       (r"RevIN",                   "#00CC66"),
    "vanilla+asinh": (r"RevIN+$\sinh^{-1}$",      "#0055FF"),
    "prefix":        (r"Prefix@$k$",              "#FF3300"),
    "prefix+asinh":  (r"Prefix@$k$+$\sinh^{-1}$", "#CC0099"),
}

PERTURB_TITLE = {
    "both": r"$\mu$ and $\sigma$ truncated",
    "mean": r"$\mu$ truncated, $\sigma$ from the full context",
    "std":  r"$\sigma$ truncated, $\mu$ from the full context",
}

def to_patches(x):
    b, L = x.shape
    return x.reshape(b, L // PATCH_LEN, PATCH_LEN)


class inject_oracle_stats:
    def __init__(self, revin, mean, std):
        self.revin = revin
        self.mean = mean
        self.std = std

    def __enter__(self):
        self._orig = self.revin._get_statistics
        mean, std = self.mean, self.std

        def patched(x):
            B = x.size(0)
            return mean.view(B, 1, 1).to(x.dtype), std.view(B, 1, 1).to(x.dtype)

        self.revin._get_statistics = patched
        return self

    def __exit__(self, *a):
        self.revin._get_statistics = self._orig


@torch.inference_mode()
def predict_next_patch(model, context, stats):
    with inject_oracle_stats(model.revin, *stats):
        out = model.forward(context)                 # (B, PN, PL, n_quantiles)
    return out[:, -1, :, 4]                          # (B, PL)


def mase_per_series(pred, true, context):
    denom = context.diff(dim=1).abs().mean(dim=1).clamp_min(EPS)   # (B,)
    err = (pred - true).abs().mean(dim=1)                          # (B,)
    return err / denom                                             # (B,)


@torch.inference_mode()
def evaluate(model, loader, perturb):
    md_all = []
    sd_all = []
    dmase_all = [[] for _ in J_GRID]
    C = CTX_PATCHES * PATCH_LEN
    for data in tqdm(loader, leave=False):
        data = data.to(DEVICE)
        context = data[:, :C]
        true_fut = data[:, C:]
        ctx_p = to_patches(context)

        # non-stationarity indices: full context vs context+future
        full_p = to_patches(data)
        o_mean = full_p.mean(dim=(1, 2))
        o_std = full_p.std(dim=(1, 2)) + EPS
        c_mean = ctx_p.mean(dim=(1, 2))
        c_std = ctx_p.std(dim=(1, 2)) + EPS
        mean_div = (o_mean - c_mean).abs() / o_std               # (B,)
        std_div = (o_std / c_std).log().abs()                    # (B,)
        md_all.append(mean_div.cpu().numpy())
        sd_all.append(std_div.cpu().numpy())

        # reference: standard inference stats (full context, j = 31)
        r_mean = ctx_p.mean(dim=(1, 2))
        r_std = ctx_p.std(dim=(1, 2)) + EPS
        ref = predict_next_patch(model, context, (r_mean, r_std))
        mase_ref = mase_per_series(ref, true_fut, context)       # (B,)

        # ablation: the selected statistic truncated to the first j patches
        for i, j in enumerate(J_GRID):
            t_mean = ctx_p[:, :j].mean(dim=(1, 2))
            t_std = ctx_p[:, :j].std(dim=(1, 2)) + EPS
            if perturb == "mean":
                t_std = r_std
            elif perturb == "std":
                t_mean = r_mean
            pred = predict_next_patch(model, context, (t_mean, t_std))
            dmase = mase_per_series(pred, true_fut, context) - mase_ref
            dmase_all[i].append(dmase.cpu().numpy())

    return (np.concatenate(md_all), np.concatenate(sd_all),
            np.stack([np.concatenate(d) for d in dmase_all]))    # (J, B)


def parse_args():
    p = argparse.ArgumentParser(
        description="Robustness to truncated normalization statistics, "
                    "decomposed per statistic.")
    p.add_argument("--perturb", choices=["both", "mean", "std"],
                   default="both",
                   help="which statistic is truncated to the first j "
                        "patches; the other keeps its full-context value")
    p.add_argument("--prefix-only", action="store_true",
                   help="plot only the prefix-based strategies (all models "
                        "are still computed and saved to the npz)")
    return p.parse_args()


def main():

    args = parse_args()

    out = f"results/delta_mase_various_k_{args.perturb}.npz"
    if os.path.exists(out):
        print(f"loading existing results -> {out}")
        results = dict(np.load(out))
    else:
        print(f"running experiment -> {out}")
        print(f"device: {DEVICE}")
        print(f"seq={SEQ_PATCHES}p  ctx={CTX_PATCHES}p  patch_len={PATCH_LEN}  "
              f"j in [{J_GRID[0]}..{J_GRID[-1]}]  perturb={args.perturb}\n")

        train_cfg = TrainConfig(checkpoint_path="../",)
        eval_cfg = EvalConfig()
        dataset = GiftEvalPretrain(
            path=eval_cfg.data_path,
            input_len=1024,
            normalize=True
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=10)
        results = {"j_grid": np.array(J_GRID)}

        for strat, asinh, pt in MODELS:
            cfg = PatchFMConfig(normalization_strategy=strat, use_asinh=asinh,
                                prefix_tokens=(pt or 4), compile=False)
            label = f"{strat}{'+asinh' if asinh else ''}"
            model = get_model(cfg, train_cfg, eval_cfg).to(DEVICE).eval()
            md, sd, dmase = evaluate(model, loader, args.perturb)
            # the divergences are model-free -> store once, dmase per model
            results.setdefault("mean_div", md)
            results.setdefault("std_div", sd)
            results[f"dmase_{label}"] = dmase
            print(f"{label:<14} median dMASE(j=1)={np.median(dmase[0]):+.3f}  "
                  f"(j={J_GRID[-1]})={np.median(dmase[-1]):+.3f}")
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        np.savez(out, **results)
        print(f"\nresults saved -> {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        j_grid = results["j_grid"]
        # bin along the divergence of the statistic actually perturbed:
        # Delta sigma when only sigma is truncated, Delta mu otherwise
        div_key = "std_div" if args.perturb == "std" else "mean_div"
        xlabel = r"$\Delta\sigma$" if div_key == "std_div" else r"$\Delta\mu$"
        if div_key not in results:
            print(f"({div_key} missing from the npz -- recompute to get it; "
                  "falling back to mean_div)")
            div_key, xlabel = "mean_div", r"$\Delta\mu$"
        div = results[div_key]
        # uniformly-spaced bins over the divergence range; drop the heavy
        # tail beyond the 98th pct so bins stay populated and truly uniform
        xhi = np.percentile(div, 98)
        keep = div <= xhi
        div_k = div[keep]
        edges = np.linspace(div_k.min(), xhi, NS_BINS + 1)
        edges[-1] += 1e-9
        idx = np.clip(np.digitize(div_k, edges[1:-1]), 0, NS_BINS - 1)
        bin_centers = [np.median(div_k[idx == b]) for b in range(NS_BINS)]

        # models to display (all results stay in the npz regardless)
        plot_models = ([m for m in MODELS if m[0] == "prefix"]
                       if args.prefix_only else MODELS)

        # per-model grid of median dMASE per (j, divergence bin) cell
        grids = {}
        for strat, asinh, pt in plot_models:
            label = f"{strat}{'+asinh' if asinh else ''}"
            dmase = results[f"dmase_{label}"][:, keep]           # (J, B)
            grid = np.zeros((len(j_grid), NS_BINS))
            for b in range(NS_BINS):
                m = idx == b
                grid[:, b] = np.median(dmase[:, m], axis=1)
            grids[label] = grid

        gmin = min(g.min() for g in grids.values())
        gmax = max(g.max() for g in grids.values())
        n = len(grids)
        ncols = 2
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(6.5 * ncols, 4.6 * nrows),
                                 gridspec_kw={"wspace": 0.12, "hspace": 0.28},
                                 sharey=True, squeeze=False)
        im = None
        for k, (ax, (label, grid)) in enumerate(zip(axes.flat, grids.items())):
            im = ax.imshow(grid, aspect="auto", origin="lower", cmap="Purples",
                           vmin=min(0.0, gmin), vmax=gmax)
            disp, color = LABEL_STYLE.get(label, (label, "black"))
            ax.set_title(disp, color=color, fontweight="bold", fontsize=13)
            ax.set_xticks(range(NS_BINS))
            ax.set_xticklabels([f"{c:.2f}" for c in bin_centers])
            yticks = [i for i, j in enumerate(j_grid) if j % 5 == 0 or j == 1]
            ax.set_yticks(yticks)
            ax.set_yticklabels([j_grid[i] for i in yticks])
            row, col = divmod(k, ncols)
            if row == nrows - 1:
                ax.set_xlabel(xlabel, fontsize=14)
            if col == 0:
                ax.set_ylabel("number of patches $j$ used for the\n"
                              "normalization statistics", fontsize=13)
        for ax in axes.flat[n:]:
            ax.axis("off")
        # reserve headroom for the suptitle BEFORE adding the colorbar, which
        # steals its own space from the axes (adjusting after would move the
        # axes back over the colorbar)
        fig.subplots_adjust(top=0.88 if nrows > 1 else 0.78)
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                            fraction=0.035, pad=0.03)
        cbar.set_label(r"median $\Delta$MASE", fontsize=13)
        fig.suptitle("Robustness to truncated normalization statistics: "
                     f"{PERTURB_TITLE[args.perturb]}\n"
                     r"$\Delta$MASE = MASE(perturbed stats) $-$ "
                     r"MASE(full-context stats)",
                     fontsize=13)
        suffix = "_prefix" if args.prefix_only else ""
        out = f"figs/delta_mase_various_k_{args.perturb}{suffix}.pdf"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"saved plot -> {out}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
