import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import sys 
sys.path.append('../')
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

METRIC_LABEL = {
    "cos": r"median cosine distance",
    "l2":  r"median $L_2$ distance (per-model latent scale)",
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
def last_patch_embeddings(model, x, stats):
    captured = {}
    handle = model.transformer_encoder.register_forward_hook(
        lambda _m, _inp, out: captured.__setitem__("emb", out)
    )
    with inject_oracle_stats(model.revin, *stats):
        model.forward(x)
    handle.remove()
    return captured["emb"][:, -1, :]                     # (B, d_model)


@torch.inference_mode()
def evaluate(model, loader):
    md_all = []
    l2_all = [[] for _ in J_GRID]
    cos_all = [[] for _ in J_GRID]
    C = CTX_PATCHES * PATCH_LEN
    for data in tqdm(loader, leave=False):
        data = data.to(DEVICE)
        context = data[:, :C]
        ctx_p = to_patches(context)

        # non-stationarity index: full context vs context+future
        full_p = to_patches(data)
        o_mean = full_p.mean(dim=(1, 2))
        o_std = full_p.std(dim=(1, 2)) + EPS
        c_mean = ctx_p.mean(dim=(1, 2))
        mean_div = (o_mean - c_mean).abs() / o_std               # (B,)
        md_all.append(mean_div.cpu().numpy())

        # reference: hidden state under standard inference stats (j = 31)
        r_mean = ctx_p.mean(dim=(1, 2))
        r_std = ctx_p.std(dim=(1, 2)) + EPS
        h_ref = last_patch_embeddings(model, context, (r_mean, r_std))

        # ablation: stats truncated to the first j patches
        for i, j in enumerate(J_GRID):
            t_mean = ctx_p[:, :j].mean(dim=(1, 2))
            t_std = ctx_p[:, :j].std(dim=(1, 2)) + EPS
            h_j = last_patch_embeddings(model, context, (t_mean, t_std))
            l2_all[i].append((h_j - h_ref).norm(dim=1).cpu().numpy())
            cos_all[i].append(
                (1.0 - F.cosine_similarity(h_j, h_ref, dim=1)).cpu().numpy())

    return (np.concatenate(md_all),
            np.stack([np.concatenate(d) for d in l2_all]),       # (J, B)
            np.stack([np.concatenate(d) for d in cos_all]))      # (J, B)


def parse_args():
    p = argparse.ArgumentParser(
        description="Embedding displacement under truncated statistics.")
    p.add_argument("--prefix-only", action="store_true",
                   help="plot only the prefix-based strategies (all models "
                        "are still computed and saved to the npz)")
    p.add_argument("--metric", choices=["cos", "l2"], default="cos",
                   help="distance shown in the heatmaps (both are saved)")
    return p.parse_args()


def main():

    args = parse_args()

    out = "results/embedding_distance_various_k.npz"
    if os.path.exists(out):
        print(f"loading existing results -> {out}")
        results = dict(np.load(out))
    else:
        os.makedirs("results", exist_ok=True)
        print(f"running experiment -> {out}")
        print(f"device: {DEVICE}")
        print(f"seq={SEQ_PATCHES}p  ctx={CTX_PATCHES}p  patch_len={PATCH_LEN}  "
              f"j in [{J_GRID[0]}..{J_GRID[-1]}]\n")

        train_cfg = TrainConfig()
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
            md, d_l2, d_cos = evaluate(model, loader)
            # the divergence is model-free -> store once, distances per model
            results.setdefault("mean_div", md)
            results[f"l2_{label}"] = d_l2
            results[f"cos_{label}"] = d_cos
            print(f"{label:<14} median cos(j=1)={np.median(d_cos[0]):.4f}  "
                  f"(j={J_GRID[-1]})={np.median(d_cos[-1]):.4f}")
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
        div = results["mean_div"]
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

        # per-model grid of median distance per (j, divergence bin) cell
        grids = {}
        for strat, asinh, pt in plot_models:
            label = f"{strat}{'+asinh' if asinh else ''}"
            dist = results[f"{args.metric}_{label}"][:, keep]     # (J, B)
            grid = np.zeros((len(j_grid), NS_BINS))
            for b in range(NS_BINS):
                m = idx == b
                grid[:, b] = np.median(dist[:, m], axis=1)
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
                ax.set_xlabel(r"$\Delta\mu$", fontsize=14)
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
        cbar.set_label(METRIC_LABEL[args.metric], fontsize=13)
        fig.suptitle("Displacement of the last context-patch hidden state "
                     "under truncated normalization statistics\n"
                     r"distance between "
                     r"$\mathrm{h}_{\mathrm{P}_{31}}^{(\text{first-}j\text{-patch stats})}$"
                     r" and "
                     r"$\mathrm{h}_{\mathrm{P}_{31}}^{(\text{full-context stats})}$",
                     fontsize=13)
        suffix = "_prefix" if args.prefix_only else ""
        out = f"figs/embedding_distance_various_k_{args.metric}{suffix}.pdf"
        os.makedirs('figs', exist_ok=True)
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"saved plot -> {out}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
