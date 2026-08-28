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
HORIZONS = [1]     # patches to forecast (context = SEQ_PATCHES - horizon)
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

def to_patches(x):
    b, L = x.shape
    return x.reshape(b, L // PATCH_LEN, PATCH_LEN)


class inject_oracle_stats:
    """Context manager forcing a (global) RevIN to use FIXED per-series stats.
    mean/std are (B,) tensors."""
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
    md_all, sd_all, l2_all, cos_all = [], [], [], []
    for data in tqdm(loader, leave=False):
        data = data.to(DEVICE)

        # context+future stats (mu*, sigma*) -- the future-informed leak,
        # computed over the full window so independent of the horizon
        full_p = to_patches(data)
        o_mean = full_p.mean(dim=(1, 2))
        o_std = full_p.std(dim=(1, 2)) + EPS

        for H in HORIZONS:
            C = (SEQ_PATCHES - H) * PATCH_LEN
            context = data[:, :C]

            # context-only stats -- what inference actually sees
            ctx_p = to_patches(context)
            c_mean = ctx_p.mean(dim=(1, 2))
            c_std = ctx_p.std(dim=(1, 2)) + EPS

            # standardized, dimensionally-clean divergences
            mean_div = (o_mean - c_mean).abs() / o_std       # (B,)
            std_div = (o_std / c_std).log().abs()            # (B,)

            # same context embedded under the two stat regimes
            e_leak = last_patch_embeddings(model, context, (o_mean, o_std))
            e_nolk = last_patch_embeddings(model, context, (c_mean, c_std))
            d_l2 = (e_leak - e_nolk).norm(dim=1)             # (B,)
            d_cos = 1.0 - F.cosine_similarity(e_leak, e_nolk, dim=1)

            md_all.append(mean_div.cpu().numpy())
            sd_all.append(std_div.cpu().numpy())
            l2_all.append(d_l2.cpu().numpy())
            cos_all.append(d_cos.cpu().numpy())

    return (np.concatenate(md_all), np.concatenate(sd_all),
            np.concatenate(l2_all), np.concatenate(cos_all))


def parse_args():
    p = argparse.ArgumentParser(
        description="Embedding displacement caused by the normalization leak.")
    p.add_argument("--with-l2", action="store_true",
                   help="also plot the L2 panel (per-model latent scale, not "
                        "comparable across models); both distances are always "
                        "saved to the npz")
    return p.parse_args()


def median_ci(y):
    """~95% confidence interval of the MEDIAN via the McGill notch formula:
    median +- 1.57*IQR/sqrt(n). Narrow for large bins (shrinks as 1/sqrt(n)),
    unlike an IQR band which shows spread; costless, no resampling."""
    med = np.median(y)
    iqr = np.percentile(y, 75) - np.percentile(y, 25)
    half = 1.57 * iqr / np.sqrt(len(y))
    return med - half, med + half


def binned(x, y, edges):
    """Bin along the divergence axis x using `edges`. Robust to the heavy
    tails of real data: per-bin center is the MEDIAN, with a ~95% CI of that
    median (McGill notch formula). Returns arrays sorted by x."""
    nbins = len(edges) - 1
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, nbins - 1)
    xs, ys, los, his = [], [], [], []
    for b in range(nbins):
        m = idx == b
        if not m.any():
            continue
        xs.append(np.median(x[m]))
        ys.append(np.median(y[m]))
        lo, hi = median_ci(y[m])
        los.append(lo)
        his.append(hi)
    xs, ys, los, his = (np.array(a) for a in (xs, ys, los, his))
    order = np.argsort(xs)
    return xs[order], ys[order], los[order], his[order]


def main():

    args = parse_args()

    out = "results/embedding_distance_gift.npz"
    if os.path.exists(out):
        print(f"loading existing results -> {out}")
        results = dict(np.load(out))
    else:
        os.makedirs("results", exist_ok=True)
        print(f"running experiment -> {out}")
        print(f"device: {DEVICE}")
        print(f"seq={SEQ_PATCHES}p  horizons={HORIZONS}  patch_len={PATCH_LEN} \n")

        train_cfg = TrainConfig()
        eval_cfg = EvalConfig()
        dataset = GiftEvalPretrain(
            path=eval_cfg.data_path,
            input_len=1024,
            normalize=True
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=10)
        results = {}

        for strat, asinh, pt in MODELS:
            cfg = PatchFMConfig(normalization_strategy=strat, use_asinh=asinh,
                                prefix_tokens=(pt or 4), compile=False)
            label = f"{strat}{'+asinh' if asinh else ''}"
            model = get_model(cfg, train_cfg, eval_cfg).to(DEVICE).eval()
            md, sd, d_l2, d_cos = evaluate(model, loader)
            # divergences are model-free -> store once, distances per model
            results.setdefault("mean_div", md)
            results.setdefault("std_div", sd)
            results[f"l2_{label}"] = d_l2
            results[f"cos_{label}"] = d_cos
            print(f"{label:<14} median L2={np.median(d_l2):.3f}  "
                  f"median cos={np.median(d_cos):.4f}")
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        np.savez(out, **results)
        print(f"\nresults saved -> {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # columns = divergence driving the x-axis; rows = distance metric.
        # cosine only by default: scale-invariant, comparable across models
        # (the L2 lives in each model's own latent scale)
        DIVS = [
            ("mean_div", r"$\Delta\mu$"),
            ("std_div",  r"$\Delta\sigma$"),
        ]
        METRICS = [("cos_", r"median cosine distance")]
        if args.with_l2:
            METRICS.insert(0, ("l2_", r"median $L_2$ distance"))

        fig, axes = plt.subplots(len(METRICS), len(DIVS),
                                 figsize=(13, 4.5 * len(METRICS)),
                                 gridspec_kw={"wspace": 0.18, "hspace": 0.3},
                                 squeeze=False)
        for r, (dprefix, ylabel) in enumerate(METRICS):
            for c, (dkey, xlabel) in enumerate(DIVS):
                ax = axes[r, c]
                div = results[dkey]
                # uniformly-spaced bins over the divergence range; drop the
                # heavy tail beyond the 98th pct so bins stay populated and
                # truly uniform
                xhi = np.percentile(div, 98)
                keep = div <= xhi
                div_k = div[keep]
                edges = np.linspace(div_k.min(), xhi, NS_BINS + 1)
                edges[-1] += 1e-9
                xall, yvals = [], []
                for strat, asinh, pt in MODELS:
                    label = f"{strat}{'+asinh' if asinh else ''}"
                    dist = results[f"{dprefix}{label}"][keep]
                    xs, ys, lo, hi = binned(div_k, dist, edges)
                    disp, color = LABEL_STYLE.get(label, (label, "black"))
                    ax.plot(xs, ys, marker="o", ms=5, color=color,
                            label=disp, zorder=3)
                    ax.fill_between(xs, lo, hi, color=color, alpha=0.18,
                                    zorder=1)
                    yvals.append(ys); xall.append(xs)
                    yvals.append(lo); yvals.append(hi)
                ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
                ax.set_xlabel(xlabel, fontsize=14)
                if c == 0:
                    ax.set_ylabel(ylabel, fontsize=13)
                xa = np.concatenate(xall)
                xpad = 0.03 * (xa.max() - xa.min())
                ax.set_xlim(xa.min() - xpad, xa.max() + xpad)
                yv = np.concatenate(yvals)
                ypad = 0.15 * (yv.max() - yv.min())
                ax.set_ylim(min(0.0, yv.min()) - ypad, yv.max() + ypad)
                ax.grid(True, alpha=0.25)
                if r == 0 and c == 0:
                    ax.legend(title="strategy", loc="upper left",
                              frameon=False)

        fig.suptitle("Displacement of the last context-patch hidden state "
                     "caused by the leaked statistics\n"
                     r"distance between "
                     r"$\mathrm{h}_{\mathrm{P}_{31}}^{(\text{context+future})}$ and "
                     r"$\mathrm{h}_{\mathrm{P}_{31}}^{(\text{context-only})}$"
                     r"  (median per bin, band = 95% CI of the median)",
                     fontsize=13)
        # leave room between the two-line suptitle and the axes
        fig.subplots_adjust(top=0.85 if len(METRICS) == 1 else 0.90)
        suffix = "_l2" if args.with_l2 else ""
        out = f"figs/embedding_distance_gift{suffix}.pdf"
        os.makedirs('figs', exist_ok=True)
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"saved plot -> {out}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
