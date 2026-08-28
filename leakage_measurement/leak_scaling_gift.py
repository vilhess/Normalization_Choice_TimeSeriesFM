import os
import torch
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
def rollout(model, context, n_fut, oracle=None):
    cur = context
    preds = []
    for _ in range(n_fut):
        if oracle is not None:
            with inject_oracle_stats(model.revin, *oracle):
                out = model.forward(cur)             # (B, PN, PL, n_quantiles)
        else:
            out = model.forward(cur)
        nxt = out[:, -1, :, 4]                        # median of the next patch
        preds.append(nxt)
        cur = torch.cat([cur, nxt], dim=1)
    return torch.cat(preds, dim=1)                    # (B, H*PL)


def mase_per_series(pred, true, context):
    """MASE per series (B,), not averaged over the batch."""
    denom = context.diff(dim=1).abs().mean(dim=1).clamp_min(EPS)   # (B,)
    err = (pred - true).abs().mean(dim=1)                          # (B,)
    return err / denom                                             # (B,)


@torch.inference_mode()
def evaluate(model, loader):
    md_all, sd_all, dmase_all = [], [], []
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
            true_fut = data[:, C:]

            # context-only stats -- what inference actually sees
            ctx_p = to_patches(context)
            c_mean = ctx_p.mean(dim=(1, 2))
            c_std = ctx_p.std(dim=(1, 2)) + EPS

            # standardized, dimensionally-clean divergences
            mean_div = (o_mean - c_mean).abs() / o_std       # (B,)
            std_div = (o_std / c_std).log().abs()            # (B,)

            # forecasts: honest (context-only) vs oracle (context+future)
            honest = rollout(model, context, H, oracle=None)
            oracle = rollout(model, context, H, oracle=(o_mean, o_std))
            dmase = (mase_per_series(honest, true_fut, context)
                     - mase_per_series(oracle, true_fut, context))

            md_all.append(mean_div.cpu().numpy())
            sd_all.append(std_div.cpu().numpy())
            dmase_all.append(dmase.cpu().numpy())

    return (np.concatenate(md_all),
            np.concatenate(sd_all), np.concatenate(dmase_all))


def median_ci(y):
    med = np.median(y)
    iqr = np.percentile(y, 75) - np.percentile(y, 25)
    half = 1.57 * iqr / np.sqrt(len(y))
    return med - half, med + half


def binned(x, y, edges):
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

    out = "results/leak_scaling_gift.npz"
    if os.path.exists(out):
        print(f"loading existing results -> {out}")
        results = dict(np.load(out))
    else:
        print(f"running experiment -> {out}")
        print(f"device: {DEVICE}")
        print(f"seq={SEQ_PATCHES}p  horizons={HORIZONS}  patch_len={PATCH_LEN} \n")

        train_cfg = TrainConfig(checkpoint_path="../",)
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
            md, sd, dmase = evaluate(model, loader)
            # divergences are model-free -> store once, dmase per model
            results.setdefault("mean_div", md)
            results.setdefault("std_div", sd)
            results[f"dmase_{label}"] = dmase
            print(f"{label:<14} mean dMASE={dmase.mean():+.3f}")
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        np.savez(out, **results)
        print(f"\nresults saved -> {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        PANELS = [
            ("mean_div",
             r"$\Delta\mu$"),
            ("std_div",
             r"$\Delta\sigma$"),
        ]

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                                 gridspec_kw={"wspace": 0.18})
        for k, (ax, (dkey, xlabel)) in enumerate(zip(axes, PANELS)):
            div = results[dkey]
            # uniformly-spaced bins over the divergence range; drop the heavy
            # tail beyond the 98th pct so bins stay populated and truly uniform
            xhi = np.percentile(div, 98)
            keep = div <= xhi
            div_k = div[keep]
            edges = np.linspace(div_k.min(), xhi, NS_BINS + 1)
            edges[-1] += 1e-9
            xall, yvals = [], []
            for strat, asinh, pt in MODELS:
                label = f"{strat}{'+asinh' if asinh else ''}"
                dmase = results[f"dmase_{label}"][keep]
                xs, ys, lo, hi = binned(div_k, dmase, edges)
                disp, color = LABEL_STYLE.get(label, (label, "black"))
                ax.plot(xs, ys, marker="o", ms=5, color=color,
                        label=disp, zorder=3)
                ax.fill_between(xs, lo, hi, color=color, alpha=0.18, zorder=1)
                # y-limits driven by the medians only: the min-max envelope of
                # heavy-tailed data would wreck the scale (bands may clip)
                yvals.append(ys); xall.append(xs)
            ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
            ax.set_xlabel(xlabel, fontsize=14)
            ax.set_ylabel(r"median $\Delta$MASE (benefit of the leak)", fontsize=14)
            # per-panel limits from the plotted bin medians (mu and sigma keep
            # independent y-scales)
            xa = np.concatenate(xall)
            xpad = 0.03 * (xa.max() - xa.min())
            ax.set_xlim(xa.min() - xpad, xa.max() + xpad)
            yv = np.concatenate(yvals)
            ypad = 0.15 * (yv.max() - yv.min())
            ax.set_ylim(yv.min() - ypad, yv.max() + ypad)
            ax.grid(True, alpha=0.25)
            if k == 0:
                ax.legend(title="strategy", loc="upper left", frameon=False)

        fig.suptitle("The benefit of the leak scales with the statistic divergence "
                     r"(median $\Delta$MASE per bin, band = 95% CI of the median)",
                     fontsize=13)
        # leave room between the suptitle and the axes
        fig.subplots_adjust(top=0.88)
        out = "figs/leak_scaling_gift.pdf"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"saved plot -> {out}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()

