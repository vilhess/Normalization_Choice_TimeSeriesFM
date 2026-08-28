import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display, save figures only
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import sys
sys.path.append('../')

from configs import PatchFMConfig, TrainConfig, EvalConfig
from model.inference import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-5

PATCH_LEN = PatchFMConfig().patch_len
SEQ_PATCHES = 32
N_TAR = 1                          # future patches (context = SEQ_PATCHES - N_TAR)
SEED = 0

# --- controlled sinusoid knobs -------------------------------------------------
# Signal family "sin(x) + x" (cf ../tsne.py): an increasing sinusoid whose
# slope and amplitude growth both scale with the level in [0, 1.5]; level=0 is a
# plain stationary sine, level=1 matches tsne.py's DRIFT signal. Each signal
# is annotated with its measured stat divergences (|mu*-mu_ctx|/sigma* and
# |log(sigma*/sigma_ctx)|), consistent with the leak_scaling_* figures.
X_MAX = 50.0      # time span
FREQ = 3.0        # sinusoid frequency
AMP_GROWTH = 0.1  # amplitude envelope: 1 + level * AMP_GROWTH * x
SLOPE = 1 / 3     # trend: level * SLOPE * x
NOISE = 0.05      # small observation noise
LEVELS = np.linspace(0.0, 1, 21).tolist()   # dense: one point per level
SHOW_LEVELS = LEVELS[::5]                     # subset shown in the signal figure

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

# regime -> (marker, line style) used in every panel
REGIMES = {
    "leak":   ("o", "-"),    # context+future stats (training regime)
    "noleak": ("X", "--"),   # context-only stats (inference regime)
}


def make_series(length, level, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    x = torch.linspace(0.0, X_MAX, length)
    amplitude = 1.0 + level * AMP_GROWTH * x          # growing envelope
    y = amplitude * torch.sin(FREQ * x) + level * SLOPE * x
    noise = torch.randn(length, generator=g) * NOISE
    return (y + noise).unsqueeze(0)


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


@torch.no_grad()
def extract_last_patch_embedding(model, signal, stats):
    captured = {}
    handle = model.transformer_encoder.register_forward_hook(
        lambda _m, _inp, out: captured.__setitem__("emb", out)
    )
    with inject_oracle_stats(model.revin, *stats):
        model.forward(signal)
    handle.remove()
    return captured["emb"].squeeze(0)[-1].cpu().numpy()  # (d_model,)


def main():

    train_cfg = TrainConfig(checkpoint_path="../ckpts")
    eval_cfg = EvalConfig()

    C = (SEQ_PATCHES - N_TAR) * PATCH_LEN

    # ---- signals: ONE series per level, with measured stat divergences -----
    signals, contexts, stats, ns_of, std_of = {}, {}, {}, {}, {}
    for lvl in LEVELS:
        data = make_series(SEQ_PATCHES * PATCH_LEN, lvl)
        full_p = to_patches(data)
        o_mean = full_p.mean(dim=(1, 2))                 # context+future (leak)
        o_std = full_p.std(dim=(1, 2)) + EPS
        ctx_p = to_patches(data[:, :C])
        c_mean = ctx_p.mean(dim=(1, 2))                  # context-only (no leak)
        c_std = ctx_p.std(dim=(1, 2)) + EPS
        signals[lvl] = data
        contexts[lvl] = data[:, :C].to(DEVICE)           # the model sees this
        stats[lvl] = {"leak": (o_mean.to(DEVICE), o_std.to(DEVICE)),
                      "noleak": (c_mean.to(DEVICE), c_std.to(DEVICE))}
        # standardized, dimensionally-clean divergences (as in leak_scaling_*)
        ns_of[lvl] = ((o_mean - c_mean).abs() / o_std).item()
        std_of[lvl] = (o_std / c_std).log().abs().item()
        print(f"level={lvl:.2f}  |mu*-mu_ctx|/sigma*={ns_of[lvl]:.3f}  "
              f"|log(sigma*/sigma_ctx)|={std_of[lvl]:.3f}")

    # ---- last-patch embedding: per model x level x regime ------------------
    embeddings = {}
    for strat, asinh, pt in MODELS:
        cfg = PatchFMConfig(normalization_strategy=strat, use_asinh=asinh,
                            prefix_tokens=(pt or 4), compile=False)
        label = f"{strat}{'+asinh' if asinh else ''}"
        model = get_model(cfg, train_cfg, eval_cfg).to(DEVICE).eval()
        for lvl in LEVELS:
            for regime in REGIMES:
                emb = extract_last_patch_embedding(model, contexts[lvl],
                                                   stats[lvl][regime])
                embeddings[(label, lvl, regime)] = emb
        print(f"{label:<14} last-patch emb {emb.shape} x {len(LEVELS)} levels "
              f"x 2 regimes")
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ---- 1. a few input signals (subset of levels) --------------------------
    fig0, axes0 = plt.subplots(len(SHOW_LEVELS), 1,
                               figsize=(11, 1.8 * len(SHOW_LEVELS)), sharex=True)
    for ax, lvl in zip(np.atleast_1d(axes0), SHOW_LEVELS):
        y = signals[lvl].squeeze(0).numpy()
        ax.plot(y, lw=0.9, color="0.2")
        ax.axvline(C, color="red", ls="--", lw=1.0)
        ax.set_ylabel(f"$\\Delta\\mu$={ns_of[lvl]:.2f}\n"
                      f"$\\Delta\\sigma$={std_of[lvl]:.2f}",
                      rotation=0, ha="right", va="center", fontsize=13)
        ax.set_yticks([])
    np.atleast_1d(axes0)[-1].set_xlabel(
        "t   (red line = context/future boundary)", fontsize=12)
    fig0.suptitle("Signals at a few non-stationarity levels\n"
                  r"$\Delta\mu = |\mu_{\text{context+future}}-"
                  r"\mu_{\text{context-only}}|\,/\,\sigma_{\text{context+future}}$"
                  r"$\qquad$"
                  r"$\Delta\sigma = |\log(\sigma_{\text{context+future}}/"
                  r"\sigma_{\text{context-only}})|$",
                  fontsize=13)
    fig0.savefig("figs/tsne_leakage_signals.pdf", dpi=130, bbox_inches="tight")

    # ---- 2. per strategy: last-patch trajectory vs ns (t-SNE row, PCA row) --
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(ns_of.values()), vmax=max(ns_of.values()))

    n_cfg = len(MODELS)
    methods = ["t-SNE", "PCA"]
    fig1, axes1 = plt.subplots(len(methods), n_cfg,
                               figsize=(4.6 * n_cfg, 4.6 * len(methods)),
                               squeeze=False)
    for idx, (strat, asinh, pt) in enumerate(MODELS):
        label = f"{strat}{'+asinh' if asinh else ''}"
        disp, color = LABEL_STYLE.get(label, (label, "black"))

        # both projections are fit on the same 2 x len(LEVELS) last-patch
        # vectors of THIS model only
        keys = [(lvl, regime) for regime in REGIMES for lvl in LEVELS]
        stacked = np.stack([embeddings[(label, lvl, r)] for lvl, r in keys])
        perplexity = max(2, min(15, stacked.shape[0] - 1))
        projections = {
            "t-SNE": TSNE(n_components=2, perplexity=perplexity,
                          random_state=42, init="pca").fit_transform(stacked),
            "PCA": PCA(n_components=2).fit_transform(stacked),
        }

        for row, method in enumerate(methods):
            ax = axes1[row, idx]
            coords = projections[method]
            L = len(LEVELS)
            # link the SAME level across the two regimes: the segment length
            # shows how far the leak moves the embedding at that ns
            for i in range(L):
                ax.plot([coords[i, 0], coords[L + i, 0]],
                        [coords[i, 1], coords[L + i, 1]],
                        "-", color="0.6", alpha=0.6, lw=0.9, zorder=2)
            for r_idx, (regime, (marker, ls)) in enumerate(REGIMES.items()):
                pts = coords[r_idx * L:(r_idx + 1) * L]
                ax.scatter(pts[:, 0], pts[:, 1],
                           c=[ns_of[lvl] for lvl in LEVELS], cmap=cmap,
                           norm=norm, marker=marker, s=55, edgecolor="k",
                           linewidth=0.3, zorder=3)
            if row == 0:
                ax.set_title(disp, color=color, fontweight="bold", fontsize=13)
            if idx == 0:
                ax.set_ylabel(method, fontweight="bold", fontsize=13)
            ax.set_xticks([]); ax.set_yticks([])

    # shared legends: regime markers + divergence colorbar
    reg_handles = [plt.Line2D([], [], marker=m, ls="", color="0.3",
                              markeredgecolor="k", markersize=9,
                              label=("context+future stats (leak)" if r == "leak"
                                     else "context-only stats (no leak)"))
                   for r, (m, _) in REGIMES.items()]
    fig1.legend(handles=reg_handles, loc="lower center", ncol=2,
                frameon=False, fontsize=15)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig1.colorbar(sm, ax=axes1.ravel().tolist(), fraction=0.025,
                         pad=0.02)
    cbar.set_label(r"$\Delta\mu$", fontsize=17, rotation=0)
    cbar.ax.tick_params(labelsize=13)
    fig1.suptitle("t-SNE (top) and PCA (bottom) of the LAST context-patch "
                  "embedding per strategy\n"
                  "same signal embedded with both stat regimes, pairs linked "
                  "at equal non-stationarity"
                  r"$\qquad\Delta\mu = |\mu_{\text{context+future}}-"
                  r"\mu_{\text{context-only}}|\,/\,\sigma_{\text{context+future}}$",
                  fontsize=16, fontweight="bold")
    fig1.savefig("figs/tsne_leakage.pdf", dpi=130, bbox_inches="tight")

    print("saved plots -> tsne_leakage_signals.pdf, tsne_leakage.pdf")


if __name__ == "__main__":
    main()
