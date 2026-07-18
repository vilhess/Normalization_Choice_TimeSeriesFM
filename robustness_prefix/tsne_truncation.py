"""Latent-space view of the truncated-statistics ablation.

For each normalization strategy, embed the SAME context under statistics
computed from only its first j patches (j = 1..31; the model always sees the
FULL 31-patch context, cf delta_mase_various_k.py) and watch the trajectory
of the last-context-patch embedding as j grows back to the full window:

  * j < 31: truncated stats  (mu, sigma from the first j patches)
  * j = 31: full-context stats -- the standard inference regime, marked with
            a star: it is the REFERENCE every trajectory should reach.

A few controlled sinusoids "sin(x) + x" (cf ../leakage_measurement/
tsne_leakage.py), one per non-stationarity level; level=0 is a stationary
control for which even vanilla RevIN should not move. If the all-or-nothing
sensitivity seen on dMASE lives in the representation, the vanilla points
for j < 31 should form a compact cluster AWAY from the star and only jump
onto it at j=31; a robust representation (Prefix@k) should keep all 31
points stacked on the star. One t-SNE is fit PER strategy: latent spaces of
different checkpoints are not mutually comparable, so panels are only
comparable within themselves.
"""

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
CTX_PATCHES = SEQ_PATCHES - 1     # full context; the last patch is predicted
J_GRID = list(range(1, CTX_PATCHES + 1))   # patches used for the stats
SEED = 0

# --- controlled sinusoid knobs -------------------------------------------------
# Signal family "sin(x) + x" (cf ../leakage_measurement/tsne_leakage.py): an
# increasing sinusoid whose slope and amplitude growth both scale with the
# level; level=0 is a plain stationary sine (negative control). Few levels
# with distinct markers keep the trajectories readable.
X_MAX = 50.0      # time span
FREQ = 3.0        # sinusoid frequency
AMP_GROWTH = 0.1  # amplitude envelope: 1 + level * AMP_GROWTH * x
SLOPE = 1 / 3     # trend: level * SLOPE * x
NOISE = 0.05      # small observation noise
LEVELS = [0.0, 0.5, 1.0]
LEVEL_MARKERS = {0.0: "o", 0.5: "s", 1.0: "^"}

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


def make_series(length, level, seed=SEED):
    """Increasing sinusoid "sin(x) + x", one series (1, L). `level` scales
    BOTH the trend slope (increasing shape) and the amplitude envelope
    (increasing magnitude); level=0 is a stationary sine."""
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


@torch.no_grad()
def extract_last_patch_embedding(model, signal, stats):
    """Encoder output of the LAST context patch (the vector used to predict
    the target patch) via a forward hook, with the RevIN statistics forced to
    `stats` = (mean, std), each a (B,) tensor. Shape: (d_model,)."""
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

    C = CTX_PATCHES * PATCH_LEN

    # ---- signals: ONE series per level, truncated stats for every j ---------
    contexts, stats, div_of = {}, {}, {}
    for lvl in LEVELS:
        context = make_series(C, lvl)
        ctx_p = to_patches(context)
        r_mean = ctx_p.mean(dim=(1, 2))                  # full-context stats
        r_std = ctx_p.std(dim=(1, 2)) + EPS
        contexts[lvl] = context.to(DEVICE)               # the model sees this
        stats[lvl], div_of[lvl] = {}, {}
        for j in J_GRID:
            t_mean = ctx_p[:, :j].mean(dim=(1, 2))       # truncated stats
            t_std = ctx_p[:, :j].std(dim=(1, 2)) + EPS
            stats[lvl][j] = (t_mean.to(DEVICE), t_std.to(DEVICE))
            # standardized divergence of the truncated stats from the
            # full-context ones (the x-axis of delta_mase_various_k.py)
            div_of[lvl][j] = ((t_mean - r_mean).abs() / r_std).item()
        print(f"level={lvl:.2f}  Delta mu(j=1)={div_of[lvl][J_GRID[0]]:.3f}  "
              f"(j={J_GRID[-1]})={div_of[lvl][J_GRID[-1]]:.3f}")

    # ---- last-patch embedding: per model x level x j ------------------------
    embeddings = {}
    for strat, asinh, pt in MODELS:
        cfg = PatchFMConfig(normalization_strategy=strat, use_asinh=asinh,
                            prefix_tokens=(pt or 4), compile=False)
        label = f"{strat}{'+asinh' if asinh else ''}"
        model = get_model(cfg, train_cfg, eval_cfg).to(DEVICE).eval()
        for lvl in LEVELS:
            for j in J_GRID:
                emb = extract_last_patch_embedding(model, contexts[lvl],
                                                   stats[lvl][j])
                embeddings[(label, lvl, j)] = emb
        print(f"{label:<14} last-patch emb {emb.shape} x {len(LEVELS)} levels "
              f"x {len(J_GRID)} truncations")
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ---- 1. the input signals, annotated with the worst-case divergence -----
    fig0, axes0 = plt.subplots(len(LEVELS), 1,
                               figsize=(11, 1.8 * len(LEVELS)), sharex=True)
    for ax, lvl in zip(np.atleast_1d(axes0), LEVELS):
        y = contexts[lvl].squeeze(0).cpu().numpy()
        ax.plot(y, lw=0.9, color="0.2")
        ax.set_ylabel(f"$\\Delta\\mu(j{{=}}1)$\n={div_of[lvl][1]:.2f}",
                      rotation=0, ha="right", va="center", fontsize=13)
        ax.set_yticks([])
    np.atleast_1d(axes0)[-1].set_xlabel("t", fontsize=12)
    fig0.suptitle("Contexts at a few non-stationarity levels\n"
                  r"$\Delta\mu(j) = |\mu_{1:j}-\mu_{1:31}|"
                  r"\,/\,\sigma_{1:31}$",
                  fontsize=13)
    fig0.savefig("tsne_truncation_signals.pdf", dpi=130, bbox_inches="tight")

    # ---- 2. per strategy: last-patch trajectory vs j (t-SNE row, PCA row) ---
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=J_GRID[0], vmax=J_GRID[-1])

    n_cfg = len(MODELS)
    methods = ["t-SNE", "PCA"]
    fig1, axes1 = plt.subplots(len(methods), n_cfg,
                               figsize=(4.6 * n_cfg, 4.6 * len(methods)),
                               squeeze=False)
    for idx, (strat, asinh, pt) in enumerate(MODELS):
        label = f"{strat}{'+asinh' if asinh else ''}"
        disp, color = LABEL_STYLE.get(label, (label, "black"))

        # both projections are fit on the same len(LEVELS) x len(J_GRID)
        # last-patch vectors of THIS model only
        keys = [(lvl, j) for lvl in LEVELS for j in J_GRID]
        stacked = np.stack([embeddings[(label, lvl, j)] for lvl, j in keys])
        perplexity = max(2, min(15, stacked.shape[0] - 1))
        projections = {
            "t-SNE": TSNE(n_components=2, perplexity=perplexity,
                          random_state=42, init="pca").fit_transform(stacked),
            "PCA": PCA(n_components=2).fit_transform(stacked),
        }

        for row, method in enumerate(methods):
            ax = axes1[row, idx]
            coords = projections[method]
            J = len(J_GRID)
            for l_idx, lvl in enumerate(LEVELS):
                pts = coords[l_idx * J:(l_idx + 1) * J]
                # the trajectory from j=1 to the full-context reference: its
                # length shows how far the truncation moves the embedding
                ax.plot(pts[:, 0], pts[:, 1], "-", color="0.6", alpha=0.6,
                        lw=0.9, zorder=2)
                ax.scatter(pts[:, 0], pts[:, 1], c=J_GRID, cmap=cmap,
                           norm=norm, marker=LEVEL_MARKERS[lvl], s=55,
                           edgecolor="k", linewidth=0.3, zorder=3)
                # j=31 = full-context stats: the reference to reach
                ax.scatter(pts[-1, 0], pts[-1, 1], marker="*", s=320,
                           facecolor=cmap(norm(J_GRID[-1])), edgecolor="k",
                           linewidth=0.8, zorder=4)
            if row == 0:
                ax.set_title(disp, color=color, fontweight="bold", fontsize=13)
            if idx == 0:
                ax.set_ylabel(method, fontweight="bold", fontsize=13)
            ax.set_xticks([]); ax.set_yticks([])

    # shared legends: level markers + reference star + truncation colorbar
    lvl_handles = [plt.Line2D([], [], marker=LEVEL_MARKERS[lvl], ls="",
                              color="0.3", markeredgecolor="k", markersize=9,
                              label=f"$\\Delta\\mu(j{{=}}1)$"
                                    f"={div_of[lvl][1]:.2f}")
                   for lvl in LEVELS]
    lvl_handles.append(plt.Line2D([], [], marker="*", ls="", color="0.3",
                                  markeredgecolor="k", markersize=15,
                                  label="$j=31$ (full-context stats, "
                                        "reference)"))
    fig1.legend(handles=lvl_handles, loc="lower center",
                ncol=len(lvl_handles), frameon=False, fontsize=15)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig1.colorbar(sm, ax=axes1.ravel().tolist(), fraction=0.025,
                         pad=0.02)
    cbar.set_label(r"$j$", fontsize=17, rotation=0)
    cbar.ax.tick_params(labelsize=13)
    fig1.suptitle("t-SNE (top) and PCA (bottom) of the LAST context-patch "
                  "embedding per strategy\n"
                  "same context embedded with statistics from its first $j$ "
                  "patches, one trajectory per signal"
                  r"$\qquad\Delta\mu(j) = |\mu_{1:j}-\mu_{1:31}|"
                  r"\,/\,\sigma_{1:31}$",
                  fontsize=16, fontweight="bold")
    fig1.savefig("tsne_truncation.pdf", dpi=130, bbox_inches="tight")

    print("saved plots -> tsne_truncation_signals.pdf, tsne_truncation.pdf")


if __name__ == "__main__":
    main()
