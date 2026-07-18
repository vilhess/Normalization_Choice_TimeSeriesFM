"""Distribution-level divergence between the truncated / full-context
embedding clouds, for every truncation level j.

Combines the two protocols: the truncated-statistics ablation of
delta_mase_various_k.py (the model always sees the FULL 31-patch context,
but the RevIN statistics are computed from only the first j patches) and the
distributional comparison of embedding_regime_divergence_gift.py. For each
strategy and each j, the two clouds of last-context-patch hidden states

  * truncated:    RevIN stats from the first j patches
  * full-context: RevIN stats from all 31 patches (standard inference, j=31)

are compared with their EXACT 2-Wasserstein distance (POT's ot.emd2: full
optimal transport via network simplex). j=31 reproduces the reference stats,
so its W2 is 0 by construction.

Latent scales differ across models, so for each (model, j) pair the two
clouds are standardized with their pooled per-dim stats before the metric.
Each observed value is read against a PERMUTATION NULL (random equal-size
splits of the pooled cloud): empirical W2 has a finite-sample floor
(convergence in N^(-1/d)), so only the excess observed-vs-null is meaningful.

Cost note: one exact W2 costs an OT solve on an N x N cost matrix, and the
null multiplies it by N_PERM; the total is len(J_GRID) * (1 + N_PERM) solves
per model, so N_PERM is kept moderate.
"""

import os
import torch
import numpy as np
import ot
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
SEED = 0

N_PERM = 10                     # permutation splits for the null (per j!)

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
    """Encoder output of the LAST context patch for a whole batch, via a
    forward hook, with the RevIN statistics forced to `stats` = (mean, std),
    each a (B,) tensor. Shape: (B, d_model)."""
    captured = {}
    handle = model.transformer_encoder.register_forward_hook(
        lambda _m, _inp, out: captured.__setitem__("emb", out)
    )
    with inject_oracle_stats(model.revin, *stats):
        model.forward(x)
    handle.remove()
    return captured["emb"][:, -1, :]                     # (B, d_model)


@torch.inference_mode()
def collect_clouds(model, loader):
    """One pass over the dataset. Returns the reference cloud (N, d_model)
    (full-context stats) and one cloud per j (truncated stats)."""
    ref_all = []
    trunc_all = [[] for _ in J_GRID]
    C = CTX_PATCHES * PATCH_LEN
    for data in tqdm(loader, leave=False):
        data = data.to(DEVICE)
        context = data[:, :C]
        ctx_p = to_patches(context)

        # reference: hidden state under standard inference stats (j = 31)
        r_mean = ctx_p.mean(dim=(1, 2))
        r_std = ctx_p.std(dim=(1, 2)) + EPS
        h_ref = last_patch_embeddings(model, context, (r_mean, r_std))
        ref_all.append(h_ref.float().cpu())

        # ablation: stats truncated to the first j patches
        for i, j in enumerate(J_GRID):
            t_mean = ctx_p[:, :j].mean(dim=(1, 2))
            t_std = ctx_p[:, :j].std(dim=(1, 2)) + EPS
            h_j = last_patch_embeddings(model, context, (t_mean, t_std))
            trunc_all[i].append(h_j.float().cpu())

    return torch.cat(ref_all), [torch.cat(t) for t in trunc_all]


def exact_w2(A, B):
    """Exact 2-Wasserstein between two clouds (numpy (N,d) arrays), uniform
    weights: full OT plan via network simplex on the squared-Euclidean cost
    matrix. emd2 returns W2^2 -> sqrt."""
    a = np.ones(len(A)) / len(A)
    b = np.ones(len(B)) / len(B)
    M = ot.dist(A, B, metric="sqeuclidean")
    return np.sqrt(ot.emd2(a, b, M, numItermax=1_000_000))


def permutation_null(A, B, n_perm=N_PERM, seed=SEED):
    """W2 between random equal-size splits of the pooled cloud: the
    finite-sample floor each observed value must be read against."""
    pooled = np.concatenate([A, B])
    n = len(A)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(len(pooled))
        null.append(exact_w2(pooled[perm[:n]], pooled[perm[n:2 * n]]))
    return np.array(null)


def main():

    out = "embedding_regime_divergence_various_k.npz"
    if os.path.exists(out):
        print(f"loading existing results -> {out}")
        results = dict(np.load(out))
    else:
        print(f"running experiment -> {out}")
        print(f"device: {DEVICE}")
        print(f"seq={SEQ_PATCHES}p  ctx={CTX_PATCHES}p  patch_len={PATCH_LEN}  "
              f"j in [{J_GRID[0]}..{J_GRID[-1]}]  n_perm={N_PERM}\n")

        train_cfg = TrainConfig()
        eval_cfg = EvalConfig()
        dataset = GiftEvalPretrain(
            path=eval_cfg.data_path,
            input_len=1024,
            normalize=True,
            max_samples=20
        )
        print(f"length of dataset: {len(dataset)}") # 3097
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=10)
        results = {"j_grid": np.array(J_GRID)}

        for strat, asinh, pt in MODELS:
            cfg = PatchFMConfig(normalization_strategy=strat, use_asinh=asinh,
                                prefix_tokens=(pt or 4), compile=False)
            label = f"{strat}{'+asinh' if asinh else ''}"
            model = get_model(cfg, train_cfg, eval_cfg).to(DEVICE).eval()
            e_ref, e_trunc = collect_clouds(model, loader)
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            w2_obs = np.zeros(len(J_GRID))
            w2_null = np.zeros((len(J_GRID), N_PERM))
            for i, j in enumerate(tqdm(J_GRID, desc=label, leave=False)):
                # per-(model, j) standardization with pooled stats -> metric
                # lives on a common scale, comparable across latent spaces
                pooled = torch.cat([e_trunc[i], e_ref]).double()
                m, s = pooled.mean(0), pooled.std(0) + EPS
                A = ((e_trunc[i].double() - m) / s).numpy()
                B = ((e_ref.double() - m) / s).numpy()
                w2_obs[i] = exact_w2(A, B)
                w2_null[i] = permutation_null(A, B)

            results[f"w2_{label}"] = w2_obs
            results[f"w2null_{label}"] = w2_null
            print(f"{label:<14} W2(j=1)={w2_obs[0]:.4f}  "
                  f"(null {w2_null[0].mean():.4f} +- {w2_null[0].std():.4f})  "
                  f"W2(j={J_GRID[-1]})={w2_obs[-1]:.4f}")

        np.savez(out, **results)
        print(f"\nresults saved -> {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        j_grid = results["j_grid"]
        labels = [f"{s}{'+asinh' if a else ''}" for s, a, _ in MODELS]
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        for label in labels:
            disp, color = LABEL_STYLE.get(label, (label, "black"))
            obs = results[f"w2_{label}"]
            null = results[f"w2null_{label}"]
            ax.plot(j_grid, obs, marker="o", ms=4, color=color,
                    label=disp, zorder=3)
            # permutation null: mean +- 2 sd, the finite-sample floor
            ax.plot(j_grid, null.mean(axis=1), color=color, ls="--",
                    lw=1.0, alpha=0.6, zorder=2)
            ax.fill_between(j_grid,
                            null.mean(axis=1) - 2 * null.std(axis=1),
                            null.mean(axis=1) + 2 * null.std(axis=1),
                            color=color, alpha=0.10, zorder=1)
        ax.set_xlabel("number of patches $j$ used for the\n"
                      "normalization statistics", fontsize=13)
        ax.set_ylabel(r"exact $W_2$", fontsize=13)
        ax.set_xticks([j for j in j_grid if j % 5 == 0 or j == 1])
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.plot([], [], color="0.3", ls="--",
                label="permutation null (mean $\\pm$ 2 sd)")
        # legend outside the axes: the curves span the whole plotting area
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  frameon=False, fontsize=10)

        ax.set_title("Exact $W_2$ between the truncated / full-context "
                     "distributions of the last context-patch hidden state\n"
                     "(clouds standardized per model; read each curve "
                     "against its finite-sample null)", fontsize=11)
        fig.tight_layout()
        out = "embedding_regime_divergence_various_k.pdf"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"saved plot -> {out}")
    except Exception as e:
        print(f"(plot skipped: {e})")

if __name__ == "__main__":
    main()
