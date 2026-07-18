"""Distribution-level divergence between the leak / no-leak embedding clouds.

Complements embedding_distribution_gift.py: instead of the per-series PAIRED
displacement, this compares the whole DISTRIBUTIONS of last-context-patch
embeddings under the two RevIN regimes:

  * leak (training regime):     RevIN stats from context+future (mu*, sigma*)
  * no leak (inference regime): RevIN stats from the context only

For each strategy the two clouds (N, d_model) are collected and their EXACT
2-Wasserstein distance is computed (POT's ot.emd2: full optimal transport via
network simplex -- tractable at this N, deterministic, and free of the
projection noise of sliced W2 / the entropic bias of Sinkhorn).

Latent scales differ across models, so embeddings are standardized per model
(per-dim z-score with the pooled leak+noleak stats) before the metric.
Each observed value is shown against a PERMUTATION NULL (random splits of the
pooled cloud): two finite samples of the same distribution never score 0, so
only the gap observed-vs-null is meaningful when comparing strategies. For
EXACT W2 this matters even more than for sliced: empirical W2 converges as
N^(-1/d), so at N~1e3 in d_model dims the raw value is dominated by the
finite-sample floor -- the excess over the null is the effect
(E[W2_emp^2] ~ W2_true^2 + floor^2, so obs^2 - null^2 estimates the true W2^2).
"""

import os
import torch
import numpy as np
import ot
from tqdm import tqdm

#import sys 
#sys.path.append('../')
from dataset import GiftEvalPretrain
from configs import PatchFMConfig, TrainConfig, EvalConfig
from model.inference import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-5

PATCH_LEN = PatchFMConfig().patch_len
SEQ_PATCHES = 32
HORIZON = 1                     # context = SEQ_PATCHES - HORIZON patches
BATCH_SIZE = 1024
SEED = 0

N_PERM = 20                     # permutation splits for the null

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
    """One pass over the dataset. Returns the two embedding clouds
    (N, d_model): same contexts embedded under leak / no-leak stats."""
    leak_all, nolk_all = [], []
    C = (SEQ_PATCHES - HORIZON) * PATCH_LEN
    for data in tqdm(loader, leave=False):
        data = data.to(DEVICE)

        full_p = to_patches(data)
        o_mean = full_p.mean(dim=(1, 2))
        o_std = full_p.std(dim=(1, 2)) + EPS

        context = data[:, :C]
        ctx_p = to_patches(context)
        c_mean = ctx_p.mean(dim=(1, 2))
        c_std = ctx_p.std(dim=(1, 2)) + EPS

        e_leak = last_patch_embeddings(model, context, (o_mean, o_std))
        e_nolk = last_patch_embeddings(model, context, (c_mean, c_std))
        leak_all.append(e_leak.float().cpu())
        nolk_all.append(e_nolk.float().cpu())

    return torch.cat(leak_all), torch.cat(nolk_all)


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

    out = "embedding_regime_divergence_gift.npz"
    if os.path.exists(out):
        print(f"loading existing results -> {out}")
        results = dict(np.load(out))
        for strat, asinh, _ in MODELS:
            label = f"{strat}{'+asinh' if asinh else ''}"
            obs = float(results[f"w2_{label}"])
            null = results[f"w2null_{label}"]
            print(f"{label:<14} W2={obs:.4f}  "
                  f"(null {null.mean():.4f} +- {null.std():.4f})")
    else:
        print(f"running experiment -> {out}")
        print(f"device: {DEVICE}")
        print(f"seq={SEQ_PATCHES}p  horizon={HORIZON}  patch_len={PATCH_LEN}\n")

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
        results = {}

        for strat, asinh, pt in MODELS:
            cfg = PatchFMConfig(normalization_strategy=strat, use_asinh=asinh,
                                prefix_tokens=(pt or 4), compile=False)
            label = f"{strat}{'+asinh' if asinh else ''}"
            model = get_model(cfg, train_cfg, eval_cfg).to(DEVICE).eval()
            e_leak, e_nolk = collect_clouds(model, loader)
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            # per-model standardization with pooled stats -> metric lives on
            # a common scale, comparable across latent spaces
            pooled = torch.cat([e_leak, e_nolk]).double()
            m, s = pooled.mean(0), pooled.std(0) + EPS
            A = ((e_leak.double() - m) / s).numpy()
            B = ((e_nolk.double() - m) / s).numpy()

            obs = exact_w2(A, B)
            null = permutation_null(A, B)
            results[f"w2_{label}"] = obs
            results[f"w2null_{label}"] = null
            print(f"{label:<14} W2={obs:.4f}  "
                  f"(null {null.mean():.4f} +- {null.std():.4f})")

        np.savez(out, **results)
        print(f"\nresults saved -> {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [f"{s}{'+asinh' if a else ''}" for s, a, _ in MODELS]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, label in enumerate(labels):
            disp, color = LABEL_STYLE.get(label, (label, "black"))
            obs = float(results[f"w2_{label}"])
            null = results[f"w2null_{label}"]
            ax.bar(i, obs, width=0.6, color=color, zorder=3)
            # permutation null: mean tick +- 2 sd, the finite-sample floor
            ax.errorbar(i, null.mean(), yerr=2 * null.std(), color="0.15",
                        fmt="_", ms=18, capsize=5, lw=1.2, zorder=4)
            ax.annotate(f"{obs:.3g}", (i, obs), ha="center", va="bottom",
                        fontsize=9, xytext=(0, 5), textcoords="offset points")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels([LABEL_STYLE.get(l, (l,))[0] for l in labels],
                           fontsize=10, rotation=12)
        ax.set_ylabel(r"exact $W_2$", fontsize=12)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.errorbar([], [], yerr=[], color="0.15", fmt="_", capsize=5,
                    label="permutation null (mean $\\pm$ 2 sd)")
        ax.legend(loc="upper left", frameon=False, fontsize=9)

        ax.set_title("Exact $W_2$ between the leak / no-leak distributions "
                     "of the last context-patch hidden state\n"
                     "(clouds standardized per model; read each bar against "
                     "its finite-sample null)", fontsize=11)
        fig.tight_layout()
        out = "embedding_regime_divergence_gift.pdf"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"saved plot -> {out}")
    except Exception as e:
        print(f"(plot skipped: {e})")

if __name__ == "__main__":
    main()
