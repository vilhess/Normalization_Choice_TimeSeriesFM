import os
import time
import argparse
import torch
import numpy as np

import sys
sys.path.append('../')
from configs import PatchFMConfig, TrainConfig
from utils import get_model_name
from model.inference.modules import PatchFM as PatchFM_NoCache
from model.inference.modules_kvcache import PatchFM as PatchFM_KVCache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATCH_LEN = PatchFMConfig().patch_len
CTX_PATCHES = 31                  # context length, in patches
SEED = 0

MODELS = [
    ("vanilla", False, None),
    ("vanilla", True,  None),
    ("prefix",  False, 4),
    ("prefix",  True,  4),
    ("causal",  False, None),
    ("causal",  True,  None),
]

CACHED_STRATEGIES = ("causal",)

LABEL_STYLE = {
    "vanilla":       r"RevIN",
    "vanilla+asinh": r"RevIN+asinh",
    "prefix":        r"Prefix@k",
    "prefix+asinh":  r"Prefix@k+asinh",
    "causal":        r"Causal",
    "causal+asinh":  r"Causal+asinh",
}


def load_model(strat, asinh, pt, kv, train_cfg):
    """The same checkpoint loaded into the cached or uncached PatchFM."""
    cfg = PatchFMConfig(normalization_strategy=strat, use_asinh=asinh,
                        prefix_tokens=(pt or 4), compile=False)
    cls = PatchFM_KVCache if kv else PatchFM_NoCache
    model = cls(
        normalization_strategy=cfg.normalization_strategy,
        patch_len=cfg.patch_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers_encoder=cfg.n_layers_encoder,
        use_asinh=cfg.use_asinh,
    )
    ckpt_path = os.path.join(train_cfg.checkpoint_path, get_model_name(cfg),
                             "patchfm-epoch=18---step-step=285000.ckpt")
    state_dict = torch.load(ckpt_path, map_location="cpu",
                            weights_only=False)["state_dict"]
    for k in list(state_dict.keys()):
        if k.startswith('model.'):
            state_dict[k[len('model.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval()


def bench(fn, n_warmup, n_repeat):
    """Wall-clock times (s) of n_repeat runs after n_warmup warmups, and the
    peak CUDA memory (bytes) across the timed runs."""
    for _ in range(n_warmup):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    mem = torch.cuda.max_memory_allocated() if DEVICE == "cuda" else 0
    return np.array(times), mem


def med_iqr(a):
    return np.median(a), np.percentile(a, 75) - np.percentile(a, 25)


def parse_args():
    p = argparse.ArgumentParser(
        description="Inference speed, with KV-cache reuse where valid.")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 8],
                   help="forecast horizons, in patches")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 128],
                   help="batch sizes (latency at 1, throughput at large)")
    p.add_argument("--warmup", type=int, default=3,
                   help="warmup runs before timing")
    p.add_argument("--repeat", type=int, default=20,
                   help="timed runs per measurement")
    return p.parse_args()


def main():

    args = parse_args()

    out = "speed_comparisons.npz"
    if os.path.exists(out):
        print(f"loading existing results -> {out}")
        results = dict(np.load(out))
    else:
        print(f"running benchmark -> {out}")
        print(f"device: {DEVICE}")
        print(f"ctx={CTX_PATCHES}p  patch_len={PATCH_LEN}  "
              f"horizons={args.horizons}p  batches={args.batches}  "
              f"repeats={args.repeat}\n")

        train_cfg = TrainConfig(checkpoint_path="../ckpts")
        g = torch.Generator().manual_seed(SEED)
        inputs = {B: torch.randn(B, CTX_PATCHES * PATCH_LEN,
                                 generator=g).to(DEVICE)
                  for B in args.batches}
        results = {"batches": np.array(args.batches),
                   "horizons": np.array(args.horizons)}

        for strat, asinh, pt in MODELS:
            label = f"{strat}{'+asinh' if asinh else ''}"
            variants = ((False, True) if strat in CACHED_STRATEGIES
                        else (False,))
            preds = {}                       # (variant, H) -> median forecast
            for kv in variants:
                variant = "kvcache" if kv else "nocache"
                model = load_model(strat, asinh, pt, kv, train_cfg)
                for B in args.batches:
                    x = inputs[B]
                    for H in args.horizons:
                        T = H * PATCH_LEN
                        # both classes expose the same forecast() interface;
                        # only the caching differs
                        fn = lambda: model.forecast(x, target_len=T)
                        times, mem = bench(fn, args.warmup, args.repeat)
                        key = f"{label}_{variant}_B{B}_H{H}"
                        results[f"lat_{key}"] = times
                        results[f"mem_{key}"] = mem
                        if B == args.batches[0]:
                            med, _ = model.forecast(x, target_len=T)
                            preds[(variant, H)] = med.float().cpu()
                        m, i = med_iqr(times)
                        print(f"{label:<14} {variant:<8} B={B:<4} H={H}p  "
                              f"latency={m*1e3:8.1f} +- {i*1e3:5.1f} ms  "
                              f"mem={mem/2**20:7.1f} MiB")
                del model
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            # sanity check of the cache exactness (cached strategies only)
            if len(variants) == 2:
                for H in args.horizons:
                    dev = (preds[("kvcache", H)]
                           - preds[("nocache", H)]).abs().max().item()
                    results[f"dev_{label}_H{H}"] = dev
                    print(f"{label:<14} max |median_kv - median_nocache| "
                          f"(H={H}p): {dev:.2e}")
            print()

        np.savez(out, **results)
        print(f"results saved -> {out}")

    # ---- summary table ------------------------------------------------------
    batches = results["batches"].tolist()
    horizons = results["horizons"].tolist()
    print(f"\n{'strategy':<15} {'B':>4} {'H':>3} "
          f"{'latency no-cache (ms)':>22} {'latency KV (ms)':>18} "
          f"{'thr no-cache (ser/s)':>21} {'thr KV (ser/s)':>17}")
    for strat, asinh, pt in MODELS:
        label = f"{strat}{'+asinh' if asinh else ''}"
        for B in batches:
            for H in horizons:
                cells = []
                for variant in ("nocache", "kvcache"):
                    key = f"lat_{label}_{variant}_B{B}_H{H}"
                    if key in results:
                        m, i = med_iqr(results[key])
                        cells.append(f"{m*1e3:.1f} +- {i*1e3:.1f}")
                        mt, it = med_iqr(B / results[key])
                        cells.append(f"{mt:.1f} +- {it:.1f}")
                    else:
                        cells.append("--")
                        cells.append("--")
                lat_nc, thr_nc, lat_kv, thr_kv = cells
                print(f"{LABEL_STYLE.get(label, label):<15} {B:>4} {H:>3} "
                      f"{lat_nc:>22} {lat_kv:>18} {thr_nc:>21} {thr_kv:>17}")


if __name__ == "__main__":
    main()
