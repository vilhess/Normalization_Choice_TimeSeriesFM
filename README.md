# Does Normalization Choice Matter for Causal Large Time-Series Models?

## Table 1: Figures/Tables and Associated Scripts

| Figure / Table | Script or Notebook | Working Directory |
|---|---|---|
| Figure 2/3 | `notebooks/traininglosses.ipynb` | repo root |
| Table 2 | `notebooks/ranking.ipynb` | repo root |
| Figure 4 | `notebooks/skillscores.ipynb` | repo root |
| Figure 5 | `notebooks/traininglosses.ipynb` | repo root |
| Figure 6 | `notebooks/skillscores.ipynb` | repo root |
| Figure 7 | `python leak_scaling_gift.py` | `leakage_measurement/` |
| Figure 8 | `python embedding_distance_gift.py` | `leakage_measurement/` |
| Table 3 | `python embedding_regime_divergence_gift.py` | `leakage_measurement/` |
| Figure 9/10 | `python delta_mase_various_k.py` | `robustness_prefix/` |
| Figure 11 | `python embedding_distance_various_k.py` | `robustness_prefix/` |
| Figure 12 | `python embedding_regime_divergence_various_k.py` | `robustness_prefix/` |
| Table 4 | `notebooks/ranking.ipynb` | repo root |
| Figure 13 | `traininglosses.ipynb` | `small_model_results/` or `mid_model_results/` |
| Figure 14 | `skillscores.ipynb` | `small_model_results/` or `mid_model_results/` |
| Table 6 | `notebooks/ranking.ipynb` | repo root |
| Table 7 | `python speed_comparisons.py` | `kv_cache_measurements/` |
| Figure 15 | `notebooks/ranking.ipynb` | repo root |
| Figure 16/17 | `python tsne_leakage.py` | `leakage_measurement/` |
| Figure 18/19 | `python delta_mase_various_k.py` | `robustness_prefix/` |
| Figure 20/21 | `python delta_mase_various_k.py` | `robustness_prefix/` |

> `delta_mase_various_k.py` takes a `--perturb {mean,std,both}` flag and produces a different figure pair for each value — run it once per value to regenerate Figures 9/10, 18/19, and 20/21. It also accepts `--prefix-only`, which restricts the plot to prefix-based normalization strategies (all models are still computed and saved to the `.npz` regardless).

This repository contains the official code for the paper:

**Does Normalization Choice Matter for Causal Large Time-Series Models?**

## Repository Structure

- **`results/<epoch>/`**  
  Contains the experimental results reported in the paper, organized into separate subfolders for each experiment.  
  `<epoch>` refers to the training epoch of the model used for evaluation. In this repository, we provide results for epoch 18, corresponding to approximately 285k training steps. All results reported in the paper are obtained from models evaluated at this training stage.

- **`ckpts/`**  
  Stores model checkpoints saved at different training epochs. These checkpoints can be used to reproduce results at various stages of training and to analyze the effect of training duration on model performance.  
  Due to storage constraints, checkpoints are not currently included in the repository. They can be requested from the authors and may be released publicly on Hugging Face in the future.

- **`model/`**  
  Contains the implementation of the proposed model as well as the different normalization strategies evaluated in the paper.

- **`traininglosses/`**  
  Contains the training losses recorded during training for each model. These logs can be used to study training dynamics, convergence behavior, and optimization stability.

- **`notebooks/`**  
  Jupyter notebooks used to generate the figures, visualizations, and plots presented in the paper.

- **`configs/`**  
  Configuration files defining model architectures, training parameters, dataset paths, and evaluation settings.

- **`scorer.py`**  
  Implementation of the evaluation metrics used in the paper, including:
  - Scaled Quantile Loss (SQL)
  - Mean Absolute Scaled Error (MASE)
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)

- **`training.py`**  
  Implementation of the training loop, optimization procedures, and logging utilities used for model training.

- **`evaluation.py`**  
  Implementation of the evaluation pipeline used to assess model performance on the benchmark datasets.

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---
## Running Experiments

To train a model with a specific normalization strategy and evaluate it on the benchmark datasets:

1. Edit the `PatchFMConfig` class in `configs/model_config.py` and configure the following parameters:

```python
normalization_strategy: str = "prefix"
use_asinh: bool = True
prefix_tokens: int = 4
```

- `normalization_strategy` defines the normalization method used during training and inference.
- `prefix_tokens` specifies the number of prefix tokens used by the normalization strategy  (necessary if normalization_strategy is "prefix").
- `use_asinh` enables or disables the inverse hyperbolic sine transformation.

2. Launch the training and evaluation pipeline:

```bash
python training.py
python evaluation.py
```

The training script saves model checkpoints and send training logs to wandb, while the evaluation script computes the benchmark metrics reported in the paper.

---
## Download Dataset

Specify dataset paths in the `DSPathsConfig` class located in `configs/dspaths_config.py`:

```python
giftpretrain_path: str = "/path/to/giftpretrain/"
chronos_kernel_synth_path: str = "/path/to/chronos_kernel_synth.npz"
chronos_tsmixup_path: str = "/path/to/chronos_tsmixup.npy"
chronos_tsmixup_shape_path: str = "/path/to/chronos_tsmixup_shape.npy"
gifteval_path: str = "/path/to/gifteval/"
```

If the GIFT-based datasets are not available locally, they will be automatically downloaded from Hugging Face during training and evaluation.

The Chronos-based datasets must be downloaded manually. Scripts for generating and downloading them are available in `dataset/chronosdata.py`. Once prepared, place the files in the paths specified above.

---

### Autoregressive Inference with Quantile Forecasting ([Moirai 2.0](https://arxiv.org/pdf/2511.11698v1))

During autoregressive inference, the model generates forecasted values patch by patch. At each time step, the predicted patch is fed back into the model as input for the next step. This iterative process continues until the desired forecast horizon is reached.

When performing quantile forecasting, the situation becomes more complex. Instead of producing a single patch per step, the model outputs multiple patches corresponding to different quantiles (e.g., 0.1, 0.5, 0.9). Since the model expects a single patch for the next time step, it is not straightforward to feed all quantile predictions back into the model simultaneously.

A common workaround is to feed only the median prediction (the 0.5 quantile) back into the model at each step. While this approach preserves the autoregressive structure, it discards the uncertainty information captured by the other quantiles.

An alternative approach is **autoregressive multi-quantile decoding**, as proposed in [Moirai 2.0](https://arxiv.org/pdf/2511.11698v1). This method enables consistent autoregressive generation while preserving the full predictive distribution across quantiles. However, it is computationally more expensive than the median-only approach as it requires duplicating the context for each quantile.

The algorithm proceeds as follows:

1. **Initialization**  
   Start with the initial context window of observed data  
   **Shape:** `(BS × L)`  
   - `BS`: batch size  
   - `L`: context length  
   - `P`: patch size  
   - `Q`: number of quantiles  
   - `H`: forecast horizon  
   - `i=1`: current algorithm step

2. **First Quantile Prediction (Forward Pass)**  
   Predict the quantiles for the next patch using the current context.  
   **Output shape:** `(BS × P × Q)`

3. **Context Duplication**  
   For each predicted quantile, create a separate context by appending the corresponding predicted patch to the current context.  
   This increases the number of contexts by a factor of `Q` at each step.  
   **New context shape:** `(BS × Q × i(L + P))`

4. **Next Forward Pass**  
   For each duplicated context, predict the quantiles of the next patch.  
   **Output shape:** `(BS × Q × P × Q)`

5. **Quantile Collapse**  
   - Permute and reshape the predictions to aggregate all possible quantile paths:  
     **Intermediate shape:** `(BS × P × Q²)`  
   - Compute the quantiles across the `Q²` predictions to obtain the final quantile estimates for the next patch.  
     **Final shape:** `(BS × P × Q)`
   - Increment the step counter `i ← i + 1`.

6. **Iteration**  
   Repeat Steps 3–5 until the forecast horizon `H` is reached, i.e., until the total number of predicted time steps satisfies  
   `i × P ≥ H`.

This procedure preserves predictive uncertainty across quantiles while maintaining the autoregressive structure of the model. Although it is computationally more expensive than feeding only the median prediction (0.5 quantile) back into the model, it remains tractable in practice and enables consistent multi-quantile forecasting.

⚠️ **Warning**  
With this strategy, the median prediction (0.5 quantile) does **not necessarily** match the prediction obtained by autoregressively feeding only the median patch back into the model at each step.

This discrepancy arises because the *quantile collapse* step aggregates predictions across all possible quantile paths. As a result, the median is computed from the combined multi-path distribution rather than from a single deterministic trajectory, which can lead to different estimates compared to the single-path (median-only) autoregressive approach.

---