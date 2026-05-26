import sys
import time

import datasets
import fev
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from pypots.imputation import Lerp

sys.path.append("..")  

from utils import get_model_name
from model.inference import get_model
from configs import PatchFMConfig, EvalConfig, TrainConfig

datasets.disable_progress_bars()

def batchify(lst: list, batch_size: int = 32, max_context_length: int = 1024):
    """Convert list into batches of desired size.
    If some elements have incompatible shapes, yield them individually.
    """
    for i in range(0, len(lst), batch_size):
        batch = lst[i : i + batch_size]
        batch = [x[-max_context_length:] for x in batch]

        try:
            # Try to stack into one tensor of shape [B, T, ...]
            batch_tensor = torch.stack(batch)
            yield batch_tensor
        except RuntimeError:
            # If stacking fails (different lengths or shapes), yield one by one
            print("Warning: yielding batch elements one by one due to incompatible shapes.")
            for x in batch:
                yield x.unsqueeze(0)  # keep batch dimension = 1

def predict_with_model(
    model,
    task: fev.Task,
    max_context_length: int = 1024,
    batch_size: int = 1,
    device: torch.device = torch.device("cpu")
) -> tuple[list[datasets.DatasetDict], float, dict]:

    inference_time = 0.0
    predictions_per_window = []
    for window in task.iter_windows(trust_remote_code=True):
        past_data, _ = fev.convert_input_data(window, adapter="datasets", as_univariate=True)
        past_data = past_data.with_format("torch").cast_column("target", datasets.Sequence(datasets.Value("float32")))
        loaded_targets = past_data["target"]
        
        start_time = time.monotonic()

        all_preds, all_quantiles = [], []
        for batch in batchify(loaded_targets, batch_size=batch_size, max_context_length=max_context_length):
            
            batch = batch.to(device)

            # preprocessing the context to be a multiple of patch_len=32, removing the first values if necessary
            max_seq_len=1024
            seq_len = batch.shape[1]
            if seq_len > max_seq_len:
                #print(f"Warning: sequence length {seq_len} exceeds max_seq_len {max_seq_len}. Truncating to the last {max_seq_len} values.")
                batch = batch[:, -max_seq_len:]
            if seq_len < model.patch_len:
                #print(f"Warning: sequence length {seq_len} is shorter than patch_len {model.patch_len}. Padding with the first value on the left to reach patch_len.")
                pad_len = model.patch_len - seq_len
                pad = batch[:, :1].repeat(1, pad_len)  # repeat first value
                batch = torch.cat([pad, batch], dim=1)
                
            if seq_len % model.patch_len != 0:
                #print(f"Warning: sequence length {seq_len} is not a multiple of patch_len {model.patch_len}. Truncating to the last {seq_len - (seq_len % model.patch_len)} values.")
                batch = batch[:, -(seq_len - (seq_len % model.patch_len)):]   
            if torch.isnan(batch).any():
                print("Warning: NaN values found in the batch. Replacing NaNs with interpolated values.")
                dtype = batch.dtype
                imputer = Lerp()
                batch = imputer.predict({"X": batch.cpu().numpy()[:, :, np.newaxis]})["imputation"].squeeze(-1)
                batch = torch.from_numpy(batch).to(device)
                batch = batch.to(dtype)         

            pred, quantiles = model.forecast(
                batch, target_len=task.horizon
            )
            requested_quantiles = task.quantile_levels
            quantiles_dic = {0.1:0, 0.2:1, 0.3:2, 0.4:3, 0.5:4, 0.6:5, 0.7:6, 0.8:7, 0.9:8}
            quantiles = quantiles[:, :, [quantiles_dic[q] for q in requested_quantiles]]
            all_preds.append(pred.cpu())
            all_quantiles.append(quantiles.cpu())
        pred = torch.cat(all_preds, dim=0).numpy()
        quantiles = torch.cat(all_quantiles, dim=0).numpy()

        inference_time += time.monotonic() - start_time

        predictions_dict = {"predictions": pred}
        for idx, level in enumerate(task.quantile_levels):
            predictions_dict[str(level)] = quantiles[:, :, idx]

        predictions_per_window.append(
            fev.combine_univariate_predictions_to_multivariate(
                datasets.Dataset.from_dict(predictions_dict), target_columns=task.target_columns
            )
        )

    return predictions_per_window, inference_time, {}


if __name__ == "__main__":

    for norm_strategy in ["causal", "vanilla", "prefix"]:
        for use_asinh in [True, False]:

            num_tasks = None 

            eval_cfg = EvalConfig(load_epoch=18)
            train_cfg = TrainConfig(checkpoint_path="../ckpts/")
            model_cfg = PatchFMConfig(normalization_strategy=norm_strategy, use_asinh=use_asinh, prefix_tokens=4)
            model_name = get_model_name(model_cfg)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = get_model(model_cfg, train_cfg, eval_cfg)
            model = model.to(device)
            if model_cfg.compile:
                print("Compiling model...")
                model = torch.compile(model)
                print("Model compiled.")

            benchmark = fev.Benchmark.from_yaml(
                "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/fev_bench/tasks.yaml"
            )
            summaries = []
            for task in tqdm(benchmark.tasks[:num_tasks], desc="Evaluating tasks"):
                predictions, inference_time, extra_info = predict_with_model(model, task, device=device)
                evaluation_summary = task.evaluation_summary(
                    predictions,
                    model_name=model_name,
                    inference_time_s=inference_time,
                    extra_info=extra_info,
                )
                summaries.append(evaluation_summary)
            # Show and save the results
            summary_df = pd.DataFrame(summaries)
            print(summary_df)
            summary_df.to_csv(f"results/{model_name}.csv", index=False)