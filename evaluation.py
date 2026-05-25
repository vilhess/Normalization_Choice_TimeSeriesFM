import torch
from lightning.pytorch import seed_everything
import tqdm
import os
import numpy as np

from model.inference import get_model
from dataset import GiftEvalDataset
from configs import PatchFMConfig, EvalConfig, TrainConfig, DSPathsConfig
from scorer import MetricScorer, save_results_npz
from utils import get_model_name

def main():

    eval_cfg = EvalConfig(load_epoch=18)
    train_cfg = TrainConfig()
    model_cfg = PatchFMConfig()
    dspaths_cfg = DSPathsConfig()
    
    model_name = get_model_name(model_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_cfg, train_cfg, eval_cfg)
    model = model.to(device)
    if model_cfg.compile:
        print("Compiling model...")
        model = torch.compile(model)
        print("Model compiled.")

    for context_len in eval_cfg.context_lens:
        print(f"Evaluating context length: {context_len}")

        seed_everything(0, workers=True)
        
        print("Loading dataset...")
        dataset = GiftEvalDataset(
            path=dspaths_cfg.gifteval_path,
            input_len=context_len+eval_cfg.max_target_len, 
            normalize=True
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=eval_cfg.batch_size,
            shuffle=False,
            num_workers=eval_cfg.num_workers,
            pin_memory=True,
        )
        print("Dataset loaded")

        scorer = MetricScorer(
            max_pred_len=eval_cfg.max_target_len,
            patch_len=model_cfg.patch_len,
        )
        print("Starting evaluation...")

        with torch.inference_mode():
            for i, batch in enumerate(tqdm.tqdm(dataloader)):

                batch_signals = batch.to(device)
                contexts = batch_signals[:, :context_len]
                targets = batch_signals[:, context_len:]
                assert targets.shape[1]==eval_cfg.max_target_len, f"Expected target length {eval_cfg.max_target_len}, but got {targets.shape[1]}"

                prediction = model.forecast(contexts, target_len=eval_cfg.max_target_len) # ((bs, target_len), (bs, target_len, 9))
                scorer.update(prediction, targets, contexts)
        results = scorer.compute()
        scorer.reset()

        if not os.path.exists(eval_cfg.results_path):
            os.makedirs(eval_cfg.results_path+"/raw")
            os.makedirs(eval_cfg.results_path+"/processed")

        current_raw_results_path = os.path.join(
            eval_cfg.results_path, "raw", eval_cfg.load_epoch, model_name, f"{context_len}.npz"
        )
        os.makedirs(os.path.dirname(current_raw_results_path), exist_ok=True)
        save_results_npz(results, current_raw_results_path)
        print(f"Results saved to {current_raw_results_path}")

        print("Aggregating results by window...")

        windows = [0] + dataset.n_window_list
        aggregated_results = {}
        for key in results.keys():
            values = results[key]

            assert len(values)==windows[-1], f"Length mismatch for {key}: {len(values)} vs {windows[-1]}"

            aggregated_means = []

            for i in range(len(windows) - 1):
                slice_data = values[windows[i] : windows[i + 1]]
                # ✅ Check for empty slice BEFORE taking mean
                if len(slice_data) == 0:
                    print(len(values), windows[-1])
                    print(f"⚠️  Empty slice at window {i} for {key}")
                    aggregated_means.append(np.nan)  # or 0, or skip

                elif np.all(np.isnan(slice_data)):
                    pass  # do not append anything if all values are NaN (happened for MASE and SQL when context is constant)

                else:
                    aggregated_means.append(np.nanmean(slice_data))

            aggregated_results[key] = np.array(aggregated_means)
        
        current_processed_results_path = os.path.join(
            eval_cfg.results_path, "processed", eval_cfg.load_epoch, model_name, f"{context_len}.npz"
        )
        os.makedirs(os.path.dirname(current_processed_results_path), exist_ok=True)
        save_results_npz(aggregated_results, current_processed_results_path)
        print(f"Aggregated results saved to {current_processed_results_path}")

        print(f"Evaluation for context length {context_len} completed.\n")

if __name__ == "__main__":
    main()