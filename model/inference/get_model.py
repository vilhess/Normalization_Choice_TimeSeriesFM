import os
import torch
from utils import get_model_name

def get_model(model_cfg, train_cfg, eval_cfg):

    if model_cfg.normalization_strategy in ["none"]:
        from model.inference.modules_kvcache import PatchFM
        model = PatchFM(
            normalization_strategy=model_cfg.normalization_strategy,
            d_model=model_cfg.d_model,
            patch_len=model_cfg.patch_len,
            n_heads=model_cfg.n_heads,
            n_layers_encoder=model_cfg.n_layers_encoder,
            use_asinh=model_cfg.use_asinh,
        )

    elif model_cfg.normalization_strategy in ["prefix", "vanilla", "optimal", "causal"]:
        from model.inference.modules import PatchFM
        model = PatchFM(
            normalization_strategy=model_cfg.normalization_strategy,
            d_model=model_cfg.d_model,
            patch_len=model_cfg.patch_len,
            n_heads=model_cfg.n_heads,
            n_layers_encoder=model_cfg.n_layers_encoder,
            use_asinh=model_cfg.use_asinh,
        )
    else:
        raise ValueError(f"Unknown normalization strategy: {model_cfg.normalization_strategy}")

    #ckpt_path = os.path.join(train_cfg.checkpoint_path, get_model_name(model_cfg), f"patchfm-epoch={eval_cfg.load_epoch}.ckpt")
    ckpt_path = os.path.join(train_cfg.checkpoint_path, get_model_name(model_cfg), f"patchfm-epoch=18---step-step=285000.ckpt")
    print(f"Loading model from {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    for k in list(state_dict.keys()):
        if k.startswith('model.'):
            state_dict[k[len('model.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    print(f"Model loaded successfully from {ckpt_path}.")
    
    return model