def get_model(train_cfg, model_cfg):
    if "optimal" in model_cfg.normalization_strategy:
        print("Using optimal model variant")
        from model.training import PatchFMOptimalLit
        model = PatchFMOptimalLit(train_config=train_cfg, model_config=model_cfg)
    else:
        from model.training import PatchFMLit
        model = PatchFMLit(train_config=train_cfg, model_config=model_cfg)
    return model

def get_model_name(model_cfg):
    
    if model_cfg.normalization_strategy == "causal":
        if model_cfg.use_asinh:
            model_name = "causalsinh"
        else:
            model_name = "causal"
    elif model_cfg.normalization_strategy == "prefix":
        if model_cfg.use_asinh:
            model_name = f"prefix{model_cfg.prefix_tokens}sinh"
        else:
            model_name = f"prefix{model_cfg.prefix_tokens}"
    elif model_cfg.normalization_strategy == "vanilla":
        if model_cfg.use_asinh:
            model_name = "vanillasinh"
        else:
            model_name = "vanilla"
    elif model_cfg.normalization_strategy == "optimal":
        if model_cfg.use_asinh:
            model_name = "optimalsinh"
        else:
            model_name = "optimal"
    elif model_cfg.normalization_strategy == "none":
        model_name = "none"
    else:
        raise ValueError(
            f"Invalid normalization strategy: {model_cfg.normalization_strategy}"
        )
    return model_name