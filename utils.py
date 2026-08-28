def get_model(train_cfg, model_cfg):
    if "singlepatch" in model_cfg.normalization_strategy:
        print("Using singlepatch model variant")
        from model.training import PatchFMSinglePatchLit
        model = PatchFMSinglePatchLit(train_config=train_cfg, model_config=model_cfg)
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
    elif model_cfg.normalization_strategy == "singlepatch":
        if model_cfg.use_asinh:
            model_name = "singlepatchsinh"
        else:
            model_name = "singlepatch"
    elif model_cfg.normalization_strategy == "none":
        model_name = "none"
    elif model_cfg.normalization_strategy == "causalpatch":
        if model_cfg.use_asinh:
            model_name = "causalsinhpatch"
        else:
            model_name = "causalpatch"
    else:
        raise ValueError(
            f"Invalid normalization strategy: {model_cfg.normalization_strategy}"
        )
    return model_name