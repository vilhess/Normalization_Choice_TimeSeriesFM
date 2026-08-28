from dataclasses import asdict, dataclass, field


@dataclass
class PatchFMConfig:
    # approximately 15M parameters
    normalization_strategy: str = "causal"
    prefix_tokens: int = 8
    use_asinh: bool = True
    max_seq_len: int = 1024
    patch_len: int = 16
    d_model: int = 512 
    n_heads: int = 8
    n_layers_encoder: int = 4
    quantiles: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    compile: bool = True

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def to_dict(self):
        return asdict(self)

