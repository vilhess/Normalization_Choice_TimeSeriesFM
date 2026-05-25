from dataclasses import asdict, dataclass, field


@dataclass
class DSPathsConfig:
    # Training data paths
    giftpretrain_path: str = "path/to/giftpretrain/"
    chronos_kernel_synth_path: str = "path/to/training_corpus_kernel_synth_1m.npz"
    chronos_tsmixup_path: str = "path/to/training_corpus_tsmixup_1m.npy"
    chronos_tsmixup_shape_path: str = "path/to/training_corpus_tsmixup_1m_shape.npy"

    # Evaluation data paths
    gifteval_path: str = "path/to/gifteval/"

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def to_dict(self):
        return asdict(self)