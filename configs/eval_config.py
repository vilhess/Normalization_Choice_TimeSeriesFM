from dataclasses import asdict, dataclass, field


@dataclass
class EvalConfig:
    max_target_len: int = 512
    context_lens: list[int] = field(
        default_factory=lambda: [64, 128, 256, 512, 1024]
    )
    batch_size: int = 256
    num_workers: int = 21
    pin_memory: bool = True
    results_path: str = "results/"
    load_epoch: int = 18

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def to_dict(self):
        return asdict(self)