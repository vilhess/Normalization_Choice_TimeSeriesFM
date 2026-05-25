import torch

from configs import DSPathsConfig
from dataset.chronosdata import ChronosDataset, ChronosDataset_mmap
from dataset.gift import GiftEvalDataset
from dataset.mixup import InnerMixUP, InterMixup


def get_dataset(dspaths_cfg: DSPathsConfig, seq_len=1024, normalize=True):
    gift_trainset = GiftEvalDataset(
        path=dspaths_cfg.giftpretrain_path, input_len=seq_len, min_stride=32, max_samples=1000, normalize=normalize
    )
    kernel_synth = ChronosDataset(file_path=dspaths_cfg.chronos_kernel_synth_path, normalize=normalize)
    tsmixup = ChronosDataset_mmap(
        file_path=dspaths_cfg.chronos_tsmixup_path,
        file_shape_path=dspaths_cfg.chronos_tsmixup_shape_path,
        normalize=normalize
    )
    mixup_1 = InnerMixUP(kernel_synth, K=4, alpha=1.5, n_samples=200_000, normalize=normalize)
    mixup_2 = InnerMixUP(gift_trainset, K=4, alpha=1.5, n_samples=200_000, normalize=normalize)
    mixup_3 = InterMixup(
        [tsmixup, gift_trainset], K=4, alpha=1.5, n_samples=200_000, normalize=normalize
    )
    mixup_4 = InterMixup(
        [kernel_synth, gift_trainset], K=4, alpha=1.5, n_samples=200_000, normalize=normalize
    )
    mixup_5 = InterMixup(
        [tsmixup, kernel_synth], K=4, alpha=1.5, n_samples=200_000, normalize=normalize
    )

    return torch.utils.data.ConcatDataset(
        [
            gift_trainset,
            kernel_synth,
            tsmixup,
            mixup_1,
            mixup_2,
            mixup_3,
            mixup_4,
            mixup_5,
        ]
    )

def get_dataset_test(dspaths_cfg: DSPathsConfig, seq_len=1024, normalize=True):
    gift_trainset = GiftEvalDataset(
        path=dspaths_cfg.gifteval_path, input_len=seq_len, min_stride=32, max_samples=1000, normalize=normalize
    )
    kernel_synth = ChronosDataset(file_path=dspaths_cfg.chronos_kernel_synth_path, normalize=normalize)

    mixup_1 = InnerMixUP(kernel_synth, K=4, alpha=1.5, n_samples=200_000, normalize=normalize)
    mixup_2 = InnerMixUP(gift_trainset, K=4, alpha=1.5, n_samples=200_000, normalize=normalize)


    return torch.utils.data.ConcatDataset(
        [
            gift_trainset,
            kernel_synth,
            mixup_1,
            mixup_2,
        ]
    )

