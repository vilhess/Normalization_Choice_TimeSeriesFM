import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
import lightning as L
from lightning.pytorch.utilities import grad_norm


class OptimalRevIN(nn.Module):
    def __init__(self, eps=1e-5, use_asinh=True):
        super().__init__()
        self.eps = eps
        self.cached_mean = None
        self.cached_std = None
        self.asinh = use_asinh

    def forward(self, x, mode: str, ctx_size_per_sample=None):
        assert x.dim() == 3, "Input tensor must be (batch, n_patches, patch_len)"

        if mode == "norm":
            mean, std = self._get_statistics(x, ctx_size_per_sample)
            self.cached_mean, self.cached_std = mean.detach(), std.detach()
            out = (x - mean) / std
            if self.asinh:
                out = torch.asinh(out)

        elif mode == "denorm":
            assert (
                self.cached_mean is not None and self.cached_std is not None
            ), "Call forward(..., 'norm') before 'denorm'"
            if self.asinh:
                x = torch.sinh(x)
            out = x * self.cached_std + self.cached_mean

        else:
            raise NotImplementedError(f"Mode '{mode}' not implemented.")
        return out

    def _get_statistics(self, x, ctx_size_per_sample):
        n_patches = x.size(1)
        mask = torch.arange(n_patches, device=x.device)[None, :] < ctx_size_per_sample[:, None]
        mask = mask[..., None]
        mean = (x * mask).sum(dim=(1, 2), keepdim=True) / (ctx_size_per_sample[:, None, None]*x.size(2))
        var_per_signal = (((x - mean) * mask) ** 2).sum(dim=(1, 2), keepdim=True) / (ctx_size_per_sample[:, None, None]*x.size(2))
        std = torch.sqrt(var_per_signal + 1e-8)
        return mean, std

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden_layer = nn.Linear(in_dim, hid_dim)
        self.output_layer = nn.Linear(hid_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        hid = self.act(self.hidden_layer(x))
        out = self.output_layer(hid)
        res = self.residual_layer(x)
        out = out + res
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert (
            d_model % n_heads == 0
        ), f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = dropout

        self.head_dim = d_model // n_heads
        self.n_heads = n_heads

        self.rope = RotaryEmbedding(dim=self.head_dim // 2)

    def forward(self, q):
        bs, context, dim = q.size()

        k = q
        v = q

        q = self.WQ(q).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.WK(k).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.WV(v).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rope.rotate_queries_or_keys(q)
        k = self.rope.rotate_queries_or_keys(k)

        values = nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )

        values = values.transpose(1, 2).reshape(bs, -1, dim)
        values = self.out_proj(values)
        return values


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1, multiple_of=256):
        super().__init__()

        hidden_dim = d_model * 4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

        self.act = nn.SiLU()
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w2(self.act(self.w1(x)) * self.w3(x))
        return self.dp(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, dropout=dropout)

    def forward(self, x):
        out_attn = self.attn(self.ln1((x)))
        x = x + out_attn
        out = x + self.ff(self.ln2(x))
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model, n_heads=n_heads, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class PatchFM(nn.Module):
    def __init__(
        self,
        normalization_strategy,
        prefix_tokens,
        use_asinh,
        patch_len,
        d_model,
        n_heads,
        n_layers_encoder,
        dropout=0.1,
        quantiles=None,
    ):
        super().__init__()

        self.patch_len = patch_len

        self.quantiles = (
            quantiles
            if quantiles is not None
            else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        self.n_quantiles = len(self.quantiles)

        self.revin = OptimalRevIN(use_asinh=use_asinh)

        self.proj_embedding = ResidualBlock(
            in_dim=patch_len, hid_dim=2 * patch_len, out_dim=d_model, dropout=dropout
        )
        self.dp = nn.Dropout(dropout)
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers_encoder,
            dropout=dropout,
        )

        self.proj_output = ResidualBlock(
            in_dim=d_model,
            hid_dim=2 * d_model,
            out_dim=patch_len * self.n_quantiles,
            dropout=dropout,
        )

        self.init_weights()

        print(f"Using normalization strategy: {normalization_strategy} ; use_asinh: {use_asinh}")
        print(f"Model parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"Number of layers: {n_layers_encoder}, d_model: {d_model}, n_heads: {n_heads}, patch_len: {patch_len}")

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.fill_(0.0)
                m.weight.data.fill_(1.0)

    def forward(self, x):
        bs, ws = x.size()

        x = rearrange(
            x, "b (pn pl) -> b pn pl", pl=self.patch_len
        )  # Reshape to (bs, patch_num, patch_len)

        n_patches = x.size(1)
        ctx_size_per_sample = torch.randint(1, n_patches-1, (bs,), device=x.device)

        target = x[torch.arange(bs), ctx_size_per_sample+1].clone().detach().unsqueeze(1)  # (bs, 1, patch_len)
        x = self.revin(x, mode="norm", ctx_size_per_sample=ctx_size_per_sample)

        x = self.proj_embedding(x)  # bs, pn, d_model
        x = self.dp(x)
        x = self.transformer_encoder(x)  # bs, pn, d_model

        forecasting = self.proj_output(x)  # bs, pn, patch_len * n_quantiles

        forecasting = self.revin(forecasting, mode="denorm")

        forecasting = rearrange(
            forecasting,
            "b pn (pl q) -> b pn pl q",
            pl=self.patch_len,
            q=self.n_quantiles,
        )  # Reshape to (bs, patch_num, pn, n_quantiles)
        forecasting = forecasting[torch.arange(bs), ctx_size_per_sample].clone().detach()  # (bs, patch_len, n_quantiles)
        forecasting = forecasting.unsqueeze(1)  # (bs, 1, patch_len, n_quantiles)

        return forecasting, target

class MultiQuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()

        if not isinstance(quantiles, torch.Tensor):
            quantiles = torch.tensor(quantiles)

        assert all(
            0 < q < 1 for q in quantiles
        ), "Quantiles must be in the range (0, 1)"
        self.quantiles = quantiles

    def forward(self, pred, target):
        assert pred.shape[-1] == len(self.quantiles)
        assert target.shape[1] == pred.shape[1]  # n_patches
        assert target.shape[2] == pred.shape[2]  # patch_len
        self.quantiles = self.quantiles.to(pred.device)
        target = target.unsqueeze(-1)
        errors = target - pred
        losses = torch.max((self.quantiles - 1) * errors, self.quantiles * errors)
        return losses.mean()




class PatchFMLit(L.LightningModule):
    def __init__(self, model_config, train_config):
        super().__init__()

        self.model = PatchFM(
            normalization_strategy=model_config.normalization_strategy,
            prefix_tokens=model_config.prefix_tokens,
            use_asinh=model_config.use_asinh,
            patch_len=model_config.patch_len,
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            n_layers_encoder=model_config.n_layers_encoder,
            dropout=train_config.dropout,
            quantiles=model_config.quantiles,
        )
        self.criterion = MultiQuantileLoss(self.model.quantiles)

        config = {**model_config.__dict__, **train_config.__dict__}
        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        x = batch

        ###
        #  Data augmentation: random sign flip and time flip
        sign_flip = torch.where(
            torch.randn(x.size(0), 1, device=x.device) > 0, 1.0, -1.0
        )
        x = sign_flip * x
        time_flip = torch.randn((x.size(0)), device=x.device) > 0.0
        x[time_flip] = x[time_flip].flip(dims=[1])
        ###

        prediction, y = self.model(x)
        loss = self.criterion(prediction, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):

        optimizer = optim.AdamW(
            self.parameters(), lr=self.hparams.start_lr, weight_decay=0.01
        )

        div_factor = self.hparams.max_lr / self.hparams.start_lr
        final_div_factor = self.hparams.start_lr / self.hparams.lower_lr
        pct_start = self.hparams.reach_max / self.hparams.iter_cycle
        onecycle = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.max_lr,
            total_steps=self.hparams.iter_cycle,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )
        constant = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=final_div_factor, total_iters=1e8
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[onecycle, constant],
            milestones=[self.hparams.iter_cycle],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)