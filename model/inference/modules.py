import torch
import torch.nn as nn
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

from model.inference.revin import CausalRevIN, RevIN, NoRevIN


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
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
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert (
            d_model % n_heads == 0
        ), f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

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
            q, k, v, is_causal=True, dropout_p=0.0
        )

        values = values.transpose(1, 2).reshape(bs, -1, dim)
        values = self.out_proj(values)
        return values


class FeedForward(nn.Module):
    def __init__(self, d_model, multiple_of=256):
        super().__init__()

        hidden_dim = d_model * 4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

        self.act = nn.SiLU()

    def forward(self, x):
        x = self.w2(self.act(self.w1(x)) * self.w3(x))
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model)

    def forward(self, x):
        out_attn = self.attn(self.ln1((x)))
        x = x + out_attn
        out = x + self.ff(self.ln2(x))
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model, n_heads=n_heads
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
        use_asinh,
        patch_len,
        d_model,
        n_heads,
        n_layers_encoder,
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

        if normalization_strategy == "causal":
            self.revin = CausalRevIN(use_asinh=use_asinh)
        elif normalization_strategy in ["vanilla", "prefix", "optimal"]:
            self.revin = RevIN(use_asinh=use_asinh)
        elif normalization_strategy == "none":
            self.revin = NoRevIN()
        else:
            raise ValueError(
                f"Invalid normalization strategy: {normalization_strategy}"
            )

        self.proj_embedding = ResidualBlock(
            in_dim=patch_len, hid_dim=2 * patch_len, out_dim=d_model
        )

        self.transformer_encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers_encoder,
        )

        self.proj_output = ResidualBlock(
            in_dim=d_model,
            hid_dim=2 * d_model,
            out_dim=patch_len * self.n_quantiles,
        )

        print(f"Loading PatchFM with normalization strategy: {normalization_strategy} and use_asinh: {use_asinh}")

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
        if self.training:
            x_patch = x[:, 1:, :].clone().detach()
        x = self.revin(x, mode="norm")

        x = self.proj_embedding(x)  # bs, pn, d_model
        x = self.transformer_encoder(x)  # bs, pn, d_model

        forecasting = self.proj_output(x)  # bs, pn, patch_len * n_quantiles

        forecasting = self.revin(forecasting, mode="denorm")

        forecasting = rearrange(
            forecasting,
            "b pn (pl q) -> b pn pl q",
            pl=self.patch_len,
            q=self.n_quantiles,
        )  # Reshape to (bs, patch_len, n_quantiles)

        return forecasting

    @torch.inference_mode()
    def forecast(self, x, target_len=None):

        if target_len is None:
            target_len = self.patch_len

        assert x.ndim in (
            1,
            2,
        ), f"Input dimension must be 1D (time) or 2D (batch, time), got {x.ndim}D."
        bs, ws = x.size()

        context = x.clone()

        rollouts = -(-target_len // self.patch_len)  # ceil division
        predictions = []

        forecasting = self.forward(x)  # Get all quantiles for the initial context
        forecasting = forecasting[
            :, -1, :, :
        ]  # Keep only the last patch for autoregressive forecasting

        context_expanded = torch.repeat_interleave(
            context.unsqueeze(-1), repeats=self.n_quantiles, dim=-1
        )  # batch x ws x n_quantiles
        base_context_expanded = torch.cat(
            (context_expanded, forecasting), dim=1
        )  # batch x ws+patch_size x n_quantiles
        context_expanded = base_context_expanded.permute(0, 2, 1).reshape(
            bs * self.n_quantiles, base_context_expanded.size(1)
        )

        x = context_expanded
        q = torch.tensor(self.quantiles, device=x.device)

        predictions.append(forecasting)

        for _ in range(rollouts - 1):

            # Forward pass
            forecasting = self.forward(x)  # batch*n_quantiles x patch_num x patch_len x n_quantiles
            forecasting = forecasting[
                :, -1, :, :
            ]  # batch*n_quantiles x patch_len x n_quantiles

            forecasting = rearrange(
                forecasting, "(b q) pl h -> b q pl h", q=self.n_quantiles
            )
            forecasting = forecasting.permute(0, 2, 1, 3).flatten(
                start_dim=-2
            )  # batch x patch_len x n_quantiles**2
            forecasting = torch.quantile(
                forecasting, q, dim=-1
            )  # n_quantiles x batch x patch_len
            forecasting = forecasting.permute(
                1, 2, 0
            )  # batch x patch_len x n_quantiles

            base_context_expanded = torch.cat(
                (base_context_expanded, forecasting), dim=1
            )  # # batch x ws+iter*patch_size x n_quantiles
            context_expanded = base_context_expanded.permute(0, 2, 1).reshape(
                bs * self.n_quantiles, base_context_expanded.size(1)
            )

            x = context_expanded
            predictions.append(forecasting)

        pred_quantiles = torch.cat(predictions, dim=1)
        pred_quantiles = pred_quantiles[:, :target_len, :]
        pred_median = pred_quantiles[:, :, 4]

        return pred_median, pred_quantiles