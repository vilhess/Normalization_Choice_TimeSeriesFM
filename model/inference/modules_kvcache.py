import torch
import torch.nn as nn
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

class NoRevIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mode: str):
        return x
    
    def clear_cache(self):        
        pass

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


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, last=False):
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

        self.k_cache = None
        self.v_cache = None

        self.last = last

    def forward(self, q):
        bs, context, dim = q.size()
        offset = 0
        is_causal = True

        k = q
        v = q

        if self.last:
            q = q[:, -1:, :]
            is_causal = False
            offset += context - 1

        q = self.WQ(q).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.WK(k).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.WV(v).reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)

        if self.k_cache is not None and self.v_cache is not None:
            offset += self.k_cache.size(2)
            is_causal = False
            factor = q.size(0) // self.k_cache.size(0)
            self.k_cache = torch.repeat_interleave(self.k_cache, repeats=factor, dim=0)
            self.v_cache = torch.repeat_interleave(self.v_cache, repeats=factor, dim=0)
            k = torch.cat([self.k_cache, k], dim=2)
            v = torch.cat([self.v_cache, v], dim=2)

        self.k_cache = k
        self.v_cache = v

        q = self.rope.rotate_queries_or_keys(q, offset=offset)
        k = self.rope.rotate_queries_or_keys(k)

        values = nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, dropout_p=0.0
        )

        values = values.transpose(1, 2).reshape(bs, -1, dim)
        values = self.out_proj(values)
        return values

    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, last):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)

        self.attn = CausalMultiHeadAttention(
            d_model=d_model, n_heads=n_heads, last=last
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
                    d_model=d_model, n_heads=n_heads, last=False
                )
                for _ in range(n_layers - 1)
            ]
        )
        self.layers.append(
            TransformerEncoderLayer(
                d_model=d_model, n_heads=n_heads, last=True
            )
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
        patch_len,
        d_model,
        n_heads,
        n_layers_encoder,
        use_asinh,
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ):
        super().__init__()

        self.patch_len = patch_len
        self.quantiles = (
            quantiles
            if quantiles is not None
            else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        self.n_quantiles = len(self.quantiles)

        if normalization_strategy == "none":
            self.revin = NoRevIN()
        else:
            raise ValueError(f"Invalid normalization strategy: {normalization_strategy}")

        self.proj_embedding = ResidualBlock(
            in_dim=patch_len, hid_dim=2 * patch_len, out_dim=d_model
        )
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers_encoder
        )
        self.proj_output = ResidualBlock(
            in_dim=d_model, hid_dim=2 * d_model, out_dim=patch_len * self.n_quantiles
        )

    @torch.inference_mode()
    def forecast(self, x, target_len=None):
        q = torch.tensor(self.quantiles, device=x.device)

        if target_len is None:
            target_len = self.patch_len
        x = rearrange(x, "b (pn pl) -> b pn pl", pl=self.patch_len)

        rollouts = -(-target_len // self.patch_len)  # ceil division
        predictions = []

        # First Forward pass
        x = self.revin(x, mode="norm")
        x = self.proj_embedding(x)

        x = self.transformer_encoder(x)

        x = x[:, -1:, :]  # Keep only the last patch for autoregressive forecasting
        forecasting = self.proj_output(x)
        forecasting = self.revin(forecasting, mode="denorm")
        # Reshape to (bs, patch_num, patch_len, n_quantiles)
        forecasting = rearrange(
            forecasting, "b 1 (pl q) -> b 1 pl q", pl=self.patch_len, q=self.n_quantiles
        )
        predictions.append(forecasting[:, 0, :, :].detach())
        x = forecasting.permute(0, 3, 1, 2).reshape(
            forecasting.size(0) * self.n_quantiles, 1, self.patch_len
        )

        for _ in range(rollouts - 1):

            # Forward pass
            x = self.revin(x, mode="norm")
            x = self.proj_embedding(x)
            x = self.transformer_encoder(x)
            x = x[:, -1:, :]  # Keep only the last patch for autoregressive forecasting
            forecasting = self.proj_output(x)
            forecasting = self.revin(forecasting, mode="denorm")
            # Reshape to (bs, patch_num, patch_len, n_quantiles)
            forecasting = rearrange(
                forecasting,
                "b 1 (pl q) -> b 1 pl q",
                pl=self.patch_len,
                q=self.n_quantiles,
            )
            forecasting = rearrange(
                forecasting,
                "(b q) 1 pl h -> b q 1 pl h",
                b=forecasting.size(0) // self.n_quantiles,
                q=self.n_quantiles,
            ).permute(0, 2, 3, 1, 4)
            forecasting = torch.flatten(forecasting, start_dim=-2)

            forecasting = torch.quantile(forecasting, q=q, dim=-1)
            x = forecasting.permute(1, 0, 2, 3).reshape(-1, 1, self.patch_len)
            predictions.append(forecasting.permute(1, 2, 3, 0)[:, 0].detach())

        predictions = torch.cat(predictions, dim=1)
        predictions = predictions[:, :target_len]
        predictions_median = predictions[:, :, 4]

        self.clear_cache()
        return predictions_median, predictions

    def clear_cache(self):
        self.revin.clear_cache()
        for layer in self.transformer_encoder.layers:
            layer.attn.clear_cache()