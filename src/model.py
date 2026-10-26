from __future__ import annotations

import math

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        inverse_frequency = theta ** (
            -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = positions.float().unsqueeze(-1) * self.inverse_frequency
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos(), angles.sin()


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        position_mode: str,
        head_dim_override: int | None = None,
    ) -> None:
        super().__init__()
        if head_dim_override is None and d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim_override or d_model // n_heads
        self.value_dim = self.head_dim * n_heads
        self.position_mode = position_mode
        self.tie_heads = False
        self.qkv = nn.Linear(d_model, 3 * self.value_dim)
        self.out = nn.Linear(self.value_dim, d_model)
        self.rotary = (
            RotaryEmbedding(self.head_dim) if position_mode == "rope" else None
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        return_attention: bool = False,
        ablate_head: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, length, _ = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim)
        if self.tie_heads:
            qkv = qkv[:, :, :, :1, :].expand(
                batch, length, 3, self.n_heads, self.head_dim
            )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.position_mode == "rope":
            if positions is None:
                raise ValueError("positions are required for rotary attention")
            cos, sin = self.rotary(positions)
            q, k = apply_rotary(q, k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1)
        if ablate_head is not None:
            attention = attention.clone()
            attention[:, ablate_head] = 0
        values = torch.matmul(attention, v)
        values = values.transpose(1, 2).contiguous().view(
            batch, length, self.value_dim
        )
        output = self.out(values)
        return output, (attention if return_attention else None)


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_mlp: bool,
        position_mode: str,
        head_dim_override: int | None = None,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(
            d_model, n_heads, position_mode, head_dim_override
        )
        self.use_mlp = use_mlp
        if use_mlp:
            self.ln_2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )
        else:
            self.ln_2 = None
            self.mlp = None

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        return_attention: bool = False,
        ablate_head: int | None = None,
        ablate_mlp: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attention_output, attention = self.attention(
            self.ln_1(x),
            positions=positions,
            return_attention=return_attention,
            ablate_head=ablate_head,
        )
        x = x + attention_output
        if self.use_mlp and not ablate_mlp:
            x = x + self.mlp(self.ln_2(x))
        return x, attention


class TinyDecoderTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_nodes: int,
        max_length: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 2,
        use_mlp: bool = True,
        dropout: float = 0.0,
        position_mode: str = "learned",
        num_answer_heads: int = 1,
        tie_heads: bool = False,
        attention_out_scale: float = 1.0,
        head_dim_override: int | None = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        if position_mode not in {"learned", "rope"}:
            raise ValueError("position_mode must be one of: learned, rope")
        self.position_mode = position_mode
        self.max_length = max_length
        # Keep this module in both modes so that initialization order and
        # parameter count are identical across learned-position and RoPE runs.
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model,
                    n_heads,
                    use_mlp,
                    position_mode,
                    head_dim_override,
                )
                for _ in range(n_layers)
            ]
        )
        if tie_heads:
            for block in self.blocks:
                block.attention.tie_heads = True
        self.ln_final = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.node_head = nn.Linear(d_model, num_nodes, bias=False)
        self.num_answer_heads = num_answer_heads
        self.extra_answer_heads = nn.ModuleList(
            [
                nn.Linear(d_model, num_nodes, bias=False)
                for _ in range(num_answer_heads - 1)
            ]
        )
        self.apply(self._init_weights)
        for block in self.blocks:
            block.attention.out.weight.data.mul_(attention_out_scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        return_attention: bool = False,
        return_hidden: bool = False,
        ablate_layer: int | None = None,
        ablate_head: int | None = None,
        ablate_mlp: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]
    ):
        batch, length = tokens.shape
        if length > self.max_length:
            raise ValueError("sequence exceeds configured maximum length")
        positions = torch.arange(length, device=tokens.device).unsqueeze(0)
        x = self.token_embedding(tokens)
        if self.position_mode == "learned":
            x = x + self.position_embedding(positions)
        x = self.dropout(x)
        attentions: list[torch.Tensor] = []
        hidden_states: list[torch.Tensor] = []
        for layer, block in enumerate(self.blocks):
            selected_head = (
                ablate_head if ablate_layer is not None and layer == ablate_layer else None
            )
            x, attention = block(
                x,
                positions=positions,
                return_attention=return_attention,
                ablate_head=selected_head,
                ablate_mlp=ablate_mlp,
            )
            if return_hidden:
                hidden_states.append(x)
            if attention is not None:
                attentions.append(attention)
        normalized = self.ln_final(x)
        logits = [self.node_head(normalized)]
        logits.extend(head(normalized) for head in self.extra_answer_heads)
        logits = torch.stack(logits, dim=-2) if self.num_answer_heads > 1 else logits[0]
        if return_attention and return_hidden:
            return logits, attentions, hidden_states
        if return_attention:
            return logits, attentions
        if return_hidden:
            return logits, hidden_states
        return logits
