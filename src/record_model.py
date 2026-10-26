from __future__ import annotations

import torch
from torch import nn

from model import Block


class RecordTransformer(nn.Module):
    """One-token records with multiple query slots at the final token."""

    def __init__(
        self,
        num_keys: int,
        num_values: int,
        num_queries: int = 2,
        d_model: int = 96,
        n_layers: int = 1,
        n_heads: int = 2,
        use_mlp: bool = True,
        head_dim_override: int | None = None,
        tie_heads: bool = False,
        max_length: int = 128,
        position_mode: str = "learned",
    ) -> None:
        super().__init__()
        if d_model % (num_queries + 1):
            raise ValueError(
                "record model requires d_model divisible by num_queries + 1"
            )
        if position_mode not in {"learned", "none"}:
            raise ValueError("position_mode must be either learned or none")
        self.num_queries = num_queries
        self.key_dim = d_model // (num_queries + 1)
        self.value_dim = d_model - num_queries * self.key_dim
        self.position_mode = position_mode
        self.max_length = max_length
        self.key_embedding = nn.Embedding(num_keys, self.key_dim)
        self.value_embedding = nn.Embedding(num_values, self.value_dim)
        self.record_type = nn.Parameter(torch.zeros(d_model))
        self.query_type = nn.Parameter(torch.zeros(d_model))
        self.query_slot = nn.Parameter(
            torch.zeros(num_queries, d_model)
        )
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model,
                    n_heads,
                    use_mlp,
                    "learned",
                    head_dim_override,
                )
                for _ in range(n_layers)
            ]
        )
        if tie_heads:
            for block in self.blocks:
                block.attention.tie_heads = True
        self.ln_final = nn.LayerNorm(d_model)
        self.answer_heads = nn.ModuleList(
            [nn.Linear(d_model, num_values) for _ in range(num_queries)]
        )
        self.apply(self._init_weights)

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
        record_keys: torch.Tensor,
        record_values: torch.Tensor,
        queries: torch.Tensor,
        ablate_head: int | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        key = self.key_embedding(record_keys)
        value = self.value_embedding(record_values)
        record_tokens = torch.cat((key,) * self.num_queries + (value,), dim=-1)
        record_tokens = record_tokens + self.record_type

        query_parts = []
        for index in range(self.num_queries):
            start = index * self.key_dim
            query_parts.append(
                self.key_embedding(queries[:, index])
                + self.query_slot[index, start : start + self.key_dim]
            )
        query_parts.append(
            torch.zeros(
                queries.shape[0],
                self.value_dim,
                device=queries.device,
            )
        )
        query_token = torch.cat(query_parts, dim=-1)
        query_token = query_token + self.query_type
        x = torch.cat((record_tokens, query_token.unsqueeze(1)), dim=1)
        if x.shape[1] > self.max_length:
            raise ValueError("sequence exceeds configured maximum length")
        positions = torch.arange(x.shape[1], device=x.device)
        if self.position_mode == "learned":
            x = x + self.position_embedding(positions).unsqueeze(0)

        attentions = []
        for block in self.blocks:
            x, attention = block(
                x,
                positions=positions.unsqueeze(0),
                return_attention=True,
                ablate_head=ablate_head,
            )
            attentions.append(attention)
        hidden = self.ln_final(x[:, -1])
        logits = torch.stack([head(hidden) for head in self.answer_heads], dim=1)
        return logits, attentions
