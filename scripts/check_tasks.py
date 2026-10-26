from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import TaskConfig, Vocabulary, sample_episodes, validate_episode
from model import RotaryEmbedding, TinyDecoderTransformer, apply_rotary
from record_model import RecordTransformer
from record_task import RecordConfig, sample_record_batch


def main() -> None:
    generator = torch.Generator().manual_seed(7)
    rotary = RotaryEmbedding(16)
    q = torch.randn(1, 1, 8, 16, generator=generator)
    k = torch.randn(1, 1, 8, 16, generator=generator)
    cos, sin = rotary(torch.arange(8).unsqueeze(0))
    qr, kr = apply_rotary(q, k, cos, sin)
    shifted_cos, shifted_sin = rotary((torch.arange(8) + 37).unsqueeze(0))
    shifted_q, shifted_k = apply_rotary(q, k, shifted_cos, shifted_sin)
    rotated_scores = torch.matmul(qr, kr.transpose(-1, -2))
    shifted_scores = torch.matmul(shifted_q, shifted_k.transpose(-1, -2))
    torch.testing.assert_close(shifted_scores, rotated_scores)
    print("rope relative-position invariance ok")

    for task, hops in (
        ("two_hop", 2),
        ("cued_two_hop", 2),
        ("cued_nhop", 4),
        ("dual_retrieval", 2),
        ("parallel_copy", 2),
        ("parallel_intersection", 2),
    ):
        config = TaskConfig(task=task, num_nodes=32, edge_count=8, hops=hops)
        batch = sample_episodes(config, 64, generator)
        for index in range(64):
            validate_episode(config, batch, index)
        vocab = Vocabulary(config.num_nodes, config.num_relations)
        for position_mode in ("learned", "rope"):
            model = TinyDecoderTransformer(
                vocab_size=vocab.size,
                num_nodes=vocab.num_nodes,
                max_length=64,
                d_model=64,
                n_layers=2,
                n_heads=2,
                use_mlp=True,
                position_mode=position_mode,
                num_answer_heads=2
                if task in {"dual_retrieval", "parallel_copy"}
                else 1,
            )
            logits, attention = model(batch.tokens, return_attention=True)
            expected_logits = (
                (64, batch.tokens.shape[1], 2, config.num_nodes)
                if task in {"dual_retrieval", "parallel_copy"}
                else (64, batch.tokens.shape[1], config.num_nodes)
            )
            assert logits.shape == expected_logits
            assert len(attention) == 2
            assert attention[0].shape == (
                64,
                2,
                batch.tokens.shape[1],
                batch.tokens.shape[1],
            )
            print(
                task,
                position_mode,
                "ok",
                "tokens",
                tuple(batch.tokens.shape),
                "attention",
                tuple(attention[0].shape),
            )

    for num_queries, d_model in ((2, 96), (3, 132), (4, 180)):
        config = RecordConfig(
            num_keys=128,
            num_values=128,
            num_records=32,
            num_queries=num_queries,
        )
        batch = sample_record_batch(config, 16, generator, torch.device("cpu"))
        assert batch.queries.shape == (16, num_queries)
        assert batch.answers.shape == (16, num_queries)
        pair_equal = batch.queries.unsqueeze(2) == batch.queries.unsqueeze(1)
        off_diagonal = ~torch.eye(
            num_queries, dtype=torch.bool
        ).unsqueeze(0)
        assert not torch.any(pair_equal & off_diagonal)
        model = RecordTransformer(
            num_keys=config.num_keys,
            num_values=config.num_values,
            num_queries=num_queries,
            d_model=d_model,
            n_layers=1,
            n_heads=num_queries,
            head_dim_override=32,
            max_length=64,
        )
        logits, attention = model(
            batch.record_keys,
            batch.record_values,
            batch.queries,
        )
        assert logits.shape == (16, num_queries, config.num_values)
        assert attention[0].shape == (
            16,
            num_queries,
            config.num_records + 1,
            config.num_records + 1,
        )
        print(
            "record",
            num_queries,
            "queries ok",
            tuple(logits.shape),
            tuple(attention[0].shape),
        )


if __name__ == "__main__":
    main()
