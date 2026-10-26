from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RecordConfig:
    num_keys: int = 32
    num_values: int = 32
    num_records: int = 8
    num_queries: int = 2

    def __post_init__(self) -> None:
        if self.num_records > self.num_keys:
            raise ValueError("each record needs a unique key")
        if not 2 <= self.num_queries <= self.num_records:
            raise ValueError("num_queries must be between 2 and num_records")


@dataclass
class RecordBatch:
    record_keys: torch.Tensor
    record_values: torch.Tensor
    queries: torch.Tensor
    answers: torch.Tensor


def sample_record_batch(
    config: RecordConfig,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> RecordBatch:
    record_keys = []
    record_values = []
    queries = []
    answers = []
    for _ in range(batch_size):
        keys = torch.randperm(config.num_keys, generator=generator)[
            : config.num_records
        ]
        values = torch.randperm(config.num_values, generator=generator)[
            : config.num_records
        ]
        query_positions = torch.randperm(
            config.num_records, generator=generator
        )[: config.num_queries]
        queries.append(keys[query_positions])
        answers.append(values[query_positions])
        record_keys.append(keys)
        record_values.append(values)
    return RecordBatch(
        record_keys=torch.stack(record_keys).to(device),
        record_values=torch.stack(record_values).to(device),
        queries=torch.stack(queries).to(device),
        answers=torch.stack(answers).to(device),
    )
