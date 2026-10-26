from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


TaskName = Literal[
    "two_hop",
    "cued_two_hop",
    "cued_nhop",
    "dual_retrieval",
    "parallel_copy",
    "parallel_intersection",
]


@dataclass(frozen=True)
class TaskConfig:
    task: TaskName = "two_hop"
    num_nodes: int = 32
    edge_count: int = 8
    num_relations: int = 2
    hops: int = 2

    def __post_init__(self) -> None:
        if self.edge_count < 1:
            raise ValueError("edge_count must be at least 1")
        if self.hops < 1:
            raise ValueError("hops must be at least 1")
        if self.task == "cued_nhop" and self.edge_count < self.hops:
            raise ValueError("edge_count must be at least the number of hops")
        if self.task == "dual_retrieval" and self.edge_count < 2:
            raise ValueError("dual_retrieval requires at least two edges")
        required_nodes = (
            3 + 2 * self.edge_count
            if self.task == "parallel_intersection"
            else 2 * self.edge_count
            if self.task in {"dual_retrieval", "parallel_copy"}
            else self.edge_count * 2 - self.hops + 1
        )
        if self.num_nodes < required_nodes:
            raise ValueError(
                "num_nodes is too small for the requested number of disjoint edges"
            )
        if (
            self.task
            in {
                "cued_two_hop",
                "cued_nhop",
                "dual_retrieval",
                "parallel_copy",
                "parallel_intersection",
            }
            and self.num_relations < 2
        ):
            raise ValueError("the selected task requires at least two relations")


class Vocabulary:
    """Node ids come first so classification can use a node-only output head."""

    def __init__(self, num_nodes: int, num_relations: int) -> None:
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.bos = num_nodes
        self.query = num_nodes + 1
        self.answer = num_nodes + 2
        self.relation_start = num_nodes + 3
        self.task_start = self.relation_start + num_relations
        self.size = self.task_start + num_relations

    def relation(self, relation: int) -> int:
        return self.relation_start + relation

    def task(self, relation: int) -> int:
        return self.task_start + relation


@dataclass
class Batch:
    tokens: torch.Tensor
    answers: torch.Tensor
    secondary_answers: torch.Tensor
    starts: torch.Tensor
    intermediates: torch.Tensor
    active_relations: torch.Tensor
    metadata: list[dict]


def sample_episodes(
    config: TaskConfig,
    batch_size: int,
    generator: torch.Generator,
) -> Batch:
    """Sample episodes without any edge that could create an alternate two-hop path."""

    vocab = Vocabulary(config.num_nodes, config.num_relations)
    sequences: list[list[int]] = []
    answers: list[int] = []
    secondary_answers: list[int] = []
    starts: list[int] = []
    intermediates: list[int] = []
    active_relations: list[int] = []
    metadata: list[dict] = []

    for _ in range(batch_size):
        if config.task in {"dual_retrieval", "parallel_copy"}:
            nodes = torch.randperm(config.num_nodes, generator=generator)
            start_1 = int(nodes[0])
            answer_1 = int(nodes[1])
            start_2 = int(nodes[2])
            answer_2 = int(nodes[3])
            all_edges = [
                (
                    start_1,
                    answer_1,
                    0,
                ),
                (
                    start_2,
                    answer_2,
                    0 if config.task == "parallel_copy" else 1,
                ),
            ]
            for index in range(config.edge_count - 2):
                key = int(nodes[4 + 2 * index])
                value = int(nodes[5 + 2 * index])
                relation = (
                    0
                    if config.task == "parallel_copy"
                    else index % config.num_relations
                )
                all_edges.append((key, value, relation))
            order = torch.randperm(len(all_edges), generator=generator).tolist()
            all_edges = [all_edges[index] for index in order]
            sequence = [vocab.bos]
            if config.task == "dual_retrieval":
                sequence.extend([vocab.task(0), vocab.task(1)])
            for key, value, relation in all_edges:
                sequence.extend([vocab.relation(relation), key, value])
            sequence.extend([vocab.query, start_1, start_2, vocab.answer])
            sequences.append(sequence)
            answers.append(answer_1)
            secondary_answers.append(answer_2)
            starts.append(start_1)
            intermediates.append(start_2)
            active_relations.append(-1)
            metadata.append(
                {
                    "edges": all_edges,
                    "start_1": start_1,
                    "answer_1": answer_1,
                    "start_2": start_2,
                    "answer_2": answer_2,
                }
            )
            continue

        if config.task == "parallel_intersection":
            nodes = torch.randperm(config.num_nodes, generator=generator)
            start_a = int(nodes[0])
            start_b = int(nodes[1])
            answer = int(nodes[2])
            cursor = 3
            all_edges = [
                (start_a, answer, 0),
                (start_b, answer, 1),
            ]
            for index in range(config.edge_count):
                key = int(nodes[cursor])
                value = int(nodes[cursor + 1])
                cursor += 2
                relation = index % config.num_relations
                all_edges.append((key, value, relation))
            order = torch.randperm(len(all_edges), generator=generator).tolist()
            all_edges = [all_edges[index] for index in order]
            sequence = [vocab.bos]
            for key, value, relation in all_edges:
                sequence.extend([vocab.relation(relation), key, value])
            sequence.extend([vocab.query, start_a, start_b, vocab.answer])

            sequences.append(sequence)
            answers.append(answer)
            secondary_answers.append(answer)
            starts.append(start_a)
            intermediates.append(start_b)
            active_relations.append(-1)
            metadata.append(
                {
                    "edges": all_edges,
                    "start_a": start_a,
                    "start_b": start_b,
                    "answer": answer,
                    "active_relation": -1,
                }
            )
            continue

        if config.task in {"cued_two_hop", "cued_nhop"}:
            active = int(torch.randint(config.num_relations, (1,), generator=generator))
        else:
            active = 0

        hops = config.hops if config.task == "cued_nhop" else 2
        # Reserve a simple path, then use disjoint pairs for distractors.
        # This prevents alternate paths and test leakage.
        required_nodes = 2 * config.edge_count - hops + 1
        nodes = torch.randperm(config.num_nodes, generator=generator)[:required_nodes]
        path = [int(node) for node in nodes[: hops + 1]]
        start = path[0]
        intermediate = path[1]
        answer = path[-1]
        all_edges = [
            (path[index], path[index + 1], active)
            for index in range(hops)
        ]
        distractor_relation = (active + 1) % config.num_relations
        for index in range(config.edge_count - hops):
            key = int(nodes[hops + 1 + 2 * index])
            value = int(nodes[hops + 2 + 2 * index])
            relation = active if config.task == "two_hop" else distractor_relation
            all_edges.append((key, value, relation))

        if config.task in {"cued_two_hop", "cued_nhop"}:
            assert all(edge[2] == active for edge in all_edges[:hops])

        order = torch.randperm(len(all_edges), generator=generator).tolist()
        all_edges = [all_edges[index] for index in order]

        sequence = [vocab.bos]
        if config.task in {"cued_two_hop", "cued_nhop"}:
            sequence.append(vocab.task(active))
        for key, value, relation in all_edges:
            sequence.extend([vocab.relation(relation), key, value])
        sequence.extend([vocab.query, start, vocab.answer])

        sequences.append(sequence)
        answers.append(answer)
        secondary_answers.append(answer)
        starts.append(start)
        intermediates.append(intermediate)
        active_relations.append(active)
        metadata.append(
            {
                "edges": all_edges,
                "start": start,
                "path": path,
                "intermediate": intermediate,
                "answer": answer,
                "active_relation": active,
            }
        )

    max_length = max(len(sequence) for sequence in sequences)
    tokens = torch.full(
        (batch_size, max_length), vocab.bos, dtype=torch.long
    )
    for row, sequence in enumerate(sequences):
        tokens[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)

    return Batch(
        tokens=tokens,
        answers=torch.tensor(answers, dtype=torch.long),
        secondary_answers=torch.tensor(secondary_answers, dtype=torch.long),
        starts=torch.tensor(starts, dtype=torch.long),
        intermediates=torch.tensor(intermediates, dtype=torch.long),
        active_relations=torch.tensor(active_relations, dtype=torch.long),
        metadata=metadata,
    )


def batch_to(
    tokens: torch.Tensor,
    answers: torch.Tensor,
    secondary_answers: torch.Tensor,
    device: torch.device,
):
    return (
        tokens.to(device, non_blocking=True),
        answers.to(device, non_blocking=True),
        secondary_answers.to(device, non_blocking=True),
    )


def validate_episode(config: TaskConfig, batch: Batch, index: int = 0) -> None:
    """Fail fast if the packed edges cannot produce the recorded target."""

    item = batch.metadata[index]
    edges = item["edges"]
    if config.task in {"dual_retrieval", "parallel_copy"}:
        relation_1 = 0
        relation_2 = 1 if config.task == "dual_retrieval" else 0
        first = [
            value
            for key, value, relation in edges
            if key == item["start_1"] and relation == relation_1
        ]
        second = [
            value
            for key, value, relation in edges
            if key == item["start_2"] and relation == relation_2
        ]
        assert first and first[0] == item["answer_1"], (first, item)
        assert second and second[0] == item["answer_2"], (second, item)
        return
    if config.task == "parallel_intersection":
        first = [
            value
            for key, value, relation in edges
            if key == item["start_a"] and relation == 0
        ]
        second = [
            value
            for key, value, relation in edges
            if key == item["start_b"] and relation == 1
        ]
        assert first and second and first[0] == second[0] == item["answer"]
        return
    active = item["active_relation"]
    current = item["start"]
    for expected in item["path"][1:]:
        matches = [
            value
            for key, value, relation in edges
            if key == current and relation == active
        ]
        assert matches and matches[0] == expected, (matches, item)
        current = expected
    assert current == item["answer"], item
