from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import TaskConfig, Vocabulary, sample_episodes
from model import TinyDecoderTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--result-json", default="")
    return parser.parse_args()


def load_model(
    checkpoint_path: str, device: torch.device
) -> tuple[TinyDecoderTransformer, TaskConfig, torch.Generator, int]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    saved = checkpoint["args"]
    config = TaskConfig(
        task=saved["task"],
        num_nodes=saved["num_nodes"],
        edge_count=saved["edge_count"],
        num_relations=saved["num_relations"],
        hops=saved.get("hops", 2),
    )
    vocab = Vocabulary(config.num_nodes, config.num_relations)
    relation_prefix = 1 if config.task == "cued_two_hop" else 0
    eval_edge_counts = sorted(
        {
            config.edge_count,
            *saved.get("eval_edge_counts", []),
        }
    )
    max_length = max(
        2 + relation_prefix + 3 * edge_count + 3
        for edge_count in eval_edge_counts
    )
    max_length = max(max_length, saved.get("max_length", 0))
    model = TinyDecoderTransformer(
        vocab_size=vocab.size,
        num_nodes=vocab.num_nodes,
        max_length=max_length,
        d_model=saved["d_model"],
        n_layers=saved["layers"],
        n_heads=saved["heads"],
        use_mlp=saved["mlp"],
        position_mode=saved.get("position_mode", "learned"),
        num_answer_heads=2
        if saved.get("task") in {"dual_retrieval", "parallel_copy"}
        else 1,
        head_dim_override=saved.get("head_dim") or None,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, torch.Generator().manual_seed(saved["seed"]), relation_prefix


def edge_positions(
    sequence: list[int],
    metadata: dict,
    config: TaskConfig,
    vocab: Vocabulary,
) -> dict[str, int]:
    start = metadata["start"]
    intermediate = metadata["intermediate"]
    active = metadata["active_relation"]
    found: dict[str, int] = {}
    edge_cursor = 1 + (1 if config.task == "cued_two_hop" else 0)
    for edge_index, (key, value, relation) in enumerate(metadata["edges"]):
        relation_position = edge_cursor + 3 * edge_index
        key_position = relation_position + 1
        value_position = relation_position + 2
        assert sequence[relation_position] == vocab.relation(relation)
        assert sequence[key_position] == key
        assert sequence[value_position] == value
        if key == start and value == intermediate and relation == active:
            found["first_key"] = key_position
            found["first_value"] = value_position
        if key == intermediate and relation == active:
            found["second_key"] = key_position
            found["second_value"] = value_position
    return found


@torch.no_grad()
def diagnose(
    model: TinyDecoderTransformer,
    config: TaskConfig,
    generator: torch.Generator,
    device: torch.device,
    num_examples: int,
) -> dict:
    vocab = Vocabulary(config.num_nodes, config.num_relations)
    batch = sample_episodes(config, num_examples, generator)
    tokens = batch.tokens.to(device)
    logits, attentions, hidden_states = model(
        tokens, return_attention=True, return_hidden=True
    )
    query_position = tokens.shape[1] - 2
    final_position = tokens.shape[1] - 1

    final_predictions = logits[:, final_position].argmax(-1).cpu()
    metrics: dict = {
        "examples": num_examples,
        "final_accuracy": float(
            (final_predictions == batch.answers).float().mean()
        ),
        "final_prediction_matches_intermediate": float(
            (final_predictions == batch.intermediates).float().mean()
        ),
        "layer_logit_lens": [],
        "attention_to_hop_tokens": [],
        "head_ablation": [],
    }

    for layer, hidden in enumerate(hidden_states):
        node_logits = model.node_head(model.ln_final(hidden))
        final_match = (
            node_logits[:, final_position].argmax(-1).cpu() == batch.answers
        ).float().mean()
        query_match = (
            node_logits[:, query_position].argmax(-1).cpu()
            == batch.intermediates
        ).float().mean()
        metrics["layer_logit_lens"].append(
            {
                "layer": layer,
                "query_predicts_intermediate": float(query_match),
                "final_predicts_answer": float(final_match),
            }
        )

    first_value_positions = []
    second_value_positions = []
    for row in range(num_examples):
        sequence = tokens[row].tolist()
        positions = edge_positions(
            sequence, batch.metadata[row], config, vocab
        )
        first_value_positions.append(positions["first_value"])
        second_value_positions.append(positions["second_value"])

    first_positions = torch.tensor(first_value_positions, device=device)
    second_positions = torch.tensor(second_value_positions, device=device)
    example_indices = torch.arange(num_examples, device=device)

    for layer, attention in enumerate(attentions):
        for head in range(attention.shape[1]):
            query_to_first = attention[
                example_indices, head, query_position, first_positions
            ]
            final_to_second = attention[
                example_indices, head, final_position, second_positions
            ]
            query_top = attention[example_indices, head, query_position].argmax(-1)
            final_top = attention[example_indices, head, final_position].argmax(-1)
            metrics["attention_to_hop_tokens"].append(
                {
                    "layer": layer,
                    "head": head,
                    "query_mass_on_first_value": float(query_to_first.mean()),
                    "query_top1_matches_first_value": float(
                        (query_top == first_positions).float().mean()
                    ),
                    "final_mass_on_second_value": float(final_to_second.mean()),
                    "final_top1_matches_second_value": float(
                        (final_top == second_positions).float().mean()
                    ),
                }
            )

    for layer in range(len(model.blocks)):
        for head in range(model.blocks[layer].attention.n_heads):
            ablated_logits, ablated_hidden = model(
                tokens,
                return_hidden=True,
                ablate_layer=layer,
                ablate_head=head,
            )
            ablated_query = model.node_head(
                model.ln_final(ablated_hidden[layer][:, query_position, :])
            )
            metrics["head_ablation"].append(
                {
                    "layer": layer,
                    "head": head,
                    "final_accuracy": float(
                        (
                            ablated_logits[:, final_position].argmax(-1).cpu()
                            == batch.answers
                        )
                        .float()
                        .mean()
                    ),
                    "query_predicts_intermediate": float(
                        (
                            ablated_query.argmax(-1).cpu()
                            == batch.intermediates
                        )
                        .float()
                        .mean()
                    ),
                }
            )
    return metrics


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, config, _, _ = load_model(args.checkpoint, device)
    metrics = diagnose(
        model,
        config,
        torch.Generator().manual_seed(args.seed),
        device,
        args.examples,
    )
    print(json.dumps(metrics, indent=2))
    if args.result_json:
        path = Path(args.result_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
