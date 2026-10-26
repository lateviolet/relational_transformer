from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import TaskConfig, Vocabulary, sample_episodes
from model import TinyDecoderTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=4)
    return parser.parse_args()


def token_name(token: int, vocab: Vocabulary) -> str:
    if token < vocab.num_nodes:
        return f"n{token}"
    if token == vocab.bos:
        return "BOS"
    if token == vocab.query:
        return "QUERY"
    if token == vocab.answer:
        return "ANS"
    if vocab.relation_start <= token < vocab.task_start:
        return f"REL{token - vocab.relation_start}"
    return f"TASK{token - vocab.task_start}"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
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

    generator = torch.Generator().manual_seed(saved["seed"] + 10_000)
    batch = sample_episodes(config, args.examples, generator)
    tokens = batch.tokens.to(device)
    with torch.no_grad():
        logits, attentions, hidden_states = model(
            tokens, return_attention=True, return_hidden=True
        )
    predictions = logits[:, -1, :].argmax(dim=-1).cpu()
    query_position = tokens.shape[1] - 2
    final_position = tokens.shape[1] - 1

    # Logit lens: inspect what the query token represents after the first
    # layer, and what the final answer slot represents after each layer.
    first_layer_query = model.node_head(
        model.ln_final(hidden_states[0][:, query_position, :])
    )
    query_logit_lens = first_layer_query.argmax(dim=-1).cpu()

    for example in range(args.examples):
        sequence = tokens[example].tolist()
        names = [token_name(token, vocab) for token in sequence]
        print(f"example={example} metadata={batch.metadata[example]}")
        print("tokens:", " ".join(names))
        print(
            "prediction:",
            int(predictions[example]),
            "target:",
            int(batch.answers[example]),
        )
        print(
            "layer-1 query logit lens:",
            int(query_logit_lens[example]),
            "expected intermediate:",
            int(batch.intermediates[example]),
        )
        for layer, attention in enumerate(attentions):
            final_query = attention[example, :, -1, :]
            for head in range(attention.shape[1]):
                values, indices = final_query[head].topk(args.top_k)
                attended = [
                    f"{names[int(position)]}:{float(score):.3f}"
                    for position, score in zip(indices.cpu(), values.cpu())
                ]
                print(f"  layer={layer} head={head}: {' | '.join(attended)}")


if __name__ == "__main__":
    main()
