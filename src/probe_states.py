from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import sample_episodes
from diagnose import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-examples", type=int, default=4096)
    parser.add_argument("--test-examples", type=int, default=1024)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--result-json", default="")
    return parser.parse_args()


@torch.no_grad()
def collect(
    model,
    config,
    examples: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]:
    batch = sample_episodes(config, examples, generator)
    tokens = batch.tokens.to(device)
    query_position = tokens.shape[1] - 2
    final_position = tokens.shape[1] - 1
    _, hidden_states = model(tokens, return_hidden=True)
    features = []
    for hidden in hidden_states:
        features.append(
            (
                hidden[:, query_position].float().cpu(),
                hidden[:, final_position].float().cpu(),
            )
        )
    return features, batch.intermediates, batch.answers


def fit_ridge(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    num_classes: int,
    ridge: float,
) -> float:
    targets = torch.nn.functional.one_hot(train_y, num_classes=num_classes).float()
    gram = train_x.T @ train_x
    gram += ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
    weights = torch.linalg.solve(gram, train_x.T @ targets)
    predictions = (test_x @ weights).argmax(-1)
    return float((predictions == test_y).float().mean())


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, config, _, _ = load_model(args.checkpoint, device)
    train_features, train_intermediate, train_answer = collect(
        model,
        config,
        args.train_examples,
        torch.Generator().manual_seed(20_000),
        device,
    )
    test_features, test_intermediate, test_answer = collect(
        model,
        config,
        args.test_examples,
        torch.Generator().manual_seed(30_000),
        device,
    )

    results = []
    for layer in range(len(train_features)):
        for position_index, position_name in ((0, "query"), (1, "final")):
            train_x = train_features[layer][position_index]
            test_x = test_features[layer][position_index]
            results.append(
                {
                    "layer": layer,
                    "position": position_name,
                    "intermediate_probe": fit_ridge(
                        train_x,
                        train_intermediate,
                        test_x,
                        test_intermediate,
                        config.num_nodes,
                        args.ridge,
                    ),
                    "answer_probe": fit_ridge(
                        train_x,
                        train_answer,
                        test_x,
                        test_answer,
                        config.num_nodes,
                        args.ridge,
                    ),
                }
            )

    print(json.dumps(results, indent=2))
    if args.result_json:
        path = Path(args.result_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
