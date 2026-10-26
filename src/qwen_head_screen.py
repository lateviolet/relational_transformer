from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="Qwen/Qwen3-0.6B-Base"
    )
    parser.add_argument("--adapter", required=True)
    parser.add_argument(
        "--data",
        default="data/nlp_pair_data/valid_id.jsonl",
    )
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int, default=28)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--result-json", required=True)
    return parser.parse_args()


def resolve_layers(model) -> torch.nn.ModuleList:
    candidates = []
    for name, module in model.named_modules():
        if (
            isinstance(module, torch.nn.ModuleList)
            and name.endswith("layers")
            and len(module) > 0
            and hasattr(module[0], "self_attn")
        ):
            candidates.append((name, module))
    if len(candidates) != 1:
        raise RuntimeError(f"could not uniquely resolve decoder layers: {candidates}")
    return candidates[0][1]


def make_examples(
    path: Path,
    tokenizer,
    count: int,
    max_length: int,
) -> tuple[dict[str, torch.Tensor], int]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ][:count]
    examples = []
    max_seen = 0
    for row in rows:
        prompt_ids = tokenizer(
            row["prompt"], add_special_tokens=False
        ).input_ids
        answer_ids = tokenizer(
            " " + row["answer"], add_special_tokens=False
        ).input_ids
        answer_ids = answer_ids + [tokenizer.eos_token_id]
        if len(prompt_ids) + len(answer_ids) > max_length:
            overflow = len(prompt_ids) + len(answer_ids) - max_length
            prompt_ids = prompt_ids[overflow:]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        examples.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": [1] * len(input_ids),
            }
        )
        max_seen = max(max_seen, len(input_ids))
    pad = tokenizer.pad_token_id
    output = {"input_ids": [], "labels": [], "attention_mask": []}
    for example in examples:
        padding = max_seen - len(example["input_ids"])
        output["input_ids"].append(example["input_ids"] + [pad] * padding)
        output["labels"].append(example["labels"] + [-100] * padding)
        output["attention_mask"].append(
            example["attention_mask"] + [0] * padding
        )
    tensors = {
        name: torch.tensor(values, dtype=torch.long)
        for name, values in output.items()
    }
    return tensors, max_seen


@torch.no_grad()
def evaluate(
    model,
    tensors: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    count = tensors["input_ids"].shape[0]
    for start in range(0, count, batch_size):
        batch = {
            name: value[start : start + batch_size].to(device)
            for name, value in tensors.items()
        }
        logits = model(**batch).logits
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = batch["labels"][:, 1:].contiguous()
        mask = shift_labels.ne(-100)
        loss = torch.nn.functional.cross_entropy(
            shift_logits[mask],
            shift_labels[mask],
            reduction="sum",
        )
        total_loss += float(loss)
        total_tokens += int(mask.sum())
        predictions = shift_logits.argmax(dim=-1)
        correct += int((predictions[mask] == shift_labels[mask]).sum())
    return total_loss / total_tokens, correct / total_tokens


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter).to(device)
    model.eval()
    layers = resolve_layers(model)
    tensors, sequence_length = make_examples(
        Path(args.data),
        tokenizer,
        args.examples,
        args.max_length,
    )
    baseline_loss, baseline_accuracy = evaluate(
        model, tensors, args.batch_size, device
    )
    head_dim = int(model.config.head_dim)
    num_heads = int(model.config.num_attention_heads)
    results = []
    for layer in range(args.layer_start, min(args.layer_end, len(layers))):
        projection = layers[layer].self_attn.o_proj
        for head in range(num_heads):
            start = head * head_dim
            end = start + head_dim

            def ablate(_module, inputs):
                hidden = inputs[0].clone()
                hidden[..., start:end] = 0
                return (hidden,) + inputs[1:]

            handle = projection.register_forward_pre_hook(ablate)
            loss, accuracy = evaluate(
                model, tensors, args.batch_size, device
            )
            handle.remove()
            results.append(
                {
                    "layer": layer,
                    "head": head,
                    "loss": loss,
                    "delta_loss": loss - baseline_loss,
                    "answer_token_accuracy": accuracy,
                    "delta_accuracy": accuracy - baseline_accuracy,
                }
            )
            print(json.dumps(results[-1]), flush=True)

    payload = {
        "adapter": args.adapter,
        "data": args.data,
        "examples": len(tensors["input_ids"]),
        "sequence_length": sequence_length,
        "layer_start": args.layer_start,
        "layer_end": args.layer_end,
        "baseline_loss": baseline_loss,
        "baseline_answer_token_accuracy": baseline_accuracy,
        "heads": sorted(
            results,
            key=lambda item: item["delta_loss"],
            reverse=True,
        ),
    }
    path = Path(args.result_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
