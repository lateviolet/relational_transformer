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
    parser.add_argument("--adapter", default="")
    parser.add_argument(
        "--data",
        default="data/nlp_data/valid_id.jsonl",
    )
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument(
        "--prompt-field",
        default="prompt",
        choices=["prompt", "alternative_prompt"],
    )
    parser.add_argument("--result-json", default="")
    parser.add_argument(
        "--match-mode",
        choices=("exact", "contains"),
        default="exact",
        help="Compare normalized output exactly or use substring matching.",
    )
    parser.add_argument(
        "--ablate-heads",
        nargs="*",
        default=[],
        help="Head specifications in LAYER:HEAD form.",
    )
    return parser.parse_args()


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split())


def resolve_layers(model) -> torch.nn.ModuleList:
    candidates = [
        module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.ModuleList)
        and name.endswith("layers")
        and len(module) > 0
        and hasattr(module[0], "self_attn")
    ]
    if len(candidates) != 1:
        raise RuntimeError("could not uniquely resolve decoder layers")
    return candidates[0]


def apply_head_ablations(model, specifications: list[str]) -> list[dict]:
    def make_hook(start: int, end: int):
        def ablate(_module, inputs):
            hidden = inputs[0].clone()
            hidden[..., start:end] = 0
            return (hidden,) + inputs[1:]

        return ablate

    layers = resolve_layers(model)
    head_dim = int(model.config.head_dim)
    applied = []
    for specification in specifications:
        layer_text, head_text = specification.split(":", 1)
        layer = int(layer_text)
        head = int(head_text)
        projection = layers[layer].self_attn.o_proj
        start = head * head_dim
        end = start + head_dim

        projection.register_forward_pre_hook(make_hook(start, end))
        applied.append({"layer": layer, "head": head})
    return applied


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    ablated_heads = apply_head_ablations(model, args.ablate_heads)

    rows = [
        json.loads(line)
        for line in Path(args.data)
        .read_text(encoding="utf-8")
        .splitlines()
    ][: args.examples]
    correct = 0
    samples = []
    with torch.no_grad():
        for row in rows:
            prompt = row[args.prompt_field]
            if args.chat_template:
                messages = [{"role": "user", "content": prompt}]
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
            completion = tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()
            expected_key = (
                "answer"
                if args.prompt_field == "prompt"
                else "alternative_answer"
            )
            expected = normalize_answer(row[expected_key])
            observed = normalize_answer(completion)
            is_correct = (
                expected == observed
                if args.match_mode == "exact"
                else expected in observed
            )
            correct += int(is_correct)
            if len(samples) < 16:
                samples.append(
                    {
                        "expected": row[expected_key],
                        "completion": completion,
                        "correct": is_correct,
                    }
                )
    result = {
        "model": args.model,
        "adapter": args.adapter,
        "data": args.data,
        "prompt_field": args.prompt_field,
        "match_mode": args.match_mode,
        "ablated_heads": ablated_heads,
        "examples": len(rows),
        "accuracy": correct / max(1, len(rows)),
        "samples": samples,
    }
    print(json.dumps(result, indent=2))
    if args.result_json:
        path = Path(args.result_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
