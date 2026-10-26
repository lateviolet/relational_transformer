from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class CausalDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        prompt_tokens = self.tokenizer(
            row["prompt"], add_special_tokens=False
        )["input_ids"]
        answer_tokens = self.tokenizer(
            " " + row["answer"], add_special_tokens=False
        )["input_ids"]
        answer_tokens = answer_tokens + [self.tokenizer.eos_token_id]
        input_ids = (prompt_tokens + answer_tokens)[: self.max_length]
        prompt_length = min(len(prompt_tokens), self.max_length - len(answer_tokens))
        labels = [-100] * prompt_length + answer_tokens
        labels = labels[: len(input_ids)]
        return {"input_ids": input_ids, "labels": labels}


class CausalCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        labels = []
        attention_mask = []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(
                feature["input_ids"] + [self.tokenizer.pad_token_id] * padding
            )
            labels.append(feature["labels"] + [-100] * padding)
            attention_mask.append([1] * len(feature["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="Qwen/Qwen3-0.6B-Base"
    )
    parser.add_argument(
        "--data-dir", default="data/nlp_data"
    )
    parser.add_argument(
        "--output-dir", default="outputs/lora_two_hop"
    )
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--eval-examples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def evaluate_generation(
    model,
    tokenizer,
    rows,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    correct = 0
    records = []
    for row in rows:
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(device)
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
        expected = row["answer"]
        is_correct = expected.lower() in completion.lower()
        correct += int(is_correct)
        if len(records) < 16:
            records.append(
                {
                    "expected": expected,
                    "completion": completion,
                    "correct": is_correct,
                }
            )
    return {
        "accuracy": correct / max(1, len(rows)),
        "examples": len(rows),
        "samples": records,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora)

    data_files = {
        "train": str(Path(args.data_dir) / "train.jsonl"),
        "valid_id": str(Path(args.data_dir) / "valid_id.jsonl"),
    }
    dataset = load_dataset("json", data_files=data_files)
    train_dataset = CausalDataset(dataset["train"], tokenizer, args.max_length)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        logging_steps=25,
        save_steps=args.max_steps,
        bf16=True,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=CausalCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    model.eval()
    device = next(model.parameters()).device
    valid_rows = [
        json.loads(line)
        for line in (
            Path(args.data_dir) / "valid_id.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ][: args.eval_examples]
    ood_candidates = (
        "test_more_distractors.jsonl",
        "test_unseen_templates.jsonl",
        "test_r32.jsonl",
    )
    test_path = next(
        (
            path
            for name in ood_candidates
            if (path := Path(args.data_dir) / name).exists()
        ),
        None,
    )
    ood_rows = (
        [
            json.loads(line)
            for line in test_path.read_text(encoding="utf-8").splitlines()
        ][: args.eval_examples]
        if test_path is not None and args.eval_examples > 0
        else []
    )
    results = {
        "valid_id": (
            evaluate_generation(model, tokenizer, valid_rows, 24, device)
            if valid_rows
            else {"accuracy": None, "examples": 0, "samples": []}
        ),
        "out_of_distribution": (
            evaluate_generation(model, tokenizer, ood_rows, 24, device)
            if ood_rows
            else {"accuracy": None, "examples": 0, "samples": []}
        ),
    }
    (output_dir / "eval_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
