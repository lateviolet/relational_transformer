from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from record_model import RecordTransformer
from record_task import RecordConfig, sample_record_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--result-json", default="")
    return parser.parse_args()


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    saved = checkpoint["args"]
    config = RecordConfig(
        num_keys=saved["num_keys"],
        num_values=saved["num_values"],
        num_records=saved["num_records"],
        num_queries=saved.get("num_queries", 2),
    )
    model = RecordTransformer(
        num_keys=config.num_keys,
        num_values=config.num_values,
        num_queries=config.num_queries,
        d_model=saved["d_model"],
        n_layers=saved["layers"],
        n_heads=saved["heads"],
        use_mlp=saved["mlp"],
        head_dim_override=saved["head_dim"] or None,
        tie_heads=saved["tie_heads"],
        max_length=saved.get("max_length", 128),
        position_mode=saved.get("position_mode", "learned"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, saved


def build_record_inputs(
    model: RecordTransformer,
    record_keys: torch.Tensor,
    record_values: torch.Tensor,
    queries: torch.Tensor,
    query_key_patch: torch.Tensor | None = None,
) -> torch.Tensor:
    key = model.key_embedding(record_keys)
    value = model.value_embedding(record_values)
    records = torch.cat((key, key, value), dim=-1) + model.record_type

    query_key_1 = model.key_embedding(queries[:, 0])
    query_key_2 = model.key_embedding(queries[:, 1])
    if query_key_patch is not None:
        query_key_1 = query_key_patch[:, 0]
        query_key_2 = query_key_patch[:, 1]
    query = torch.cat(
        (
            query_key_1 + model.query_slot[0, : model.key_dim],
            query_key_2 + model.query_slot[1, model.key_dim : 2 * model.key_dim],
            torch.zeros(
                queries.shape[0],
                model.value_dim,
                device=queries.device,
            ),
        ),
        dim=-1,
    )
    query = query + model.query_type
    x = torch.cat((records, query.unsqueeze(1)), dim=1)
    positions = torch.arange(x.shape[1], device=x.device)
    if model.position_mode == "learned":
        return x + model.position_embedding(positions).unsqueeze(0)
    return x


def manual_forward(
    model: RecordTransformer,
    record_keys: torch.Tensor,
    record_values: torch.Tensor,
    queries: torch.Tensor,
    query_key_patch: torch.Tensor | None = None,
    patch_head: tuple[int, int, torch.Tensor] | None = None,
    capture: bool = False,
):
    """Forward pass that exposes per-head values for activation patching."""
    x = build_record_inputs(
        model, record_keys, record_values, queries, query_key_patch
    )
    cache = []
    for layer_index, block in enumerate(model.blocks):
        attention_module = block.attention
        batch, length, _ = x.shape
        qkv = attention_module.qkv(block.ln_1(x)).view(
            batch,
            length,
            3,
            attention_module.n_heads,
            attention_module.head_dim,
        )
        if attention_module.tie_heads:
            qkv = qkv[:, :, :, :1, :].expand(
                batch,
                length,
                3,
                attention_module.n_heads,
                attention_module.head_dim,
            )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            attention_module.head_dim
        )
        mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1)
        head_values = torch.matmul(attention, v).transpose(1, 2).contiguous()
        if patch_head is not None:
            patch_layer, patch_index, donor_values = patch_head
            if patch_layer == layer_index:
                head_values = head_values.clone()
                head_values[:, -1, patch_index] = donor_values
        attention_output = attention_module.out(
            head_values.reshape(
                batch,
                length,
                attention_module.n_heads * attention_module.head_dim,
            )
        )
        x = x + attention_output
        if block.use_mlp:
            x = x + block.mlp(block.ln_2(x))
        if capture:
            cache.append(
                {
                    "head_values": head_values,
                    "attention": attention,
                    "residual": x,
                }
            )
    hidden = model.ln_final(x[:, -1])
    logits = torch.stack([head(hidden) for head in model.answer_heads], dim=1)
    return logits, cache


def accuracy(
    logits: torch.Tensor, answers: torch.Tensor
) -> dict[str, float]:
    predictions = logits.argmax(dim=-1)
    first = predictions[:, 0] == answers[:, 0]
    second = predictions[:, 1] == answers[:, 1]
    return {
        "answer_1": float(first.float().mean()),
        "answer_2": float(second.float().mean()),
        "joint": float((first & second).float().mean()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, config, saved = load_model(Path(args.checkpoint), device)
    if config.num_queries != 2:
        raise ValueError(
            "analyze_record_patching currently supports exactly two queries"
        )
    batch = sample_record_batch(
        config,
        args.examples,
        torch.Generator().manual_seed(saved["seed"] + 50_000),
        device,
    )
    donor_queries = batch.queries.flip(1)
    donor_keys = model.key_embedding(donor_queries)
    clean_keys = model.key_embedding(batch.queries)

    with torch.no_grad():
        model_logits, _ = model(
            batch.record_keys, batch.record_values, batch.queries
        )
        clean_logits, clean_cache = manual_forward(
            model,
            batch.record_keys,
            batch.record_values,
            batch.queries,
            capture=True,
        )
        swapped_logits, swapped_cache = manual_forward(
            model,
            batch.record_keys,
            batch.record_values,
            donor_queries,
            capture=True,
        )
        unchanged_logits, _ = manual_forward(
            model,
            batch.record_keys,
            batch.record_values,
            batch.queries,
            query_key_patch=clean_keys,
        )
        query_patches = {}
        for name, slots in (
            ("slot_0", (0,)),
            ("slot_1", (1,)),
            ("both_slots", (0, 1)),
        ):
            patched_keys = clean_keys.clone()
            for slot in slots:
                patched_keys[:, slot] = donor_keys[:, slot]
            patched_logits, _ = manual_forward(
                model,
                batch.record_keys,
                batch.record_values,
                batch.queries,
                query_key_patch=patched_keys,
            )
            patched_predictions = patched_logits.argmax(dim=-1)
            query_patches[name] = {
                **accuracy(patched_logits, batch.answers),
                "answer_1_becomes_query_2": float(
                    (patched_predictions[:, 0] == batch.answers[:, 1])
                    .float()
                    .mean()
                ),
                "answer_2_becomes_query_1": float(
                    (patched_predictions[:, 1] == batch.answers[:, 0])
                    .float()
                    .mean()
                ),
            }

        head_patches = []
        for layer in range(len(model.blocks)):
            for head in range(model.blocks[layer].attention.n_heads):
                donor_head = swapped_cache[layer]["head_values"][:, -1, head]
                patched_logits, _ = manual_forward(
                    model,
                    batch.record_keys,
                    batch.record_values,
                    batch.queries,
                    patch_head=(layer, head, donor_head),
                )
                predictions = patched_logits.argmax(dim=-1)
                head_patches.append(
                    {
                        "layer": layer,
                        "head": head,
                        **accuracy(patched_logits, batch.answers),
                        "answer_1_becomes_query_2": float(
                            (predictions[:, 0] == batch.answers[:, 1])
                            .float()
                            .mean()
                        ),
                        "answer_2_becomes_query_1": float(
                            (predictions[:, 1] == batch.answers[:, 0])
                            .float()
                            .mean()
                        ),
                    }
            )

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "examples": args.examples,
        "manual_matches_model": float(
            (
                clean_logits.argmax(dim=-1)
                == model_logits.argmax(dim=-1)
            )
            .float()
            .mean()
        ),
        "query_patch_no_op": float(
            (
                clean_logits.argmax(dim=-1)
                == unchanged_logits.argmax(dim=-1)
            )
            .float()
            .mean()
        ),
        "clean": accuracy(clean_logits, batch.answers),
        "query_swap": accuracy(swapped_logits, batch.answers.flip(1)),
        "query_slot_patching": query_patches,
        "layer_head_patching_from_swapped_query": head_patches,
    }
    print(json.dumps(result, indent=2))
    if args.result_json:
        path = Path(args.result_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
