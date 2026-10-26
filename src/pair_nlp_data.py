from __future__ import annotations

import json
import random
from pathlib import Path

from nlp_data import HELD_OUT_NAMES, TRAIN_NAMES


PAIR_TEMPLATES = (
    (
        "Use only these facts:\n{facts}\n"
        "Question: Following direct \"reports to\" links, what are the report "
        "targets of {key_1} and {key_2}, respectively? Give exactly two names "
        "in the same order, separated by a comma.\nAnswer:"
    ),
    (
        "Facts:\n{facts}\n"
        "Two lookups are required. Find the person that {key_1} reports to "
        "and the person that {key_2} reports to. Answer with two names only, "
        "in order.\nAnswer:"
    ),
    (
        "Relationship records:\n{facts}\n"
        "Return the direct report targets for {key_1} and {key_2}, in that "
        "order. Separate the two names with a comma.\nResult:"
    ),
)
PAIR_UNSEEN_TEMPLATES = (
    (
        "Available relations:\n{facts}\n"
        "Look up the direct report targets of {key_1} and {key_2}. "
        "Return only those two names, in the requested order.\nAnswer:"
    ),
    (
        "Trace these two independent lookups:\n{facts}\n"
        "What does {key_1} report to, and what does {key_2} report to? "
        "Separate the answers with a comma.\nAnswer:"
    ),
)
PAIR_UNSEEN_RELATION_TEMPLATE = (
    "Use only these facts:\n{facts}\n"
    "Question: Following direct \"supervises\" links, who are the direct "
    "supervisees of {key_1} and {key_2}, respectively? Give exactly two names "
    "in the same order, separated by a comma.\nAnswer:"
)


def make_pair_examples(
    count: int,
    seed: int,
    held_out: bool,
    num_records: int = 16,
) -> list[dict]:
    rng = random.Random(seed)
    names = HELD_OUT_NAMES if held_out else TRAIN_NAMES
    if 2 * num_records > len(names):
        raise ValueError("not enough names for paired-record examples")
    examples = []
    for example_index in range(count):
        selected = rng.sample(list(names), 2 * num_records)
        keys = selected[:num_records]
        values = selected[num_records:]
        rng.shuffle(values)
        facts = list(zip(keys, values))
        rng.shuffle(facts)
        query_indices = rng.sample(range(num_records), 2)
        key_1, answer_1 = facts[query_indices[0]]
        key_2, answer_2 = facts[query_indices[1]]
        fact_text = "\n".join(f"- {key} reports to {value}." for key, value in facts)
        template = PAIR_TEMPLATES[(seed + example_index) % len(PAIR_TEMPLATES)]
        prompt = template.format(
            facts=fact_text,
            key_1=key_1,
            key_2=key_2,
        )
        examples.append(
            {
                "prompt": prompt,
                "answer": f"{answer_1}, {answer_2}",
                "metadata": {
                    "facts": facts,
                    "query_keys": [key_1, key_2],
                    "answers": [answer_1, answer_2],
                    "held_out_names": held_out,
                },
            }
        )
    return examples


def write_splits(output_dir: Path) -> None:
    unseen_base = make_pair_examples(1000, seed=103, held_out=True)
    unseen_template_examples = []
    for index, example in enumerate(unseen_base):
        fact_text = "\n".join(
            f"- {key} reports to {value}."
            for key, value in example["metadata"]["facts"]
        )
        template = PAIR_UNSEEN_TEMPLATES[index % len(PAIR_UNSEEN_TEMPLATES)]
        unseen_template_examples.append(
            {
                **example,
                "prompt": template.format(
                    facts=fact_text,
                    key_1=example["metadata"]["query_keys"][0],
                    key_2=example["metadata"]["query_keys"][1],
                ),
            }
        )
    query_swap_base = make_pair_examples(1000, seed=104, held_out=True)
    query_swap_examples = []
    for index, example in enumerate(query_swap_base):
        facts = example["metadata"]["facts"]
        key_1, key_2 = example["metadata"]["query_keys"]
        answer_1, answer_2 = example["metadata"]["answers"]
        swapped_keys = [key_2, key_1]
        swapped_answers = [answer_2, answer_1]
        template = PAIR_TEMPLATES[index % len(PAIR_TEMPLATES)]
        query_swap_examples.append(
            {
                "prompt": template.format(
                    facts="\n".join(
                        f"- {key} reports to {value}." for key, value in facts
                    ),
                    key_1=swapped_keys[0],
                    key_2=swapped_keys[1],
                ),
                "answer": f"{swapped_answers[0]}, {swapped_answers[1]}",
                "metadata": {
                    **example["metadata"],
                    "query_keys": swapped_keys,
                    "answers": swapped_answers,
                    "query_swap_of_seed": 104,
                },
            }
        )
    relation_base = make_pair_examples(1000, seed=106, held_out=True)
    unseen_relation_examples = []
    for example in relation_base:
        facts = example["metadata"]["facts"]
        unseen_relation_examples.append(
            {
                **example,
                "prompt": PAIR_UNSEEN_RELATION_TEMPLATE.format(
                    facts="\n".join(
                        f"- {key} supervises {value}." for key, value in facts
                    ),
                    key_1=example["metadata"]["query_keys"][0],
                    key_2=example["metadata"]["query_keys"][1],
                ),
            }
        )
    splits = {
        "train": make_pair_examples(12000, seed=101, held_out=False),
        "valid_id": make_pair_examples(1000, seed=102, held_out=True),
        "test_unseen_templates": unseen_template_examples,
        "test_query_swap": query_swap_examples,
        "test_unseen_relation": unseen_relation_examples,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, examples in splits.items():
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"{name}: {len(examples)} examples")
