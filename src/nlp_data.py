from __future__ import annotations

import json
import random
from pathlib import Path


RELATIONS = ("reports to", "mentors")
UNSEEN_RELATIONS = ("supports", "advises")
TRAIN_NAMES = (
    "Alice",
    "Bruno",
    "Chloe",
    "Diego",
    "Elena",
    "Farid",
    "Grace",
    "Hugo",
    "Iris",
    "Jonas",
    "Kira",
    "Liam",
    "Maya",
    "Nico",
    "Olga",
    "Pablo",
    "Quinn",
    "Rina",
    "Samir",
    "Tara",
    "Umar",
    "Vera",
    "Wen",
    "Xavier",
    "Yara",
    "Zane",
    "Aiko",
    "Benedikt",
    "Camila",
    "Dmitri",
    "Emeka",
    "Fatima",
)
HELD_OUT_NAMES = (
    "Gita",
    "Hassan",
    "Ines",
    "Jae",
    "Kofi",
    "Lucia",
    "Mateo",
    "Noor",
    "Orla",
    "Priya",
    "Rafael",
    "Sofia",
    "Tariq",
    "Uma",
    "Victor",
    "Yuki",
    "Zoya",
    "Amara",
    "Bilal",
    "Cyrus",
    "Dalia",
    "Eitan",
    "Freya",
    "Gideon",
    "Hana",
    "Idris",
    "Jamal",
    "Keiko",
    "Leila",
    "Mina",
    "Nadir",
    "Omar",
    "Pavel",
    "Rhea",
    "Sami",
    "Talia",
    "Vikram",
    "Willa",
    "Yusuf",
    "Zara",
    "Avery",
    "Bianka",
    "Cesar",
    "Daria",
    "Emil",
    "Fiona",
    "Gustav",
    "Helena",
    "Imani",
    "Jasper",
    "Kalani",
    "Lorenzo",
    "Mira",
    "Nikolai",
    "Opal",
    "Petra",
    "Rami",
    "Selin",
    "Teo",
    "Ulyana",
    "Valeria",
    "Winston",
    "Yosef",
    "Zoe",
)

TEMPLATES = (
    (
        "You are tracing a network of relationships.\n"
        "Facts:\n{facts}\n"
        "Question: Starting at {start}, follow exactly two \"{relation}\" "
        "links. Which person do you reach?\nAnswer:"
    ),
    (
        "Use only the facts below to answer.\n{facts}\n"
        "If you begin with {start} and take two \"{relation}\" edges, "
        "who is reached? Give only the name.\nAnswer:"
    ),
    (
        "Relationship records:\n{facts}\n"
        "Trace two edges of type \"{relation}\", starting from {start}. "
        "What is the endpoint?\nEndpoint:"
    ),
)
UNSEEN_TEMPLATES = (
    (
        "Relations:\n{facts}\n"
        "Start at {start}. Apply exactly two consecutive \"{relation}\" "
        "relations. Return only the endpoint name.\nEndpoint:"
    ),
    (
        "We have the following directed facts.\n{facts}\n"
        "What entity is obtained after taking two \"{relation}\" steps "
        "from {start}? Do not explain.\nResult:"
    ),
)


def _pick_names(names: tuple[str, ...], count: int, rng: random.Random) -> list[str]:
    if count > len(names):
        raise ValueError("not enough held-out names for this task")
    return rng.sample(list(names), count)


def make_examples(
    count: int,
    seed: int,
    distractor_count: int,
    held_out: bool = False,
    template_pool: tuple[str, ...] = TEMPLATES,
    relation_pool: tuple[str, ...] = RELATIONS,
) -> list[dict]:
    rng = random.Random(seed)
    names = HELD_OUT_NAMES if held_out else TRAIN_NAMES
    examples: list[dict] = []
    required_names = 6 + 2 * distractor_count
    if required_names > len(names):
        raise ValueError(
            f"{distractor_count} distractors require {required_names} names, "
            f"but only {len(names)} are available"
        )

    for example_index in range(count):
        active_relation = rng.choice(relation_pool)
        other_relation = next(
            relation for relation in relation_pool if relation != active_relation
        )
        selected = _pick_names(names, required_names, rng)
        start, intermediate, answer = selected[:3]
        other_start, other_intermediate, other_answer = selected[3:6]
        edges = [
            (start, intermediate, active_relation),
            (intermediate, answer, active_relation),
            (other_start, other_intermediate, other_relation),
            (other_intermediate, other_answer, other_relation),
        ]
        for index in range(distractor_count):
            key = selected[6 + 2 * index]
            value = selected[7 + 2 * index]
            relation = active_relation if index % 2 == 0 else other_relation
            edges.append((key, value, relation))
        rng.shuffle(edges)

        fact_lines = [
            f"- {key} {relation} {value}." for key, value, relation in edges
        ]
        template = template_pool[(seed + example_index) % len(template_pool)]
        prompt = template.format(
            facts="\n".join(fact_lines),
            start=start,
            relation=active_relation,
        )
        alternative_prompt = template.format(
            facts="\n".join(fact_lines),
            start=other_start,
            relation=other_relation,
        )
        examples.append(
            {
                "prompt": prompt,
                "answer": answer,
                "alternative_prompt": alternative_prompt,
                "alternative_answer": other_answer,
                "metadata": {
                    "start": start,
                    "intermediate": intermediate,
                    "answer": answer,
                    "active_relation": active_relation,
                    "other_start": other_start,
                    "other_intermediate": other_intermediate,
                    "other_answer": other_answer,
                    "distractor_count": distractor_count,
                    "held_out_names": held_out,
                    "edges": edges,
                },
            }
        )
    return examples


def write_splits(output_dir: Path) -> None:
    splits = {
        "train": make_examples(12000, seed=1, distractor_count=3),
        "valid_id": make_examples(
            1000, seed=2, distractor_count=3, held_out=True
        ),
        "test_more_distractors": make_examples(
            1000, seed=3, distractor_count=8, held_out=True
        ),
        "test_unseen_templates": make_examples(
            1000,
            seed=4,
            distractor_count=3,
            held_out=True,
            template_pool=UNSEEN_TEMPLATES,
        ),
        "test_16_distractors": make_examples(
            1000, seed=5, distractor_count=16, held_out=True
        ),
        "test_unseen_relations": make_examples(
            1000,
            seed=6,
            distractor_count=3,
            held_out=True,
            relation_pool=UNSEEN_RELATIONS,
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, examples in splits.items():
        path = output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"{name}: {len(examples)} examples -> {path}")
