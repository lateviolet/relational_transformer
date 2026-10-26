# Generated datasets

Datasets are generated locally rather than committed to Git.

```bash
python scripts/export_pair_data.py --output-dir data/nlp_pair_data
python scripts/export_nlp_data.py --output-dir data/nlp_data
```

The paired-retrieval dataset contains synthetic `reports to` records, held-out
entity names, shuffled facts, seen and unseen prompt templates, query-order
swaps, and longer-context splits.

