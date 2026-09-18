# NSTrainer

Training infrastructure for the OOM model family. Implements NS-filtered fine-tuning: scoring and selecting training samples by importance so the model reaches a target quality in far fewer steps - plus knowledge-distillation and analytical weight-init experiments. Demonstrated NS-filtered training matching a 1000-step baseline at step 132 (86.8% fewer steps) on the C4 dataset.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python3 src/ns_train.py --dataset c4 --scorer bigram
```

See `data/DATASETS.md` for required datasets before running any training scripts. Training checkpoints and large artifacts are written to `checkpoints/` and `results/` (both gitignored).

## Layout

- `src/` - training, distillation, ablation, and analysis scripts (`run_all.sh` is the RunPod pipeline entry point)
- `tests/` - test/benchmark scripts (run with `PYTHONPATH=src`)
- `docs/` - `paper_draft.md`, methodology and results writeup
- `data/` - dataset docs, download script, and sample corpora
- `cpp/` - C++ KV-box / llama components
- `checkpoints/`, `results/`: generated artifacts (gitignored)

## Methodology

See `docs/paper_draft.md` for the full methodology, ablation results, and the NS-filtering proof.

## License

OOM Commercial License v1.0.
