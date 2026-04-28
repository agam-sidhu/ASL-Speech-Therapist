# Gloss Data Layout

The active demo pipeline uses only three data locations:

- `raw/`: source inputs and provenance files.
- `active/`: final datasets consumed by training.
- `reports/`: rebuild/audit output.

Everything else is historical and lives under `archive/`. Archived files are not read by the active pipeline.

## Active Files

- `active/aslg_pc12_pretrain.json`: cleaned ASLG-PC12 pairs for pretraining only.
- `active/project_finetune_v2_v4_contrastive.json`: clean v2+v4 conversational data plus generated and contrastive augmentation for fine-tuning.
- `reports/data_pipeline_report.json`: counts, duplicate/conflict checks, and active output paths.

## Raw Inputs

- `raw/train.csv`: raw ASLG-PC12 CSV.
- `raw/asl_gloss_conversational.json`: clean v2+v4 project base.
- `raw/asl_gloss_pairs_v2.json`: v2 provenance.
- `raw/asl_gloss_pairs_v4.json`: v4 provenance.

The v3 source is intentionally archived because it introduced conflicting gloss order. It must not be used by active build, training, or evaluation commands.

## Rebuild

```bash
python scripts/run_full_pipeline.py --stage build-data
```

This writes:

- `data/active/aslg_pc12_pretrain.json`
- `data/active/project_finetune_v2_v4_contrastive.json`
- `data/reports/data_pipeline_report.json`

## Train And Evaluate

```bash
python scripts/run_full_pipeline.py --stage pretrain
python scripts/run_full_pipeline.py --stage finetune
python scripts/run_full_pipeline.py --stage evaluate
```

Run the full sequence with:

```bash
python scripts/run_full_pipeline.py --stage all
```
