# ASL Text-to-Gloss Colab Experiment

Minimal package for the clean demo pipeline:

1. Pretrain on `data/active/aslg_pc12_pretrain.json`
2. Fine-tune on `data/active/project_finetune_v2_v4_contrastive.json`
3. Evaluate with `test_batch.py`

The archived v3 data is not used.

## Setup

```bash
!bash colab_setup.sh
```

## Pretrain

```bash
!python src/training/train.py \
  --mode pretrain \
  --dataset data/active/aslg_pc12_pretrain.json \
  --device cuda
```

If Colab GPU is unavailable, use `--device cpu`.

## Fine-Tune

```bash
!python src/training/train.py \
  --mode finetune \
  --dataset data/active/project_finetune_v2_v4_contrastive.json \
  --init_checkpoint checkpoints/aslg_pc12_pretrain/best_model.pt \
  --device cuda
```

## Evaluate

```bash
!python test_batch.py --checkpoint checkpoints/aslg_pc12_pretrain/best_model.pt --beam_width 3
!python test_batch.py --checkpoint checkpoints/project_finetune_v2_v4_contrastive/best_model.pt --beam_width 3
```

## Package Results

```bash
!bash zip_results.sh
```
