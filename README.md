# ASL Speech Therapist

This repo supports an end-to-end backend pipeline:

`audio -> ASR -> text normalization -> learned English->ASL gloss model -> structured gloss output`

## Architecture (Current)

```text
src/
  audio/
    record_audio.py
    preprocess_audio.py
    asr.py

  nlp/
    normalize_text.py
    text_to_gloss.py

  asl/
    schema.py
    postprocess_gloss.py
    fallback_rules.py

  models/
    english_to_asl_model.py
    tokenizer_utils.py
    inference.py

  data/
    dataset.py
    preprocess_dataset.py
    collate.py
    prepare_hf_dataset.py

  training/
    train.py
    evaluate.py
    losses.py

  pipeline/
    run_audio_pipeline.py
    run_text_inference.py

  utils/
    config.py
    io.py
    seed.py

data/
  examples/
    toy_asl_pairs.json
  asl_translation/
    train.json
    val.json
```

## Installation

Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` now includes:

- `torch`
- `faster-whisper`
- `datasets`
- core numeric/audio deps

## Dataset Support

The training format remains:

```json
{
  "english": "can you help me today",
  "gloss": "TODAY YOU HELP ME"
}
```

### Prepare ASLG-PC12 from Hugging Face

Dataset: `achrafothman/aslg_pc12`

`src/data/prepare_hf_dataset.py`:

- loads dataset via `datasets.load_dataset`
- maps fields `text -> english`, `gloss -> gloss`
- applies normalization/cleanup
- creates reproducible train/val split
- saves JSON files by default

Command:

```bash
python src/data/prepare_hf_dataset.py \
  --dataset_name achrafothman/aslg_pc12 \
  --output_dir data/asl_translation \
  --val_split 0.1 \
  --seed 42
```

Output files:

- `data/asl_translation/train.json`
- `data/asl_translation/val.json`

## Model (Stronger Default Transformer)

`src/models/english_to_asl_model.py` defaults:

- `d_model=256`
- `nhead=4`
- `num_encoder_layers=3`
- `num_decoder_layers=3`
- `dim_feedforward=512`
- `dropout=0.1`
- `batch_first=True`
- separate `src_pad_idx` and `tgt_pad_idx`

Generation uses greedy decoding for now.

## Training

`src/training/train.py` supports:

- dataset file (`.json/.jsonl/.csv`) or dataset directory (`train.json` + optional `val.json`)
- stronger architecture defaults
- CPU training
- early stopping by validation loss (`--patience`)
- configurable DataLoader workers (`--num_workers`)
- checkpoint save directory (`--save_dir`)

### Train on ASLG-PC12 prepared split

```bash
python src/training/train.py \
  --dataset data/asl_translation \
  --save_dir checkpoints/aslg_pc12_transformer \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-3 \
  --patience 5 \
  --device cpu \
  --num_workers 0
```

### Train on toy dataset (quick smoke test)

```bash
python src/training/train.py \
  --dataset data/examples/toy_asl_pairs.json \
  --epochs 25 \
  --batch_size 16 \
  --device cpu
```

## Text Inference

```bash
python src/pipeline/run_text_inference.py \
  --text "can you help me today" \
  --checkpoint checkpoints/aslg_pc12_transformer/best_model.pt \
  --device cpu
```

Debug mode:

```bash
python src/pipeline/run_text_inference.py \
  --text "can you help me today" \
  --checkpoint checkpoints/aslg_pc12_transformer/best_model.pt \
  --device cpu \
  --debug
```

## Audio Inference

Microphone:

```bash
python src/pipeline/run_audio_pipeline.py \
  --mic \
  --checkpoint checkpoints/aslg_pc12_transformer/best_model.pt \
  --device cpu
```

Audio file:

```bash
python src/pipeline/run_audio_pipeline.py \
  --audio_file example.wav \
  --checkpoint checkpoints/aslg_pc12_transformer/best_model.pt \
  --device cpu
```

## Notes

- This keeps the existing audio->ASR->model inference flow.
- Fallback rules are still available for comparison/debugging, but learned translation is the primary path.
- This repo does not yet include sign animation or CV sign-feedback modules.
