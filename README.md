# ASL Speech Therapist

## Project Shift: Baseline -> ML Translation Scaffold

The original baseline focused on:

`audio -> ASR -> normalization -> mostly rule-based gloss`

This refactor changes the core design to match the real research/engineering goal:

`audio -> ASR -> normalized English text -> learned English-to-ASL model -> structured ASL output`

The key architectural decision is that **English-to-ASL is now modeled as a learned sequence prediction task**, not a dictionary/rule system.

## Demo Gloss Translation Model

The demo checkpoint is:

```text
checkpoints/project_finetune_v2_v4_contrastive/best_model.pt
```

It performs:

```text
English text -> ASL gloss
```

Example:

```text
i need help -> I NEED HELP
```

It does not perform video-to-sign recognition. Webcam classification issues such as `HELP -> RICH 88%` come from the vision model path, not this text-to-gloss model.

Quick local demo:

```bash
python scripts/demo_gloss_translate.py --text "i need help"
```

Frontend integration notes are in [docs/gloss_model_integration.md](docs/gloss_model_integration.md).

## Scope Boundaries

- `src/services/`, `src/audio/`, `src/nlp/`, `src/models/`, `src/data/`, `src/evaluation/`, and `src/training/` are the audio/text -> ASL translation side.
- `src/ASL_visual_recognition/` is preserved for the separate vision-based sign-recognition track. This repo cleanup does not delete or rewrite that module.
- WLASL belongs to the later sign/gloss mapping side, not ASR training.

## New Architecture

```text
src/
  services/
    asl_pipeline.py            # shared runtime service used by CLI/demo scripts

  audio/
    record_audio.py
    preprocess_audio.py
    asr.py

  nlp/
    normalize_text.py
    text_to_gloss.py            # deprecated compatibility wrapper

  asl/
    schema.py
    postprocess_gloss.py
    fallback_rules.py           # debug/baseline only

  models/
    english_to_asl_model.py
    tokenizer_utils.py
    inference.py

  data/
    dataset.py
    preprocess_dataset.py
    collate.py
    splits.py

  training/
    train.py
    evaluate.py
    losses.py

  evaluation/
    audit_translation_dataset.py
    evaluate_translation.py
    evaluate_grammar_challenge.py
    evaluate_asr.py
    translation_analysis.py
    asr_metrics.py

  pipeline/
    run_audio_pipeline.py
    run_text_inference.py

  utils/
    config.py
    io.py
    seed.py

  ASL_visual_recognition/      # preserved vision assets / teammate-owned track

data/
  raw/                           # raw ASLG-PC12 and v2/v4 project provenance
  active/                        # final training datasets
  reports/                       # reproducible data build report
  archive/                       # historical unused files, including v3

requirements.txt
README.md
```

## Why This Design Changed

Rule-based gloss conversion is useful as a quick baseline, but it does not scale to real translation quality or richer ASL representations.

This refactor introduces:

- trainable seq2seq model scaffolding
- paired dataset loading pipeline
- training and validation loops with checkpoints
- model-based text and audio inference
- structured output schema for downstream modules

## Current Model Baseline

The default model is a **small Transformer encoder-decoder** (`EnglishToASLTransformer`) implemented with PyTorch.

Default toy-data hyperparameters are intentionally small:

- `d_model=64`
- `nhead=2`
- `num_encoder_layers=1`
- `num_decoder_layers=1`
- `dim_feedforward=128`

Input: normalized English token ids  
Output: ASL gloss token sequence

## Installation

Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Format

Paired records are expected in JSON/JSONL/CSV with fields:

```json
{
  "english": "can you help me today",
  "gloss": "TODAY YOU HELP ME"
}
```

Active datasets:

- **Pretraining**: `data/active/aslg_pc12_pretrain.json`
- **Fine-tuning/demo**: `data/active/project_finetune_v2_v4_contrastive.json`

Raw inputs live in `data/raw/`. Historical files, including v3, live in `data/archive/` and are not read by the active pipeline. v3 is archived because it introduced conflicting ASL gloss order.

Rebuild active data:

```bash
python scripts/run_full_pipeline.py --stage build-data
```

## How To Train

**Pretrain on ASLG-PC12**

```bash
python scripts/run_full_pipeline.py --stage pretrain
```

**Fine-tune on clean v2+v4 project data**

```bash
python scripts/run_full_pipeline.py --stage finetune
```

The fine-tuning dataset includes clean v2+v4 conversational pairs, 150 generated augmentation pairs, and 200 contrastive pairs.

Training still uses:

- Uses reproducible train/validation/test partitioning when `--test_split` is enabled
- Larger model (128-dim embeddings, 2 encoder/decoder layers)
- Cosine annealing learning rate with 8-epoch warmup
- Gradient clipping (max norm 1.0) for stability
- Label smoothing (0.1) for better generalization
- Pretraining checkpoint: `checkpoints/aslg_pc12_pretrain/best_model.pt`
- Fine-tuned checkpoint: `checkpoints/project_finetune_v2_v4_contrastive/best_model.pt`

## Overfit Sanity-Check Mode

Use this to verify the model can memorize tiny data:

```bash
python src/training/train.py \
  --mode finetune \
  --dataset data/active/project_finetune_v2_v4_contrastive.json \
  --epochs 200 \
  --batch_size 2 \
  --device cpu \
  --max_train_samples 2 \
  --no_val
```

Options:

- `--max_train_samples N`: truncate training split to first N examples.
- `--no_val`: skip validation completely and checkpoint by train loss.

## How To Run Text Inference

### Interactive Testing (Recommended)

Test the model interactively with any English phrase:

```bash
python test_examples.py --checkpoint checkpoints/best_model.pt --beam_width 5
```

This launches an interactive prompt:

```
📝 English text: where do you work
✓ Translation:
   ASL Gloss: YOU WORK WHERE
   Tokens: ['YOU', 'WORK', 'WHERE']

📝 English text: i love music
✓ Translation:
   ASL Gloss: I LOVE MUSIC
   Tokens: ['I', 'LOVE', 'MUSIC']
```

Type `exit` or `quit` to stop.

### Batch Testing (Predefined Categories)

Test on predefined phrase categories with accuracy metrics:

```bash
# All categories
python test_batch.py --checkpoint checkpoints/best_model.pt --beam_width 3

# Specific category (greetings, questions, emotions, learning, daily_activities, etc.)
python test_batch.py --checkpoint checkpoints/best_model.pt --category greetings

# Show all results (not just failures)
python test_batch.py --checkpoint checkpoints/best_model.pt --show_all
```

### Single Input

```bash
python src/pipeline/run_text_inference.py \
  --text "can you help me today" \
  --checkpoint checkpoints/best_model.pt \
  --device cpu
```

### With Beam Search

Test with different beam widths (higher = better quality, slower):

```bash
# Greedy decoding (fastest)
python src/pipeline/run_text_inference.py \
  --text "where is the library" \
  --checkpoint checkpoints/best_model.pt

# Beam width 3 (recommended)
python src/pipeline/run_text_inference.py \
  --text "where is the library" \
  --checkpoint checkpoints/best_model.pt --beam_width 3

# Beam width 5 (slowest, best quality)
python src/pipeline/run_text_inference.py \
  --text "where is the library" \
  --checkpoint checkpoints/best_model.pt --beam_width 5
```

### Full Evaluation

Run full evaluation on entire dataset with BLEU scores:

```bash
python src/training/evaluate_checkpoint.py \
  --checkpoint checkpoints/project_finetune_v2_v4_contrastive/best_model.pt \
  --dataset data/active/project_finetune_v2_v4_contrastive.json \
  --beam_width 3 \
  --show_examples 20
```

Output:

```
Corpus BLEU: 0.8674
Exact match accuracy: 326/396 (82.3%)
1-gram precision: 0.9335
2-gram precision: 0.8849
...
```

### Debug Mode

```bash
python src/pipeline/run_text_inference.py \
  --text "can you help me today" \
  --checkpoint checkpoints/best_model.pt \
  --debug
```

Debug JSON fields include:

- `normalized_input_text`
- `source_tokens`
- `source_ids`
- `raw_generated_ids`
- `raw_decoded_tokens`
- `cleaned_gloss_tokens`
- `empty_after_postprocess`

Runtime JSON also includes:

- `normalized_tokens`
- `gloss_items` with per-token `lookup_key` values for later sign mapping

## How To Run Audio Inference

Microphone mode:

```bash
python src/pipeline/run_audio_pipeline.py --mic --checkpoint checkpoints/best_model.pt
```

Audio file mode:

```bash
python src/pipeline/run_audio_pipeline.py --audio_file example.wav --checkpoint checkpoints/best_model.pt
```

Audio debug mode:

```bash
python src/pipeline/run_audio_pipeline.py \
  --audio_file example.wav \
  --checkpoint checkpoints/best_model.pt \
  --debug
```

Pipeline:

`mic/audio -> preprocess (mono 16kHz) -> Whisper ASR -> normalize -> learned model inference`

The shared runtime source of truth is `src/services/asl_pipeline.py`, which exposes:

- `run_text_to_asl(...)`
- `run_audio_to_asl(...)`

CLI wrappers in `src/pipeline/` are intentionally thin wrappers around that service.

## Translation Evaluation

Use the translation evaluator for overlap metrics plus grammar-oriented diagnostics:

```bash
python src/evaluation/evaluate_translation.py \
  --checkpoint checkpoints/project_finetune_v2_v4_contrastive/best_model.pt \
  --dataset data/active/project_finetune_v2_v4_contrastive.json \
  --beam_width 3 \
  --split all
```

This reports:

- corpus BLEU
- exact match
- aligned token accuracy
- token-overlap F1
- English-order copy rate
- function-word leak rate
- reorder-sensitive case count
- reference-order success on reorder-sensitive cases
- category breakdown using either inferred dataset categories or curated challenge tags

## Grammar Challenge Evaluation

Use the curated challenge set to stress-test grammar phenomena directly:

```bash
python src/evaluation/evaluate_grammar_challenge.py \
  --checkpoint checkpoints/best_model.pt \
  --beam_width 3 \
  --show_failures_only
```

The challenge set focuses on:

- WH-questions
- yes/no question endings
- negation
- time-fronting
- function-word deletion
- topic-comment-like possessive phrases
- strong English-to-gloss reordering

## Dataset Audit

Audit the paired data itself before claiming the model has learned ASL grammar:

```bash
python src/evaluation/audit_translation_dataset.py \
  --dataset data/active/project_finetune_v2_v4_contrastive.json \
  --val_split 0.15 \
  --test_split 0.10
```

This reports:

- corpus size and vocabulary size
- source/target length statistics
- counts of copy/reorder/function-word-drop style patterns
- coarse grammar-template frequencies
- repeated gloss forms
- split-overlap risk under the current split configuration
- example pairs for each detected grammar category

## ASR Evaluation

The repo currently treats ASR as inference-only. You can still evaluate transcript quality on labeled audio using a manifest file with `audio_path` and `reference_text`:

```bash
python src/evaluation/evaluate_asr.py \
  --manifest your_asr_manifest.json \
  --model_size base
```

This reports:

- WER
- CER
- transcript examples with normalized reference/hypothesis text

Manifest records should contain:

```json
{
  "audio_path": "path/to/example.wav",
  "reference_text": "where is the bathroom"
}
```

## Fallback Rules (Debug Only)

A fallback module exists in `src/asl/fallback_rules.py` for debugging and comparison.

It is **not** the main translation path. Use explicitly:

```bash
python src/pipeline/run_text_inference.py --text "hello" --use_fallback
```

## Debugging Workflow (Recommended)

1. Train on toy dataset and confirm checkpoint is produced.
2. Run inference on exact training phrases.
3. Use `--debug` and check whether model emits `<eos>` immediately.
4. If needed, run overfit mode with `--max_train_samples 2 --no_val`.
5. Compare learned output and fallback output using `--include_fallback_compare`.

## Model Performance

**Historical Baseline Results (Older v2 Dataset)**

- **Corpus BLEU**: 0.8674
- **Exact Match Accuracy**: 82.3% (326/396 examples)
- **1-gram Precision**: 0.9335
- **2-gram Precision**: 0.8849
- **Model Size**: 803K parameters
- **Training Time**: ~40 minutes on CPU (80 epochs)

The model successfully learns ASL grammatical patterns including:

- Topic-comment structure (time references first)
- WH-question fronting (interrogatives at end)
- Copula/article deletion
- Word reordering for ASL syntax

## Current Limitations

- Gloss text output only (not sign video/animation)
- ASR is inference-only in this repo; no Whisper training/fine-tuning pipeline is included
- No phoneme-level pronunciation scoring yet
- No computer-vision sign feedback yet
- Greedy + beam search decoding (no advanced techniques like length normalization tuning)
- Still limited by small-to-medium paired data, with heavy template overlap across random splits even after cleanup

## Future Extensions

1. **Data**: Integration with ASLG-PC12 corpus (~87K pairs) for large-scale training
2. **Modeling**: Subword tokenization (BPE), larger models, pretrained encoders
3. **Decoding**: Advanced beam search, diverse beam search, minimum risk training
4. **Integration**: Sign video generation, computer vision feedback, pronunciation scoring
5. **Representation**: Non-manual markers, spatial positioning, classifier expressions
6. **Personalization**: LoRA-based fine-tuning for individual user adaptation

## Notes

- Microphone permission prompts are handled by OS/browser/device settings through `sounddevice`.
- For notebook/Colab workflows, audio file mode is usually more reliable than direct mic capture.
- All pipeline scripts return JSON-serializable outputs for easier integration with future components.
