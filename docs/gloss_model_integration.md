# Gloss Model Integration

## Model

- Model name: `project_finetune_v2_v4_contrastive`
- Checkpoint path: `checkpoints/project_finetune_v2_v4_contrastive/best_model.pt`
- Dataset: `data/active/project_finetune_v2_v4_contrastive.json`
- Performance: `87.8%` test_batch accuracy (`36/41`), average BLEU about `0.67`

## Task

This is a text-to-gloss model:

```text
English text -> ASL gloss
```

Example:

```text
Input: i need help
Output: I NEED HELP
```

It does not perform video-based sign recognition.

## Local Demo

```bash
python scripts/demo_gloss_translate.py --text "i need help"
```

Expected output:

```text
Input: i need help
ASL Gloss: I NEED HELP
```

## Python API

Use the lightweight service wrapper:

```python
from src.services.gloss_inference import translate_text_to_gloss

gloss = translate_text_to_gloss("i need help")
print(gloss)
```

The wrapper caches the model load, so repeated calls reuse the checkpoint in memory.

## Website Integration

Call this model after the app has English text:

- direct text input -> this model -> ASL gloss
- speech -> ASR -> text -> this model -> ASL gloss

Do not call this model on video frames. If the frontend is using webcam input, the pipeline should be either:

```text
video -> vision model -> predicted sign classification
```

or:

```text
speech -> ASR -> text -> this model -> ASL gloss
```

The website issue where `HELP` is classified as `RICH 88%` is from the vision model path, not this text-to-gloss model.
