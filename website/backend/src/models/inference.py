"""Inference helpers for the learned English->ASL model."""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
import time
from typing import Callable

import torch

from src.asl.postprocess_gloss import clean_gloss_tokens, to_gloss_text
from src.asl.schema import ASLPrediction
from src.models.english_to_asl_model import EnglishToASLTransformer
from src.models.tokenizer_utils import SimpleWhitespaceTokenizer, Vocab
from src.nlp.normalize_text import normalize_text


@dataclass
class InferenceBundle:
    """Loaded model artifacts used by inference scripts and pipelines."""

    model: EnglishToASLTransformer
    src_tokenizer: SimpleWhitespaceTokenizer
    tgt_tokenizer: SimpleWhitespaceTokenizer
    src_vocab: Vocab
    tgt_vocab: Vocab
    model_name: str


def _trace_enabled(debug: bool) -> bool:
    env_flag = os.getenv("ASL_INFERENCE_TRACE", "0").strip().lower()
    return debug or env_flag in {"1", "true", "yes", "on"}


def _trace(
    enabled: bool,
    message: str,
    start_ts: float | None = None,
) -> float:
    now = time.perf_counter()
    if enabled:
        if start_ts is None:
            print(f"[inference] {message}", file=sys.stderr, flush=True)
        else:
            elapsed_ms = (now - start_ts) * 1000
            print(f"[inference] {message} ({elapsed_ms:.2f} ms)", file=sys.stderr, flush=True)
    return now


def load_inference_bundle(checkpoint_path: str, device: str = "cpu") -> InferenceBundle:
    """Load model/vocabs/tokenizers from a saved training checkpoint."""
    trace = _trace_enabled(debug=False)
    t0 = _trace(trace, f"load_inference_bundle:start checkpoint={checkpoint_path} device={device}")

    payload = torch.load(checkpoint_path, map_location=device)
    t1 = _trace(trace, "checkpoint loaded via torch.load", t0)

    model_config = payload["model_config"]
    src_vocab = Vocab.from_dict(payload["src_vocab"])
    tgt_vocab = Vocab.from_dict(payload["tgt_vocab"])

    src_tokenizer = SimpleWhitespaceTokenizer(**payload["src_tokenizer"])
    tgt_tokenizer = SimpleWhitespaceTokenizer(**payload["tgt_tokenizer"])

    # Backward compatible with checkpoints saved before src/tgt pad split.
    src_pad_idx = model_config.get("src_pad_idx", src_vocab.pad_idx)
    tgt_pad_idx = model_config.get("tgt_pad_idx", tgt_vocab.pad_idx)

    model = EnglishToASLTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        d_model=model_config["d_model"],
        nhead=model_config["nhead"],
        num_encoder_layers=model_config["num_encoder_layers"],
        num_decoder_layers=model_config["num_decoder_layers"],
        dim_feedforward=model_config["dim_feedforward"],
        dropout=model_config["dropout"],
    )
    t2 = _trace(trace, "model skeleton created", t1)

    model.load_state_dict(payload["model_state_dict"])
    t3 = _trace(trace, "state_dict loaded", t2)

    model.to(device)
    model.eval()
    _trace(trace, "model moved to device and set to eval", t3)

    model_name = payload.get("model_name", "english_to_asl_transformer")

    return InferenceBundle(
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        model_name=model_name,
    )


def predict_gloss(
    text: str,
    bundle: InferenceBundle,
    device: str = "cpu",
    max_len: int = 32,
    debug: bool = False,
    beam_width: int = 1,
) -> ASLPrediction:
    """Run end-to-end text inference using the learned translation model.

    Args:
        text: Input English text.
        bundle: Loaded model artifacts.
        device: Device to run inference on.
        max_len: Maximum output sequence length.
        debug: If True, include detailed debug info in output.
        beam_width: Beam search width. 1 = greedy, >1 = beam search.
    """
    trace = _trace_enabled(debug=debug)
    t0 = _trace(trace, f"predict_gloss:start text_len={len(text)} beam_width={beam_width} max_len={max_len}")

    normalized = normalize_text(text)
    t1 = _trace(trace, "normalize_text complete", t0)

    clean_text = normalized["clean_text"]

    src_tokens = bundle.src_tokenizer.tokenize(clean_text)
    t2 = _trace(trace, f"tokenization complete token_count={len(src_tokens)}", t1)

    src_ids = [bundle.src_vocab.bos_idx]
    src_ids += bundle.src_vocab.encode(src_tokens)
    src_ids += [bundle.src_vocab.eos_idx]
    t3 = _trace(trace, f"vocab encoding complete id_count={len(src_ids)}", t2)

    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    t4 = _trace(trace, f"input tensor prepared shape={tuple(src_tensor.shape)}", t3)

    # If inference blocks, this is the last trace line we will see.
    _trace(trace, "calling model.generate")
    generated = bundle.model.generate(
        src_ids=src_tensor,
        bos_idx=bundle.tgt_vocab.bos_idx,
        eos_idx=bundle.tgt_vocab.eos_idx,
        max_len=max_len,
        beam_width=beam_width,
    )
    t5 = _trace(trace, "model.generate returned", t4)

    generated_ids = generated.squeeze(0).tolist()
    t6 = _trace(trace, f"generated ids extracted count={len(generated_ids)}", t5)

    raw_tokens = bundle.tgt_vocab.decode(generated_ids)
    t7 = _trace(trace, f"decoded raw tokens count={len(raw_tokens)}", t6)

    gloss_tokens = clean_gloss_tokens(raw_tokens)
    empty_after_postprocess = len(gloss_tokens) == 0
    t8 = _trace(trace, f"postprocess complete gloss_count={len(gloss_tokens)}", t7)

    debug_info = None
    if debug or empty_after_postprocess:
        debug_info = {
            "normalized_input_text": clean_text,
            "source_tokens": src_tokens,
            "source_ids": src_ids,
            "raw_generated_ids": generated_ids,
            "raw_decoded_tokens": raw_tokens,
            "cleaned_gloss_tokens": gloss_tokens,
            "trace_enabled": trace,
        }

    _trace(trace, "predict_gloss:end", t8)

    return ASLPrediction(
        clean_text=clean_text,
        predicted_gloss_tokens=gloss_tokens,
        predicted_gloss_text=to_gloss_text(gloss_tokens),
        model_name=bundle.model_name,
        used_fallback=False,
        empty_after_postprocess=empty_after_postprocess,
        debug_info=debug_info,
    )
