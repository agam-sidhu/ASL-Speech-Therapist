"""Training entry point for English->ASL gloss seq2seq model.

Improvements over the original baseline:
- Learning rate scheduling (cosine annealing with warmup)
- Gradient clipping to prevent exploding gradients
- Label smoothing for better generalization
- BLEU score evaluation during validation
- Scaled-up default hyperparameters for real dataset training
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    DEFAULT_ASLG_PRETRAIN_CHECKPOINT_DIR,
    DEFAULT_ASLG_PRETRAIN_DATASET_PATH,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_MASTER_DATASET_PATH,
    DEFAULT_PROJECT_FINETUNE_CHECKPOINT_DIR,
    DEFAULT_PROJECT_FINETUNE_DATASET_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train English->ASL gloss transformer.")
    parser.add_argument("--mode", choices=["pretrain", "finetune"], default="finetune")
    parser.add_argument("--dataset", default=None, help="Path to paired dataset file")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument(
        "--log_path",
        default=None,
        help="Optional JSONL file path for per-epoch metrics and run metadata.",
    )
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        help="Optional checkpoint to warm-start from. Supports partial vocab transfer.",
    )

    # Scaled-up model defaults for real dataset.
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--num_decoder_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument(
        "--tiny_model",
        action="store_true",
        help="Force tiny toy-data-friendly architecture defaults.",
    )

    # Training improvements
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor.")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs for LR scheduling.")
    parser.add_argument(
        "--bleu_eval_every",
        type=int,
        default=10,
        help="Evaluate generation BLEU every N epochs. Set to 1 for checkpointing by BLEU every epoch.",
    )
    parser.add_argument(
        "--bleu_eval_samples",
        type=int,
        default=20,
        help="Number of validation examples to use for BLEU. Use 0 to evaluate all validation examples.",
    )
    parser.add_argument(
        "--early_stopping_metric",
        choices=["train_loss", "val_loss", "val_token_accuracy", "val_bleu"],
        default=None,
        help="Optional metric for early stopping. Disabled by default.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Stop after this many checks without improvement on --early_stopping_metric. 0 disables it.",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.0,
        help="Minimum metric improvement required to reset early stopping patience.",
    )

    parser.add_argument("--max_src_len", type=int, default=64)
    parser.add_argument("--max_tgt_len", type=int, default=64)

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="If set, truncate training set to first N samples (debug/overfit mode).",
    )
    parser.add_argument(
        "--no_val",
        action="store_true",
        help="Disable validation loop and checkpoint on train loss.",
    )
    return parser.parse_args()


def resolve_mode_defaults(args: argparse.Namespace) -> None:
    """Fill mode-specific defaults while preserving explicit CLI overrides."""
    if args.mode == "pretrain":
        if args.dataset is None:
            args.dataset = str(DEFAULT_ASLG_PRETRAIN_DATASET_PATH)
        if args.save_dir is None:
            args.save_dir = str(DEFAULT_ASLG_PRETRAIN_CHECKPOINT_DIR)
        return

    if args.dataset is None:
        args.dataset = str(DEFAULT_PROJECT_FINETUNE_DATASET_PATH)
    if args.save_dir is None:
        args.save_dir = str(DEFAULT_PROJECT_FINETUNE_CHECKPOINT_DIR)
    if args.init_checkpoint is None:
        args.init_checkpoint = str(DEFAULT_ASLG_PRETRAIN_CHECKPOINT_DIR / "best_model.pt")


def validate_training_dataset_path(dataset_path: str) -> None:
    parts = Path(dataset_path).parts
    for blocked in (("data", "archive"), ("data", "review")):
        if any(parts[index : index + 2] == blocked for index in range(max(len(parts) - 1, 0))):
            raise ValueError(f"Active training cannot read from {'/'.join(blocked)}: {dataset_path}")


def maybe_apply_tiny_model(args: argparse.Namespace) -> None:
    """Apply tiny model config if explicitly requested."""
    if not args.tiny_model:
        return
    args.d_model = 64
    args.nhead = 2
    args.num_encoder_layers = 1
    args.num_decoder_layers = 1
    args.dim_feedforward = 128


def warm_start_from_checkpoint(
    *,
    model,
    checkpoint_path: str,
    src_vocab,
    tgt_vocab,
    device: str,
) -> dict[str, int | str]:
    """Partially initialize a model from a checkpoint with possibly different vocabs."""
    import torch

    from src.models.tokenizer_utils import Vocab

    payload = torch.load(checkpoint_path, map_location=device)
    old_state = payload["model_state_dict"]
    old_src_vocab = Vocab.from_dict(payload["src_vocab"])
    old_tgt_vocab = Vocab.from_dict(payload["tgt_vocab"])

    new_state = model.state_dict()
    copied_tensors = 0
    skipped_tensors = 0

    for name, tensor in old_state.items():
        if name in {
            "src_embedding.weight",
            "tgt_embedding.weight",
            "output_projection.weight",
            "output_projection.bias",
        }:
            continue
        if name in new_state and new_state[name].shape == tensor.shape:
            new_state[name] = tensor.clone()
            copied_tensors += 1
        else:
            skipped_tensors += 1

    source_tokens_copied = 0
    if "src_embedding.weight" in old_state and new_state["src_embedding.weight"].shape[1:] == old_state["src_embedding.weight"].shape[1:]:
        src_embedding = new_state["src_embedding.weight"].clone()
        for token, new_idx in src_vocab.stoi.items():
            old_idx = old_src_vocab.stoi.get(token)
            if old_idx is not None:
                src_embedding[new_idx] = old_state["src_embedding.weight"][old_idx]
                source_tokens_copied += 1
        new_state["src_embedding.weight"] = src_embedding
        copied_tensors += 1
    else:
        skipped_tensors += 1

    target_tokens_copied = 0
    if "tgt_embedding.weight" in old_state and new_state["tgt_embedding.weight"].shape[1:] == old_state["tgt_embedding.weight"].shape[1:]:
        tgt_embedding = new_state["tgt_embedding.weight"].clone()
        for token, new_idx in tgt_vocab.stoi.items():
            old_idx = old_tgt_vocab.stoi.get(token)
            if old_idx is not None:
                tgt_embedding[new_idx] = old_state["tgt_embedding.weight"][old_idx]
                target_tokens_copied += 1
        new_state["tgt_embedding.weight"] = tgt_embedding
        copied_tensors += 1
    else:
        skipped_tensors += 1

    if "output_projection.weight" in old_state and new_state["output_projection.weight"].shape[1:] == old_state["output_projection.weight"].shape[1:]:
        output_weight = new_state["output_projection.weight"].clone()
        for token, new_idx in tgt_vocab.stoi.items():
            old_idx = old_tgt_vocab.stoi.get(token)
            if old_idx is not None:
                output_weight[new_idx] = old_state["output_projection.weight"][old_idx]
        new_state["output_projection.weight"] = output_weight
        copied_tensors += 1
    else:
        skipped_tensors += 1

    if "output_projection.bias" in old_state:
        output_bias = new_state["output_projection.bias"].clone()
        for token, new_idx in tgt_vocab.stoi.items():
            old_idx = old_tgt_vocab.stoi.get(token)
            if old_idx is not None:
                output_bias[new_idx] = old_state["output_projection.bias"][old_idx]
        new_state["output_projection.bias"] = output_bias
        copied_tensors += 1

    model.load_state_dict(new_state)
    return {
        "init_checkpoint": checkpoint_path,
        "copied_tensors": copied_tensors,
        "skipped_tensors": skipped_tensors,
        "source_tokens_copied": source_tokens_copied,
        "source_vocab_size": len(src_vocab),
        "target_tokens_copied": target_tokens_copied,
        "target_vocab_size": len(tgt_vocab),
    }


def metric_is_better(
    *,
    metric_name: str,
    metric_value: float,
    best_value: float | None,
    min_delta: float = 0.0,
) -> bool:
    """Return whether a metric improved according to its optimization direction."""
    if best_value is None:
        return True
    if metric_name in {"train_loss", "val_loss"}:
        return metric_value < best_value - min_delta
    return metric_value > best_value + min_delta


def main() -> None:
    args = parse_args()
    resolve_mode_defaults(args)
    validate_training_dataset_path(str(args.dataset))
    maybe_apply_tiny_model(args)

    import math

    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from src.data.collate import TranslationCollator
    from src.data.dataset import EnglishASLDataset, load_paired_records
    from src.data.preprocess_dataset import preprocess_records
    from src.data.splits import split_records
    from src.models.english_to_asl_model import EnglishToASLTransformer
    from src.models.tokenizer_utils import SimpleWhitespaceTokenizer, build_vocabs
    from src.training.evaluate import evaluate_model
    from src.training.losses import seq2seq_cross_entropy
    from src.training.metrics import compute_bleu
    from src.utils.seed import set_seed

    set_seed(args.seed)

    records = load_paired_records(args.dataset)
    records = preprocess_records(records)

    if args.no_val:
        train_records = records
        val_records: list[dict[str, str]] = []
        test_records: list[dict[str, str]] = []
    else:
        train_records, val_records, test_records = split_records(
            records,
            val_split=args.val_split,
            test_split=args.test_split,
            seed=args.seed,
        )

    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("max_train_samples must be > 0")
        train_records = train_records[: args.max_train_samples]

    if not train_records:
        raise ValueError("Training set is empty after filtering/truncation.")

    print(
        f"Dataset: {len(train_records)} train, {len(val_records)} val, {len(test_records)} test samples"
    )

    src_tokenizer = SimpleWhitespaceTokenizer(lowercase=True)
    tgt_tokenizer = SimpleWhitespaceTokenizer(lowercase=False)

    # Build vocabs using training split only.
    src_vocab, tgt_vocab = build_vocabs(train_records, src_tokenizer, tgt_tokenizer)
    print(f"Vocabulary: {len(src_vocab)} source tokens, {len(tgt_vocab)} target tokens")

    train_dataset = EnglishASLDataset(
        train_records,
        src_tokenizer,
        tgt_tokenizer,
        src_vocab,
        tgt_vocab,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
    )

    collator = TranslationCollator(pad_idx=tgt_vocab.pad_idx)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    if args.no_val:
        val_loader = None
    else:
        val_dataset = EnglishASLDataset(
            val_records,
            src_tokenizer,
            tgt_tokenizer,
            src_vocab,
            tgt_vocab,
            max_src_len=args.max_src_len,
            max_tgt_len=args.max_tgt_len,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    model = EnglishToASLTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_pad_idx=src_vocab.pad_idx,
        tgt_pad_idx=tgt_vocab.pad_idx,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    warm_start_stats = None
    if args.init_checkpoint:
        warm_start_stats = warm_start_from_checkpoint(
            model=model,
            checkpoint_path=args.init_checkpoint,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=args.device,
        )
        print(f"Warm-start: {json.dumps(warm_start_stats)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    # Cosine annealing with warmup
    def lr_lambda(epoch: int) -> float:
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Label smoothing cross-entropy
    label_smooth_loss = nn.CrossEntropyLoss(
        ignore_index=tgt_vocab.pad_idx,
        label_smoothing=args.label_smoothing,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "best_model.pt"
    best_val_loss_path = save_dir / "best_val_loss.pt"
    best_val_bleu_path = save_dir / "best_val_bleu.pt"
    best_val_token_accuracy_path = save_dir / "best_val_token_accuracy.pt"
    best_train_loss_path = save_dir / "best_train_loss.pt"

    log_path = Path(args.log_path) if args.log_path else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps(
                {
                    "event": "run_start",
                    "dataset": str(args.dataset),
                    "train_size": len(train_records),
                    "val_size": len(val_records),
                    "test_size": len(test_records),
                    "save_dir": str(save_dir),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_paths": {
                        "best_model": str(checkpoint_path),
                        "best_val_loss": str(best_val_loss_path),
                        "best_val_bleu": str(best_val_bleu_path),
                        "best_val_token_accuracy": str(best_val_token_accuracy_path),
                        "best_train_loss": str(best_train_loss_path),
                    },
                    "init_checkpoint": args.init_checkpoint,
                    "warm_start": warm_start_stats,
                    "model_config": {
                        "d_model": args.d_model,
                        "nhead": args.nhead,
                        "num_encoder_layers": args.num_encoder_layers,
                        "num_decoder_layers": args.num_decoder_layers,
                        "dim_feedforward": args.dim_feedforward,
                        "dropout": args.dropout,
                        "tiny_model": args.tiny_model,
                    },
                    "training_config": {
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "grad_clip": args.grad_clip,
                        "label_smoothing": args.label_smoothing,
                        "warmup_epochs": args.warmup_epochs,
                        "bleu_eval_every": args.bleu_eval_every,
                        "bleu_eval_samples": args.bleu_eval_samples,
                        "early_stopping_metric": args.early_stopping_metric,
                        "early_stopping_patience": args.early_stopping_patience,
                        "early_stopping_min_delta": args.early_stopping_min_delta,
                        "val_split": args.val_split,
                        "test_split": args.test_split,
                        "seed": args.seed,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def save_checkpoint(path: Path, metric_name: str, metric_value: float, epoch: int) -> None:
        torch.save(
            {
                "model_name": "english_to_asl_transformer",
                "model_config": {
                    "d_model": args.d_model,
                    "nhead": args.nhead,
                    "num_encoder_layers": args.num_encoder_layers,
                    "num_decoder_layers": args.num_decoder_layers,
                    "dim_feedforward": args.dim_feedforward,
                    "dropout": args.dropout,
                    "src_pad_idx": src_vocab.pad_idx,
                    "tgt_pad_idx": tgt_vocab.pad_idx,
                },
                "src_vocab": src_vocab.to_dict(),
                "tgt_vocab": tgt_vocab.to_dict(),
                "src_tokenizer": {"lowercase": src_tokenizer.lowercase},
                "tgt_tokenizer": {"lowercase": tgt_tokenizer.lowercase},
                "model_state_dict": model.state_dict(),
                "best_metric": metric_value,
                "best_metric_name": metric_name,
                "best_epoch": epoch,
                "dataset_path": str(args.dataset),
                "init_checkpoint": args.init_checkpoint,
                "warm_start": warm_start_stats,
                "split_config": {
                    "seed": args.seed,
                    "val_split": 0.0 if args.no_val else args.val_split,
                    "test_split": 0.0 if args.no_val else args.test_split,
                },
                "train_size": len(train_records),
                "val_size": len(val_records),
                "test_size": len(test_records),
                "no_val": args.no_val,
            },
            path,
        )

    best_metrics: dict[str, float | None] = {
        "train_loss": None,
        "val_loss": None,
        "val_token_accuracy": None,
        "val_bleu": None,
    }
    best_checkpoints: dict[str, str | None] = {
        "best_model": str(checkpoint_path),
        "best_train_loss": None,
        "best_val_loss": None,
        "best_val_bleu": None,
        "best_val_token_accuracy": None,
    }
    early_stopping_best: float | None = None
    early_stopping_bad_checks = 0
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            src_ids = batch["src_ids"].to(args.device)
            tgt_input_ids = batch["tgt_input_ids"].to(args.device)
            tgt_output_ids = batch["tgt_output_ids"].to(args.device)

            optimizer.zero_grad()
            logits = model(src_ids=src_ids, tgt_input_ids=tgt_input_ids)

            # Use label smoothing loss
            vocab_size = logits.size(-1)
            loss = label_smooth_loss(logits.reshape(-1, vocab_size), tgt_output_ids.reshape(-1))
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / max(len(train_loader), 1)

        metrics: dict[str, float | int | str | bool | None] = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "no_val": args.no_val,
            "train_size": len(train_records),
            "test_size": len(test_records),
        }

        if args.no_val:
            monitor_value = train_loss
            metrics["monitor_metric"] = "train_loss"
            metrics["monitor_value"] = round(monitor_value, 4)
        else:
            assert val_loader is not None
            val_metrics = evaluate_model(model, val_loader, device=args.device, pad_idx=tgt_vocab.pad_idx)
            metrics.update({k: round(v, 4) if isinstance(v, float) else v for k, v in val_metrics.items()})
            monitor_value = val_metrics["val_loss"]
            metrics["monitor_metric"] = "val_loss"
            metrics["monitor_value"] = round(monitor_value, 4)

            # Compute BLEU on a validation sample on the requested cadence.
            should_eval_bleu = args.bleu_eval_every > 0 and (
                epoch % args.bleu_eval_every == 0 or epoch == args.epochs
            )
            if should_eval_bleu:
                bleu_refs = []
                bleu_hyps = []
                bleu_records = val_records if args.bleu_eval_samples == 0 else val_records[: args.bleu_eval_samples]
                model.eval()
                with torch.no_grad():
                    for rec in bleu_records:
                        src_tokens = src_tokenizer.tokenize(rec["english"])
                        src_ids_list = [src_vocab.bos_idx] + src_vocab.encode(src_tokens) + [src_vocab.eos_idx]
                        src_tensor = torch.tensor([src_ids_list], dtype=torch.long, device=args.device)

                        gen = model.generate(src_tensor, tgt_vocab.bos_idx, tgt_vocab.eos_idx, max_len=32)
                        gen_ids = gen.squeeze(0).tolist()
                        raw_tokens = tgt_vocab.decode(gen_ids)
                        from src.asl.postprocess_gloss import clean_gloss_tokens
                        pred_tokens = clean_gloss_tokens(raw_tokens)

                        ref_tokens = tgt_tokenizer.tokenize(rec["gloss"])
                        ref_tokens = [t.upper() for t in ref_tokens]

                        bleu_refs.append(ref_tokens)
                        bleu_hyps.append(pred_tokens)

                from src.training.metrics import corpus_bleu
                bleu_result = corpus_bleu(bleu_refs, bleu_hyps)
                metrics["val_bleu"] = round(bleu_result["corpus_bleu"], 4)
                metrics["val_bleu_samples"] = len(bleu_records)

        print(json.dumps(metrics))
        if log_path:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "epoch", **metrics}) + "\n")

        current_metrics = {k: v for k, v in metrics.items() if k in best_metrics and isinstance(v, (float, int))}
        for metric_name, raw_metric_value in current_metrics.items():
            metric_value = float(raw_metric_value)
            if not metric_is_better(
                metric_name=metric_name,
                metric_value=metric_value,
                best_value=best_metrics[metric_name],
            ):
                continue

            best_metrics[metric_name] = metric_value
            if metric_name == "train_loss" and args.no_val:
                save_checkpoint(best_train_loss_path, metric_name, metric_value, epoch)
                save_checkpoint(checkpoint_path, metric_name, metric_value, epoch)
                best_checkpoints["best_train_loss"] = str(best_train_loss_path)
            elif metric_name == "val_loss":
                save_checkpoint(best_val_loss_path, metric_name, metric_value, epoch)
                save_checkpoint(checkpoint_path, metric_name, metric_value, epoch)
                best_checkpoints["best_val_loss"] = str(best_val_loss_path)
            elif metric_name == "val_bleu":
                save_checkpoint(best_val_bleu_path, metric_name, metric_value, epoch)
                best_checkpoints["best_val_bleu"] = str(best_val_bleu_path)
            elif metric_name == "val_token_accuracy":
                save_checkpoint(best_val_token_accuracy_path, metric_name, metric_value, epoch)
                best_checkpoints["best_val_token_accuracy"] = str(best_val_token_accuracy_path)

        if args.early_stopping_metric:
            early_metric_value = current_metrics.get(args.early_stopping_metric)
            if early_metric_value is not None:
                early_metric_value = float(early_metric_value)
                if metric_is_better(
                    metric_name=args.early_stopping_metric,
                    metric_value=early_metric_value,
                    best_value=early_stopping_best,
                    min_delta=args.early_stopping_min_delta,
                ):
                    early_stopping_best = early_metric_value
                    early_stopping_bad_checks = 0
                else:
                    early_stopping_bad_checks += 1
                    if (
                        args.early_stopping_patience > 0
                        and early_stopping_bad_checks >= args.early_stopping_patience
                    ):
                        stopped_early = True
                        print(
                            json.dumps(
                                {
                                    "event": "early_stopping",
                                    "epoch": epoch,
                                    "metric": args.early_stopping_metric,
                                    "best_value": round(early_stopping_best, 4)
                                    if early_stopping_best is not None
                                    else None,
                                    "bad_checks": early_stopping_bad_checks,
                                }
                            )
                        )
                        break

    if log_path:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "run_complete",
                        "best_checkpoint": str(checkpoint_path),
                        "best_checkpoints": best_checkpoints,
                        "best_metrics": best_metrics,
                        "stopped_early": stopped_early,
                    }
                )
                + "\n"
            )

    print(f"\nTraining complete. Loss-compatible checkpoint: {checkpoint_path}")
    print(f"Best checkpoints: {json.dumps(best_checkpoints)}")


if __name__ == "__main__":
    main()
