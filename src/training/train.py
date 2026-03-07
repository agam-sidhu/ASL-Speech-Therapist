"""Training entry point for English->ASL gloss seq2seq model."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import DEFAULT_CHECKPOINT_DIR, DEFAULT_TOY_DATASET_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train English->ASL gloss transformer.")
    parser.add_argument("--dataset", default=str(DEFAULT_TOY_DATASET_PATH), help="Dataset file or directory")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience on val loss")
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

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


def split_records(records: list[dict[str, str]], val_split: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Randomly split records into train and validation sets."""
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1")

    shuffled = records[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - val_split))
    split_idx = max(1, min(split_idx, len(shuffled) - 1))

    train_records = shuffled[:split_idx]
    val_records = shuffled[split_idx:]
    return train_records, val_records


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader

    from src.data.collate import TranslationCollator
    from src.data.dataset import EnglishASLDataset, load_dataset_splits
    from src.data.preprocess_dataset import preprocess_records
    from src.models.english_to_asl_model import EnglishToASLTransformer
    from src.models.tokenizer_utils import SimpleWhitespaceTokenizer, build_vocabs
    from src.training.evaluate import evaluate_model
    from src.training.losses import seq2seq_cross_entropy
    from src.utils.seed import set_seed

    set_seed(args.seed)

    dataset_path = Path(args.dataset)
    loaded_train, loaded_val = load_dataset_splits(dataset_path)

    train_records = preprocess_records(loaded_train)
    val_records = preprocess_records(loaded_val) if loaded_val else None

    if args.no_val:
        val_records = []
    elif val_records is None:
        train_records, val_records = split_records(train_records, val_split=args.val_split, seed=args.seed)

    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("max_train_samples must be > 0")
        train_records = train_records[: args.max_train_samples]

    if not train_records:
        raise ValueError("Training set is empty after filtering/truncation.")

    src_tokenizer = SimpleWhitespaceTokenizer(lowercase=True)
    tgt_tokenizer = SimpleWhitespaceTokenizer(lowercase=False)

    src_vocab, tgt_vocab = build_vocabs(train_records, src_tokenizer, tgt_tokenizer)

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
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    if args.no_val or not val_records:
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
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
        )

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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "best_model.pt"

    best_metric = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            src_ids = batch["src_ids"].to(args.device)
            tgt_input_ids = batch["tgt_input_ids"].to(args.device)
            tgt_output_ids = batch["tgt_output_ids"].to(args.device)

            optimizer.zero_grad()
            logits = model(src_ids=src_ids, tgt_input_ids=tgt_input_ids)
            loss = seq2seq_cross_entropy(logits, tgt_output_ids, pad_idx=tgt_vocab.pad_idx)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(len(train_loader), 1)

        metrics: dict[str, float | int | str | bool | None] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_size": len(train_records),
            "val_size": len(val_records or []),
            "no_val": args.no_val,
        }

        if val_loader is None:
            monitor_value = train_loss
            monitor_name = "train_loss"
        else:
            val_metrics = evaluate_model(model, val_loader, device=args.device, pad_idx=tgt_vocab.pad_idx)
            metrics.update(val_metrics)
            monitor_value = float(val_metrics["val_loss"])
            monitor_name = "val_loss"

        metrics["monitor_metric"] = monitor_name
        metrics["monitor_value"] = monitor_value

        print(json.dumps(metrics))

        improved = monitor_value < best_metric
        if improved:
            best_metric = monitor_value
            epochs_without_improvement = 0
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
                    "best_metric": best_metric,
                    "best_metric_name": monitor_name,
                    "train_size": len(train_records),
                    "val_size": len(val_records or []),
                    "no_val": args.no_val,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if (not args.no_val) and args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"Early stopping: no {monitor_name} improvement for {epochs_without_improvement} epochs "
                f"(patience={args.patience})."
            )
            break

    print(f"Training complete. Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
