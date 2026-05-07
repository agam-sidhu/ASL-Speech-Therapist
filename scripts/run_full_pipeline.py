"""Run the reproducible ASL gloss data/training pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "gloss_pipeline.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "false"}:
        return value == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_simple_yaml(path: Path) -> dict[str, dict[str, Any]]:
    """Load the small two-level YAML config without requiring PyYAML."""
    config: dict[str, dict[str, Any]] = {}
    current_section: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" "):
            if not line.endswith(":"):
                raise ValueError(f"Unsupported config line: {raw_line}")
            current_section = line[:-1].strip()
            config[current_section] = {}
            continue
        if current_section is None:
            raise ValueError(f"Config key outside section: {raw_line}")
        key, sep, value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"Unsupported config line: {raw_line}")
        config[current_section][key.strip()] = parse_scalar(value)
    return config


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_checkpoint(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")


def run_command(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def build_data(config: dict[str, dict[str, Any]]) -> None:
    from src.data.build_active_gloss_pipeline import build_active_datasets

    paths = config["paths"]
    report = build_active_datasets(
        raw_aslg_csv=project_path(str(paths["raw_aslg_csv"])),
        raw_conversational_path=project_path(str(paths["raw_project_conversational"])),
        raw_v2_path=project_path(str(paths["raw_project_v2"])),
        raw_v4_path=project_path(str(paths["raw_project_v4"])),
        aslg_output_path=project_path(str(paths["aslg_pretrain_dataset"])),
        project_output_path=project_path(str(paths["project_finetune_dataset"])),
        report_output_path=project_path(str(paths["data_report"])),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def training_args(config: dict[str, dict[str, Any]], section_name: str) -> list[str]:
    training = dict(config["training"])
    training.update(config.get(section_name, {}))
    ordered_keys = [
        "epochs",
        "batch_size",
        "lr",
        "d_model",
        "nhead",
        "num_encoder_layers",
        "num_decoder_layers",
        "dim_feedforward",
        "dropout",
        "grad_clip",
        "label_smoothing",
        "warmup_epochs",
        "val_split",
        "test_split",
        "seed",
        "device",
        "save_dir",
        "log_path",
        "init_checkpoint",
    ]
    args: list[str] = []
    for key in ordered_keys:
        value = training.get(key)
        if value is None:
            continue
        args.extend([f"--{key}", str(value)])
    return args


def pretrain(config: dict[str, dict[str, Any]]) -> None:
    paths = config["paths"]
    command = [
        sys.executable,
        "src/training/train.py",
        "--mode",
        "pretrain",
        "--dataset",
        str(paths["aslg_pretrain_dataset"]),
    ]
    command.extend(training_args(config, "pretrain"))
    run_command(command)


def finetune(config: dict[str, dict[str, Any]]) -> None:
    paths = config["paths"]
    require_checkpoint(project_path(str(paths["pretrain_checkpoint"])))
    command = [
        sys.executable,
        "src/training/train.py",
        "--mode",
        "finetune",
        "--dataset",
        str(paths["project_finetune_dataset"]),
    ]
    command.extend(training_args(config, "finetune"))
    run_command(command)


def evaluate(config: dict[str, dict[str, Any]]) -> None:
    paths = config["paths"]
    eval_config = config["evaluation"]
    checkpoints = [
        paths["pretrain_checkpoint"],
        paths["finetune_checkpoint"],
    ]
    for checkpoint in checkpoints:
        checkpoint_path = project_path(str(checkpoint))
        require_checkpoint(checkpoint_path)
        run_command(
            [
                sys.executable,
                "test_batch.py",
                "--checkpoint",
                str(checkpoint),
                "--device",
                str(eval_config["device"]),
                "--beam_width",
                str(eval_config["beam_width"]),
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ASL gloss data/training pipeline stages.")
    parser.add_argument(
        "--stage",
        choices=["build-data", "pretrain", "finetune", "evaluate", "all"],
        required=True,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_simple_yaml(project_path(args.config))

    if args.stage in {"build-data", "all"}:
        build_data(config)
    if args.stage in {"pretrain", "all"}:
        pretrain(config)
    if args.stage in {"finetune", "all"}:
        finetune(config)
    if args.stage in {"evaluate", "all"}:
        evaluate(config)


if __name__ == "__main__":
    main()
