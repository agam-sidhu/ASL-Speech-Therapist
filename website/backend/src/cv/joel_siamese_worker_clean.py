# from __future__ import annotations
# import sys
# import types
# import tensorflow as tf

# # Create fake keras.src.models.functional module
# keras_module = sys.modules.setdefault("keras", types.ModuleType("keras"))
# src_module = sys.modules.setdefault("keras.src", types.ModuleType("keras.src"))
# models_module = sys.modules.setdefault("keras.src.models", types.ModuleType("keras.src.models"))
# functional_module = sys.modules.setdefault("keras.src.models.functional", types.ModuleType("keras.src.models.functional"))

# setattr(keras_module, "src", src_module)
# setattr(src_module, "models", models_module)
# setattr(models_module, "functional", functional_module)

# # Map Functional → tf.keras.Model
# setattr(functional_module, "Functional", tf.keras.Model)
# setattr(functional_module, "Model", tf.keras.Model)
# print(tf.config.list_physical_devices('GPU'))


# import argparse
# import json
# import sys
# import tempfile
# import zipfile
# from pathlib import Path

# import keras
# import numpy as np

# # 🔥 PATCH LayerNormalization deserialization (CRITICAL FIX)
# import tensorflow as tf

# original_layernorm_from_config = tf.keras.layers.LayerNormalization.from_config

# @classmethod
# def fixed_layernorm_from_config(cls, config):
#     config = dict(config)
#     config.pop("rms_scaling", None)  # 🔥 REMOVE unsupported arg
#     return original_layernorm_from_config(config)

# tf.keras.layers.LayerNormalization.from_config = fixed_layernorm_from_config
# try:
#     from keras.src import dtype_policies as _dtype_policies_impl
# except ImportError:
#     import keras.mixed_precision as _dtype_policies_impl

# try:
#     if not hasattr(keras.dtype_policies, "Policy"):
#         keras.dtype_policies.Policy = keras.dtype_policies.DTypePolicy
# except:
#     pass
# if not hasattr(keras.mixed_precision, "Policy"):
#     keras.mixed_precision.Policy = keras.dtype_policies.DTypePolicy
# if not hasattr(_dtype_policies_impl, "Policy"):
#     _dtype_policies_impl.Policy = _dtype_policies_impl.DTypePolicy

# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

# from src.cv.joel_custom_layers import (  # noqa: E402
#     AbsDiffLayer,
#     LateDropout,
#     MambaStyleBlock,
#     PositionalEmbedding,
#     SkeletonAugmentation,
#     SpatialGraphConv,
#     TransformerBlock,
# )


# def _load_label_encoder(candidate: Path):
#     try:
#         import joblib

#         return joblib.load(str(candidate))
#     except Exception:
#         import pickle

#         with candidate.open("rb") as f:
#             return pickle.load(f)


# class JsonLabelEncoder:
#     def __init__(self, classes: list[str]) -> None:
#         if not classes:
#             raise ValueError("Label classes are empty")
#         self.classes_ = np.array(classes, dtype=object)

#     def inverse_transform(self, indices):
#         out = []
#         for idx in indices:
#             i = int(idx)
#             if i < 0 or i >= len(self.classes_):
#                 raise ValueError(f"Label index out of range: {i}")
#             out.append(self.classes_[i])
#         return np.array(out, dtype=object)


# def _read_encoder(path: Path):
#     if path.suffix.lower() == ".json":
#         payload = json.loads(path.read_text(encoding="utf-8"))
#         if isinstance(payload, dict):
#             classes = payload.get("classes", [])
#         else:
#             classes = payload
#         return JsonLabelEncoder([str(v) for v in classes])
#     return _load_label_encoder(path)


# def _normalize_keras_config(value):
#     if isinstance(value, dict):
#         class_name = value.get("class_name")
#         module_name = value.get("module")
#         if class_name == "Policy" and module_name == "keras":
#             config = value.get("config", {})
#             return config.get("name", "float32")
#         return {key: _normalize_keras_config(item) for key, item in value.items()}
#     if isinstance(value, list):
#         return [_normalize_keras_config(item) for item in value]
#     return value


# def _load_model_with_normalized_dtype_policy(model_path: Path, custom_objects: dict[str, object]):
#     with zipfile.ZipFile(model_path, "r") as source_zip:
#         config_payload = json.loads(source_zip.read("config.json").decode("utf-8"))
#         config_payload = _normalize_keras_config(config_payload)

#         with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as temp_file:
#             temp_path = Path(temp_file.name)

#         try:
#             with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as target_zip:
#                 for member in source_zip.infolist():
#                     if member.filename == "config.json":
#                         target_zip.writestr("config.json", json.dumps(config_payload, ensure_ascii=False))
#                     else:
#                         target_zip.writestr(member, source_zip.read(member.filename))
#             return keras.models.load_model(
#                 str(temp_path),
#                 custom_objects=custom_objects,
#                 compile=False,
#                 safe_mode=False,
#             )
#         finally:
#             try:
#                 temp_path.unlink(missing_ok=True)
#             except Exception:
#                 pass


# def _preprocess_siamese_sequence(seq_3d: np.ndarray, sequence_len: int = 100) -> np.ndarray:
#     if seq_3d.size == 0:
#         return np.zeros((sequence_len, 75, 9), dtype=np.float32)

#     center = seq_3d[:, 11:12, :]
#     seq_3d = seq_3d - center
#     shoulder_dist = np.linalg.norm(seq_3d[:, 11] - seq_3d[:, 12], axis=-1)
#     scale = float(np.mean(shoulder_dist) + 1e-6)
#     seq_3d = seq_3d / scale

#     vel = np.zeros_like(seq_3d)
#     vel[1:] = seq_3d[1:] - seq_3d[:-1]

#     acc = np.zeros_like(seq_3d)
#     acc[1:] = vel[1:] - vel[:-1]

#     kinematic = np.concatenate([seq_3d, vel, acc], axis=-1)
#     if kinematic.shape[0] >= sequence_len:
#         return kinematic[:sequence_len].astype(np.float32)

#     pad_width = sequence_len - kinematic.shape[0]
#     return np.pad(
#         kinematic,
#         ((0, pad_width), (0, 0), (0, 0)),
#         mode="constant",
#         constant_values=0.0,
#     ).astype(np.float32)


# def _to_model_sequence(sample: np.ndarray, sequence_len: int = 100) -> np.ndarray:
#     arr = np.asarray(sample)

#     if arr.ndim == 3 and arr.shape[1] == 75 and arr.shape[2] == 9:
#         arr = arr.astype(np.float32)
#         if arr.shape[0] >= sequence_len:
#             return arr[:sequence_len]
#         pad = np.zeros((sequence_len - arr.shape[0], 75, 9), dtype=np.float32)
#         return np.concatenate([arr, pad], axis=0)

#     if arr.ndim == 2 and arr.shape[1] == 225:
#         arr = arr.reshape(arr.shape[0], 75, 3).astype(np.float32)
#         return _preprocess_siamese_sequence(arr, sequence_len=sequence_len)

#     if arr.ndim == 3 and arr.shape[1] == 75 and arr.shape[2] == 3:
#         return _preprocess_siamese_sequence(arr.astype(np.float32), sequence_len=sequence_len)

#     # Handle edge case: if shape doesn't match, try reshaping to (T, 225) format
#     if arr.ndim == 2 and arr.shape[1] % 225 == 0:
#         # Multiple samples stacked, take first one
#         arr = arr[0] if arr.shape[0] > 225 else arr
    
#     if arr.ndim == 1 and len(arr) == 225:
#         # Single flattened frame - reshape to (1, 225)
#         arr = arr.reshape(1, 225)
    
#     if arr.ndim == 2 and arr.shape[1] == 225:
#         arr = arr.reshape(arr.shape[0], 75, 3).astype(np.float32)
#         return _preprocess_siamese_sequence(arr, sequence_len=sequence_len)

#     raise ValueError(f"Unsupported sample shape for Joel Siamese worker: {arr.shape}")


# def build_custom_objects() -> dict[str, object]:
#     return {
#         "SkeletonAugmentation": SkeletonAugmentation,
#         "Custom>SkeletonAugmentation": SkeletonAugmentation,
#         "PositionalEmbedding": PositionalEmbedding,
#         "Custom>PositionalEmbedding": PositionalEmbedding,
#         "TransformerBlock": TransformerBlock,
#         "Custom>TransformerBlock": TransformerBlock,
#         "LateDropout": LateDropout,
#         "Custom>LateDropout": LateDropout,
#         "SpatialGraphConv": SpatialGraphConv,
#         "Custom>SpatialGraphConv": SpatialGraphConv,
#         "MambaStyleBlock": MambaStyleBlock,
#         "Custom>MambaStyleBlock": MambaStyleBlock,
#         "AbsDiffLayer": AbsDiffLayer,
#         "Custom>AbsDiffLayer": AbsDiffLayer,
#     }


# def run_worker(
#     model_path: Path,
#     query_path: Path,
#     reference_x_path: Path,
#     reference_y_path: Path,
#     encoder_path: Path,
#     top_n: int = 5,
# ) -> dict[str, object]:
#     sequence_len = 100
#     model = _load_model_with_normalized_dtype_policy(model_path, build_custom_objects())

#     query = _to_model_sequence(np.load(str(query_path), allow_pickle=True), sequence_len=sequence_len)
#     try:
#         reference_samples = np.load(str(reference_x_path), mmap_mode="r")
#     except ValueError:
#         reference_samples = np.load(str(reference_x_path), allow_pickle=True)
#     reference_labels_raw = np.load(str(reference_y_path), allow_pickle=True)

#     if getattr(reference_labels_raw, "dtype", None) is not None and reference_labels_raw.dtype.kind in {"i", "u"}:
#         label_encoder = _read_encoder(encoder_path)
#         reference_labels = np.array(label_encoder.inverse_transform(reference_labels_raw.astype(int)), dtype=object)
#     else:
#         reference_labels = reference_labels_raw.astype(object)

#     # unique_labels, first_idx = np.unique(reference_labels, return_index=True)
#     # reference_labels = unique_labels
#     unique_labels, first_idx = np.unique(reference_labels, return_index=True)

#     # 🔥 LIMIT SEARCH SIZE (CRITICAL FIX)
#     MAX_SAMPLES = 200
#     unique_labels = unique_labels[:MAX_SAMPLES]
#     first_idx = first_idx[:MAX_SAMPLES]

#     reference_labels = unique_labels
    
#     # Normalize selected samples, skipping any that fail
#     selected_samples = []
#     valid_labels = []
#     for label, idx in zip(unique_labels, first_idx):
#         try:
#             normalized = _to_model_sequence(reference_samples[int(idx)], sequence_len=sequence_len)
#             # Verify the output shape is exactly (100, 75, 9)
#             if normalized.shape != (sequence_len, 75, 9):
#                 continue
#             selected_samples.append(normalized)
#             valid_labels.append(label)
#         except Exception:
#             continue
    
#     if not selected_samples:
#         raise RuntimeError("No valid reference samples could be normalized for Siamese comparison")
    
#     reference_samples = np.stack(selected_samples, axis=0)
#     reference_labels = np.array(valid_labels, dtype=object)

#     query_batch = np.repeat(query[np.newaxis, ...], len(reference_labels), axis=0)
#     scores = model.predict([query_batch, reference_samples], batch_size=64, verbose=0).reshape(-1)

#     order = np.argsort(scores)[::-1]
#     top_n = min(top_n, len(order))
#     predicted_signs = [str(reference_labels[i]) for i in order[:top_n]]
#     confidences = [float(scores[i]) for i in order[:top_n]]

#     return {
#         "predicted_signs": predicted_signs,
#         "confidences": confidences,
#         "model_type": "siamese_topk_similarity",
#     }


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Run Joel Siamese inference in a clean Keras 3 runtime.")
#     parser.add_argument("--model", required=True)
#     parser.add_argument("--query", required=True)
#     parser.add_argument("--reference-x", required=True)
#     parser.add_argument("--reference-y", required=True)
#     parser.add_argument("--encoder", required=True)
#     parser.add_argument("--top-n", type=int, default=5)
#     return parser.parse_args()


# def main() -> int:
#     args = parse_args()
#     result = run_worker(
#         model_path=Path(args.model),
#         query_path=Path(args.query),
#         reference_x_path=Path(args.reference_x),
#         reference_y_path=Path(args.reference_y),
#         encoder_path=Path(args.encoder),
#         top_n=args.top_n,
#     )
#     print(json.dumps(result, ensure_ascii=False))
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())





# from __future__ import annotations
# import sys
# import types
# import tensorflow as tf
# import time

# # Create fake keras.src.models.functional module
# keras_module = sys.modules.setdefault("keras", types.ModuleType("keras"))
# src_module = sys.modules.setdefault("keras.src", types.ModuleType("keras.src"))
# models_module = sys.modules.setdefault("keras.src.models", types.ModuleType("keras.src.models"))
# functional_module = sys.modules.setdefault("keras.src.models.functional", types.ModuleType("keras.src.models.functional"))

# setattr(keras_module, "src", src_module)
# setattr(src_module, "models", models_module)
# setattr(models_module, "functional", functional_module)

# # Map Functional → tf.keras.Model
# setattr(functional_module, "Functional", tf.keras.Model)
# setattr(functional_module, "Model", tf.keras.Model)

# # 🔥 GPU CHECK
# print("GPUs:", tf.config.list_physical_devices('GPU'))

# import argparse
# import json
# import tempfile
# import zipfile
# from pathlib import Path

# import keras
# import numpy as np

# # 🔥 PATCH LayerNormalization
# original_layernorm_from_config = tf.keras.layers.LayerNormalization.from_config

# @classmethod
# def fixed_layernorm_from_config(cls, config):
#     config = dict(config)
#     config.pop("rms_scaling", None)
#     return original_layernorm_from_config(config)

# tf.keras.layers.LayerNormalization.from_config = fixed_layernorm_from_config

# try:
#     from keras.src import dtype_policies as _dtype_policies_impl
# except ImportError:
#     import keras.mixed_precision as _dtype_policies_impl

# try:
#     if not hasattr(keras.dtype_policies, "Policy"):
#         keras.dtype_policies.Policy = keras.dtype_policies.DTypePolicy
# except:
#     pass

# if not hasattr(keras.mixed_precision, "Policy"):
#     keras.mixed_precision.Policy = keras.dtype_policies.DTypePolicy

# if not hasattr(_dtype_policies_impl, "Policy"):
#     _dtype_policies_impl.Policy = _dtype_policies_impl.DTypePolicy

# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

# from src.cv.joel_custom_layers import (
#     AbsDiffLayer,
#     LateDropout,
#     MambaStyleBlock,
#     PositionalEmbedding,
#     SkeletonAugmentation,
#     SpatialGraphConv,
#     TransformerBlock,
# )

# def _load_label_encoder(candidate: Path):
#     try:
#         import joblib
#         return joblib.load(str(candidate))
#     except Exception:
#         import pickle
#         with candidate.open("rb") as f:
#             return pickle.load(f)

# class JsonLabelEncoder:
#     def __init__(self, classes: list[str]) -> None:
#         self.classes_ = np.array(classes, dtype=object)

#     def inverse_transform(self, indices):
#         return np.array([self.classes_[int(i)] for i in indices], dtype=object)

# def _read_encoder(path: Path):
#     if path.suffix.lower() == ".json":
#         payload = json.loads(path.read_text(encoding="utf-8"))
#         classes = payload.get("classes", []) if isinstance(payload, dict) else payload
#         return JsonLabelEncoder([str(v) for v in classes])
#     return _load_label_encoder(path)

# def _normalize_keras_config(value):
#     if isinstance(value, dict):
#         if value.get("class_name") == "Policy" and value.get("module") == "keras":
#             return value.get("config", {}).get("name", "float32")
#         return {k: _normalize_keras_config(v) for k, v in value.items()}
#     if isinstance(value, list):
#         return [_normalize_keras_config(v) for v in value]
#     return value

# def _load_model_with_normalized_dtype_policy(model_path: Path, custom_objects: dict):
#     with zipfile.ZipFile(model_path, "r") as source_zip:
#         config_payload = json.loads(source_zip.read("config.json").decode("utf-8"))
#         config_payload = _normalize_keras_config(config_payload)

#         with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as temp_file:
#             temp_path = Path(temp_file.name)

#         try:
#             with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as target_zip:
#                 for member in source_zip.infolist():
#                     if member.filename == "config.json":
#                         target_zip.writestr("config.json", json.dumps(config_payload))
#                     else:
#                         target_zip.writestr(member, source_zip.read(member.filename))

#             return keras.models.load_model(
#                 str(temp_path),
#                 custom_objects=custom_objects,
#                 compile=False,
#                 safe_mode=False,
#             )
#         finally:
#             temp_path.unlink(missing_ok=True)

# def _preprocess_siamese_sequence(seq_3d: np.ndarray, sequence_len=100):
#     seq_3d = seq_3d.astype(np.float32)

#     center = seq_3d[:, 11:12, :]
#     seq_3d -= center

#     scale = np.mean(np.linalg.norm(seq_3d[:, 11] - seq_3d[:, 12], axis=-1)) + 1e-6
#     seq_3d /= scale

#     vel = np.zeros_like(seq_3d)
#     vel[1:] = seq_3d[1:] - seq_3d[:-1]

#     acc = np.zeros_like(seq_3d)
#     acc[1:] = vel[1:] - vel[:-1]

#     kinematic = np.concatenate([seq_3d, vel, acc], axis=-1)

#     if kinematic.shape[0] >= sequence_len:
#         return kinematic[:sequence_len]

#     pad = np.zeros((sequence_len - kinematic.shape[0], 75, 9), dtype=np.float32)
#     return np.concatenate([kinematic, pad], axis=0)

# def _to_model_sequence(sample, sequence_len=100):
#     arr = np.asarray(sample)

#     if arr.ndim == 3 and arr.shape[2] == 9:
#         arr = arr.astype(np.float32)
#         return arr[:sequence_len] if arr.shape[0] >= sequence_len else np.pad(arr, ((0, sequence_len-arr.shape[0]), (0,0),(0,0)))

#     if arr.ndim == 3 and arr.shape[2] == 3:
#         return _preprocess_siamese_sequence(arr, sequence_len)

#     if arr.ndim == 2 and arr.shape[1] == 225:
#         arr = arr.reshape(arr.shape[0], 75, 3)
#         return _preprocess_siamese_sequence(arr, sequence_len)

#     raise ValueError(f"Unsupported shape: {arr.shape}")

# def build_custom_objects():
#     return {
#         "SkeletonAugmentation": SkeletonAugmentation,
#         "PositionalEmbedding": PositionalEmbedding,
#         "TransformerBlock": TransformerBlock,
#         "LateDropout": LateDropout,
#         "SpatialGraphConv": SpatialGraphConv,
#         "MambaStyleBlock": MambaStyleBlock,
#         "AbsDiffLayer": AbsDiffLayer,
#     }

# def run_worker(model_path, query_path, reference_x_path, reference_y_path, encoder_path, top_n=5):
#     t0 = time.time()

#     model = _load_model_with_normalized_dtype_policy(model_path, build_custom_objects())

#     query = _to_model_sequence(np.load(str(query_path), allow_pickle=True))

#     reference_samples = np.load(str(reference_x_path), mmap_mode="r")
#     reference_labels_raw = np.load(str(reference_y_path), allow_pickle=True)

#     if reference_labels_raw.dtype.kind in {"i", "u"}:
#         label_encoder = _read_encoder(encoder_path)
#         reference_labels = label_encoder.inverse_transform(reference_labels_raw.astype(int))
#     else:
#         reference_labels = reference_labels_raw.astype(object)

#     unique_labels, first_idx = np.unique(reference_labels, return_index=True)

#     # 🔥 REDUCED FOR SPEED
#     MAX_SAMPLES = 75
#     unique_labels = unique_labels[:MAX_SAMPLES]
#     first_idx = first_idx[:MAX_SAMPLES]

#     selected = []
#     labels = []

#     for label, idx in zip(unique_labels, first_idx):
#         try:
#             seq = _to_model_sequence(reference_samples[int(idx)])
#             if seq.shape == (100, 75, 9):
#                 selected.append(seq)
#                 labels.append(label)
#         except:
#             continue

#     reference_samples = np.stack(selected).astype(np.float32)
#     reference_labels = np.array(labels)

#     print("Preprocessing time:", time.time() - t0)

#     # 🔥 Prediction timing
#     t1 = time.time()

#     # 🔥 BUILD EMBEDDING MODEL (one branch of siamese)
#     embedding_model = tf.keras.Model(
#         inputs=model.input[0],
#         outputs=model.layers[-2].output  # adjust if needed
#     )

#     # Compute embeddings once
#     query_emb = embedding_model.predict(query[np.newaxis, ...], verbose=0)
#     ref_embs = embedding_model.predict(reference_samples, batch_size=32, verbose=0)

#     # 🔥 Cosine similarity
#     query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
#     ref_embs = ref_embs / np.linalg.norm(ref_embs, axis=1, keepdims=True)

#     scores = np.dot(ref_embs, query_emb.T).reshape(-1)

#     print("Prediction time:", time.time() - t1)

#     order = np.argsort(scores)[::-1][:top_n]

#     return {
#         "predicted_signs": [str(reference_labels[i]) for i in order],
#         "confidences": [float(scores[i]) for i in order],
#         "model_type": "siamese_topk_similarity",
#     }

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", required=True)
#     parser.add_argument("--query", required=True)
#     parser.add_argument("--reference-x", required=True)
#     parser.add_argument("--reference-y", required=True)
#     parser.add_argument("--encoder", required=True)
#     parser.add_argument("--top-n", type=int, default=5)
#     args = parser.parse_args()

#     result = run_worker(
#         Path(args.model),
#         Path(args.query),
#         Path(args.reference_x),
#         Path(args.reference_y),
#         Path(args.encoder),
#         args.top_n,
#     )

#     print(json.dumps(result))

# if __name__ == "__main__":
#     main()

from __future__ import annotations
import sys
import types
from matplotlib.pylab import norm
import tensorflow as tf
import time
import json
import numpy as np
import argparse
import zipfile
import tempfile
from pathlib import Path

import sys
from pathlib import Path

# 🔥 FIX PATH FOR SUBPROCESS
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = CURRENT_DIR.parent.parent  # goes to /backend

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
# ------------------------------
# Fix keras compatibility
# ------------------------------
keras_module = sys.modules.setdefault("keras", types.ModuleType("keras"))
src_module = sys.modules.setdefault("keras.src", types.ModuleType("keras.src"))
models_module = sys.modules.setdefault("keras.src.models", types.ModuleType("keras.src.models"))
functional_module = sys.modules.setdefault("keras.src.models.functional", types.ModuleType("keras.src.models.functional"))

setattr(keras_module, "src", src_module)
setattr(src_module, "models", models_module)
setattr(models_module, "functional", functional_module)

setattr(functional_module, "Functional", tf.keras.Model)
setattr(functional_module, "Model", tf.keras.Model)

print("GPUs:", tf.config.list_physical_devices('GPU'), file=sys.stderr)

import keras
import tensorflow as tf

original_layernorm_from_config = tf.keras.layers.LayerNormalization.from_config

@classmethod
def fixed_layernorm_from_config(cls, config):
    config = dict(config)
    config.pop("rms_scaling", None)
    return original_layernorm_from_config(config)

tf.keras.layers.LayerNormalization.from_config = fixed_layernorm_from_config
from src.cv.joel_custom_layers import (
    SpatialGraphConv,
    MambaStyleBlock,
    TransformerBlock,
    SkeletonAugmentation,
    PositionalEmbedding,
    LateDropout,
    AbsDiffLayer
)

# ------------------------------
# Load model
# ------------------------------
def _normalize_keras_config(value):
    if isinstance(value, dict):
        if value.get("class_name") == "Policy" and value.get("module") == "keras":
            return value.get("config", {}).get("name", "float32")
        return {k: _normalize_keras_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_keras_config(v) for v in value]
    return value


def load_model_fast(model_path: Path):
    with zipfile.ZipFile(model_path, "r") as source_zip:
        config_payload = json.loads(source_zip.read("config.json").decode("utf-8"))
        config_payload = _normalize_keras_config(config_payload)

        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as target_zip:
            for member in source_zip.infolist():
                if member.filename == "config.json":
                    target_zip.writestr("config.json", json.dumps(config_payload))
                else:
                    target_zip.writestr(member, source_zip.read(member.filename))

        model = keras.models.load_model(
            str(temp_path),
            compile=False,
            safe_mode=False,
            custom_objects={
                "SpatialGraphConv": SpatialGraphConv,
                "MambaStyleBlock": MambaStyleBlock,
                "TransformerBlock": TransformerBlock,
                "SkeletonAugmentation": SkeletonAugmentation,
                "PositionalEmbedding": PositionalEmbedding,
                "LateDropout": LateDropout,
                "AbsDiffLayer": AbsDiffLayer,
            }
        )
        temp_path.unlink(missing_ok=True)

    return model


# ------------------------------
# Preprocess
# ------------------------------
def _preprocess_siamese_sequence(seq_3d: np.ndarray, sequence_len=100):
    seq_3d = seq_3d.astype(np.float32)

    center = seq_3d[:, 11:12, :]
    seq_3d -= center

    scale = np.mean(np.linalg.norm(seq_3d[:, 11] - seq_3d[:, 12], axis=-1)) + 1e-6
    seq_3d /= scale

    vel = np.zeros_like(seq_3d)
    vel[1:] = seq_3d[1:] - seq_3d[:-1]

    acc = np.zeros_like(seq_3d)
    acc[1:] = vel[1:] - vel[:-1]

    kinematic = np.concatenate([seq_3d, vel, acc], axis=-1)

    if kinematic.shape[0] >= sequence_len:
        return kinematic[:sequence_len]

    pad = np.zeros((sequence_len - kinematic.shape[0], 75, 9), dtype=np.float32)
    return np.concatenate([kinematic, pad], axis=0)


def _to_model_sequence(sample, sequence_len=100):
    arr = np.asarray(sample)

    if arr.ndim == 3 and arr.shape[2] == 9:
        return arr.astype(np.float32)

    if arr.ndim == 3 and arr.shape[2] == 3:
        return _preprocess_siamese_sequence(arr, sequence_len)

    raise ValueError(f"Unsupported shape: {arr.shape}")


# ------------------------------
# Load reference library
# ------------------------------
def load_reference_library(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)

    labels = list(data.keys())
    embeddings = np.array(list(data.values()), dtype=np.float32)

    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    return labels, embeddings


# ------------------------------
# Worker
# ------------------------------
def run_worker(model, labels, ref_embeddings, query_path: Path, top_n=5):

    t0 = time.time()

    query = _to_model_sequence(np.load(str(query_path), allow_pickle=True))

    # Try to find embedding layer safely
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 2:
            embedding_layer = layer
            break

    embedding_model = tf.keras.Model(
        inputs=model.input[0],
        outputs=embedding_layer.output
    )

    query_emb = embedding_model.predict(query[np.newaxis, ...], verbose=0)
    norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
    query_emb = query_emb / (norm + 1e-8)

    scores = np.dot(ref_embeddings, query_emb.T).reshape(-1)

    order = np.argsort(scores)[::-1][:top_n]

    print("Inference time:", time.time() - t0, file=sys.stderr)

    return {
        "predicted_signs": [labels[i] for i in order],
        "confidences": [float(scores[i]) for i in order],
        "model_type": "embedding_similarity",
    }


# ------------------------------
# CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--reference-json", required=True)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    model = load_model_fast(Path(args.model))
    labels, ref_embeddings = load_reference_library(Path(args.reference_json))

    result = run_worker(
        model=model,
        labels=labels,
        ref_embeddings=ref_embeddings,
        query_path=Path(args.query),
        top_n=args.top_n,
    )

    print(json.dumps(result))


if __name__ == "__main__":
    main()