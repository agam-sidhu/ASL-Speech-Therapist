from __future__ import annotations

import importlib
import json
import sys
import types
from collections import Counter, deque
from pathlib import Path
from typing import Any

import cv2  
import joblib
import keras
import mediapipe as mp  # type: ignore[import]
import numpy as np
import tensorflow as tf

# Compatibility alias for pickles produced with NumPy 2.x internals.
if "numpy._core" not in sys.modules:
    try:
        import numpy.core as _numpy_core  # type: ignore[import]

        sys.modules["numpy._core"] = _numpy_core
    except Exception:
        pass


def _load_mediapipe_solutions():
    solutions = getattr(mp, "solutions", None)
    if solutions is not None:
        return solutions

    try:
        from mediapipe.python import solutions as python_solutions  # type: ignore[import]

        return python_solutions
    except Exception:
        return None


def _load_mediapipe_hands_class():
    errors: list[str] = []

    mp_hands = getattr(getattr(mp, "solutions", None), "hands", None)
    hands_from_mp = getattr(mp_hands, "Hands", None)
    if hands_from_mp is not None:
        return hands_from_mp, None
    errors.append("mediapipe.solutions.hands.Hands missing")

    try:
        hands_module = importlib.import_module("mediapipe.python.solutions.hands")
        hands_cls = getattr(hands_module, "Hands", None)
        if hands_cls is not None:
            return hands_cls, None
        errors.append("mediapipe.python.solutions.hands.Hands missing")
    except Exception as exc:
        errors.append(f"mediapipe.python.solutions.hands import failed: {exc}")

    return None, " | ".join(errors)


def _create_hands_tracker(preferred_hands_class=None):
    hands_class = preferred_hands_class
    load_error: str | None = None
    if hands_class is None:
        hands_class, load_error = _load_mediapipe_hands_class()

    if hands_class is None:
        return None, load_error or "MediaPipe hands class is unavailable."

    try:
        return (
            hands_class(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ),
            None,
        )
    except Exception as exc:
        return None, f"Failed to initialize MediaPipe Hands tracker: {exc}"


MP_SOLUTIONS = _load_mediapipe_solutions()
MP_HANDS_CLASS, MP_HANDS_IMPORT_ERROR = _load_mediapipe_hands_class()


def _normalize_layer_config_dtype(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    dtype_cfg = cfg.get("dtype")
    if isinstance(dtype_cfg, dict):
        class_name = dtype_cfg.get("class_name")
        if class_name in {"DTypePolicy", "Policy"}:
            cfg["dtype"] = dtype_cfg.get("config", {}).get("name", "float32")
    return cfg


def _install_keras_functional_compat_shim() -> None:
    try:
        import keras as standalone_keras  # type: ignore[import]
    except Exception:
        standalone_keras = None

    functional_class = getattr(getattr(standalone_keras, "src", None), "models", None)
    functional_class = getattr(functional_class, "functional", None)
    functional_class = getattr(functional_class, "Functional", None)

    if functional_class is None:
        try:
            from keras.engine.functional import Functional as keras_functional_class  # type: ignore[import]

            functional_class = keras_functional_class
        except Exception:
            functional_class = tf.keras.Model

    if standalone_keras is not None:
        if not hasattr(standalone_keras, "DTypePolicy"):
            setattr(standalone_keras, "DTypePolicy", tf.keras.mixed_precision.Policy)
        if not hasattr(standalone_keras, "Policy"):
            setattr(standalone_keras, "Policy", tf.keras.mixed_precision.Policy)

    keras_module = sys.modules.setdefault("keras", types.ModuleType("keras"))
    src_module = sys.modules.setdefault("keras.src", types.ModuleType("keras.src"))
    models_module = sys.modules.setdefault("keras.src.models", types.ModuleType("keras.src.models"))
    functional_module = sys.modules.setdefault(
        "keras.src.models.functional", types.ModuleType("keras.src.models.functional")
    )

    setattr(keras_module, "src", src_module)
    setattr(src_module, "models", models_module)
    setattr(models_module, "functional", functional_module)
    setattr(functional_module, "Functional", functional_class)
    setattr(functional_module, "Model", tf.keras.Model)

    class CompatibleInputLayer(tf.keras.layers.InputLayer):
        @classmethod
        def from_config(cls, config):
            config = dict(config)
            batch_shape = config.pop("batch_shape", None)
            if batch_shape is not None and "batch_input_shape" not in config:
                config["batch_input_shape"] = batch_shape
            config.pop("optional", None)
            return super().from_config(config)

    original_input_layer_from_config = tf.keras.layers.InputLayer.from_config

    @classmethod
    def _compat_input_layer_from_config(cls, config):
        config = dict(config)
        batch_shape = config.pop("batch_shape", None)
        if batch_shape is not None and "batch_input_shape" not in config:
            config["batch_input_shape"] = batch_shape
        config.pop("optional", None)
        return original_input_layer_from_config(config)

    tf.keras.layers.InputLayer.from_config = _compat_input_layer_from_config  # type: ignore[assignment]

    original_dense_from_config = tf.keras.layers.Dense.from_config

    @classmethod
    def _compat_dense_from_config(cls, config):
        config = dict(config)
        config.pop("quantization_config", None)
        return original_dense_from_config(config)

    tf.keras.layers.Dense.from_config = _compat_dense_from_config  # type: ignore[assignment]

    original_layernorm_from_config = tf.keras.layers.LayerNormalization.from_config

    @classmethod
    def _compat_layernorm_from_config(cls, config):
        config = dict(config)
        config.pop("rms_scaling", None)
        return original_layernorm_from_config(config)

    tf.keras.layers.LayerNormalization.from_config = _compat_layernorm_from_config  # type: ignore[assignment]

    original_batchnorm_from_config = tf.keras.layers.BatchNormalization.from_config

    @classmethod
    def _compat_batchnorm_from_config(cls, config):
        config = _normalize_layer_config_dtype(dict(config))
        return original_batchnorm_from_config(config)

    tf.keras.layers.BatchNormalization.from_config = _compat_batchnorm_from_config  # type: ignore[assignment]

    tf.keras.utils.get_custom_objects()["DTypePolicy"] = tf.keras.mixed_precision.Policy
    tf.keras.utils.get_custom_objects()["Policy"] = tf.keras.mixed_precision.Policy

    input_layer_module = sys.modules.setdefault(
        "keras.src.layers.core.input_layer", types.ModuleType("keras.src.layers.core.input_layer")
    )
    layers_core_module = sys.modules.setdefault("keras.src.layers.core", types.ModuleType("keras.src.layers.core"))
    layers_module = sys.modules.setdefault("keras.src.layers", types.ModuleType("keras.src.layers"))

    setattr(layers_module, "core", layers_core_module)
    setattr(layers_core_module, "input_layer", input_layer_module)
    setattr(input_layer_module, "InputLayer", CompatibleInputLayer)
    setattr(input_layer_module, "Layer", tf.keras.layers.Layer)


_install_keras_functional_compat_shim()


@tf.keras.utils.register_keras_serializable(package="Custom")
class SkeletonAugmentation(tf.keras.layers.Layer):
    def __init__(self, noise_factor: float = 0.0, scale_range: tuple[float, float] | list[float] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.noise_factor = noise_factor
        self.scale_range = tuple(scale_range) if scale_range is not None else None

    def call(self, inputs, training=None):
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "noise_factor": self.noise_factor,
                "scale_range": self.scale_range,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**_normalize_layer_config_dtype(config))


@tf.keras.utils.register_keras_serializable(package="Custom")
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.coord_proj = tf.keras.layers.Dense(embed_dim)

    def call(self, x, training=None):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.coord_proj(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({"maxlen": self.maxlen, "embed_dim": self.embed_dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**_normalize_layer_config_dtype(config))


@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        if isinstance(attn_output, (tuple, list)):
            attn_output = attn_output[0]
        training_flag = bool(training)
        attn_output = self.dropout1(attn_output, training=training_flag)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training_flag)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "rate": self.rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**_normalize_layer_config_dtype(config))


@tf.keras.utils.register_keras_serializable(package="Custom")
class LateDropout(tf.keras.layers.Layer):
    def __init__(self, rate: float = 0.5, start_step: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.start_step = start_step

    def build(self, input_shape):
        self.step_counter = self.add_weight(
            name="step_counter",
            shape=(),
            initializer="zeros",
            trainable=False,
            dtype=tf.int64,
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        if not training:
            return inputs

        self.step_counter.assign_add(1)
        return tf.cond(
            self.step_counter >= tf.cast(self.start_step, tf.int64),
            lambda: tf.nn.dropout(inputs, rate=self.rate),
            lambda: inputs,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rate": self.rate,
                "start_step": self.start_step,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**_normalize_layer_config_dtype(config))


@tf.keras.utils.register_keras_serializable(package="Custom")
class SpatialGraphConv(tf.keras.layers.Layer):
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.A = self._build_adjacency_matrix()
        self.dense = tf.keras.layers.Dense(units, activation="relu")

    @staticmethod
    def _build_adjacency_matrix() -> tf.Tensor:
        num_joints = 75
        a = np.zeros((num_joints, num_joints), dtype=np.float32)

        pose_edges = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
            (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
        ]
        hand_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
        ]

        edges: list[tuple[int, int]] = []
        edges.extend(pose_edges)
        for u, v in hand_edges:
            edges.append((u + 33, v + 33))
        for u, v in hand_edges:
            edges.append((u + 54, v + 54))
        edges.append((15, 33))
        edges.append((16, 54))

        for u, v in edges:
            a[u, v] = 1.0
            a[v, u] = 1.0

        a = a + np.eye(num_joints, dtype=np.float32)
        d = np.sum(a, axis=1)
        d_inv_sqrt = np.power(d, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat = np.diag(d_inv_sqrt)
        a_normalized = np.dot(d_mat, np.dot(a, d_mat))
        return tf.constant(a_normalized, dtype=tf.float32)

    def call(self, inputs, training=None):
        h = self.dense(inputs)
        return tf.einsum("vw,btwc->btvc", self.A, h)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**_normalize_layer_config_dtype(config))


@tf.keras.utils.register_keras_serializable(package="Custom")
class MambaStyleBlock(tf.keras.layers.Layer):
    def __init__(self, d_model: int, kernel_size: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.norm = tf.keras.layers.LayerNormalization()
        self.conv1d = tf.keras.layers.Conv1D(
            filters=d_model,
            kernel_size=kernel_size,
            padding="causal",
            activation="silu",
        )
        self.ssm_scan = tf.keras.layers.GRU(d_model, return_sequences=True)
        self.out_proj = tf.keras.layers.Dense(d_model)

    def call(self, inputs, training=None):
        x = self.norm(inputs)
        x_conv = self.conv1d(x)
        x_ssm = self.ssm_scan(x_conv)
        out = self.out_proj(x_ssm)
        return inputs + out

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "kernel_size": self.kernel_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**_normalize_layer_config_dtype(config))


@tf.keras.utils.register_keras_serializable(package="Custom")
class AbsDiffLayer(tf.keras.layers.Layer):
    def call(self, inputs, training=None):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            return tf.abs(inputs[0] - inputs[1])
        return tf.abs(inputs)

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = Path(__file__).resolve().parents[3]

JOEL_535_DIR_CANDIDATES = [
    PROJECT_DIR / "535 Project",
    PROJECT_DIR / "ASL-Speech-Therapist" / "Everything" / "Joel's work" / "535 Project",
]
JOEL_535_DIR = next((path for path in JOEL_535_DIR_CANDIDATES if path.exists()), JOEL_535_DIR_CANDIDATES[0])

MODEL_PATHS = [
    BACKEND_DIR / "artifacts" / "joel" / "best_graph_siamese_simease2000_compat.keras",
]

ENCODER_PATHS = [
    JOEL_535_DIR / "wlasl_label_encoder_simease2000.pkl",
]

SIA_REFERENCE_PATHS = {
    "best_graph_siamese_simease2000_compat.keras": (
        JOEL_535_DIR / "X_data_standardized.npy",
        JOEL_535_DIR / "y_labels.npy",
    ),
    "best_graph_siamese_simease2000.keras": (
        JOEL_535_DIR / "X_data_standardized.npy",
        JOEL_535_DIR / "y_labels.npy",
    ),
}

CONFIDENCE_THRESHOLD = 0.85
BUFFER_SIZE = 5


class JsonLabelEncoder:
    def __init__(self, classes: list[str]) -> None:
        if not classes:
            raise ValueError("Label classes are empty")
        self.classes_ = np.array(classes, dtype=object)

    def inverse_transform(self, indices):
        out = []
        for idx in indices:
            i = int(idx)
            if i < 0 or i >= len(self.classes_):
                raise ValueError(f"Label index out of range: {i}")
            out.append(self.classes_[i])
        return np.array(out, dtype=object)


def _load_label_encoder(candidate: Path):
    if candidate.suffix.lower() == ".json":
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            classes = payload.get("classes", [])
        elif isinstance(payload, list):
            classes = payload
        else:
            raise ValueError(f"Unsupported label JSON structure in {candidate}")
        if not isinstance(classes, list):
            raise ValueError(f"Label classes must be a list in {candidate}")
        return JsonLabelEncoder([str(v) for v in classes])

    return joblib.load(str(candidate))


class JoelASLRecognizer:
    def __init__(
        self,
        model_path: Path | None = None,
        encoder_path: Path | None = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        buffer_size: int = BUFFER_SIZE,
    ) -> None:
        model_candidates: list[Path]
        if model_path is not None:
            model_candidates = [model_path]
        else:
            model_candidates = [path for path in MODEL_PATHS if path.exists()]
            if not model_candidates:
                model_candidates = MODEL_PATHS

        custom_objects = {
            "SkeletonAugmentation": SkeletonAugmentation,
            "Custom>SkeletonAugmentation": SkeletonAugmentation,
            "PositionalEmbedding": PositionalEmbedding,
            "Custom>PositionalEmbedding": PositionalEmbedding,
            "TransformerBlock": TransformerBlock,
            "Custom>TransformerBlock": TransformerBlock,
            "LateDropout": LateDropout,
            "Custom>LateDropout": LateDropout,
            "SpatialGraphConv": SpatialGraphConv,
            "Custom>SpatialGraphConv": SpatialGraphConv,
            "MambaStyleBlock": MambaStyleBlock,
            "Custom>MambaStyleBlock": MambaStyleBlock,
            "AbsDiffLayer": AbsDiffLayer,
            "Custom>AbsDiffLayer": AbsDiffLayer,
            "DTypePolicy": tf.keras.mixed_precision.Policy,
            "Policy": tf.keras.mixed_precision.Policy,
        }

        model_errors: list[str] = []
        loaded_model: Any | None = None
        loaded_model_path: Path | None = None
        try:
            tf.keras.config.enable_unsafe_deserialization()
        except Exception:
            pass
        for candidate in model_candidates:
            try:
                try:
                    loaded_model = keras.models.load_model(
                        str(candidate),
                        custom_objects=custom_objects,
                        compile=False,
                        safe_mode=False,
                    )
                except Exception:
                    loaded_model = tf.keras.models.load_model(
                        str(candidate),
                        custom_objects=custom_objects,
                        compile=False,
                        safe_mode=False,
                    )
                loaded_model_path = candidate
                break
            except Exception as exc:
                model_errors.append(f"{candidate}: {exc}")

        if loaded_model is None or loaded_model_path is None:
            raise RuntimeError(
                "Failed to load Joel ASL model from candidates. "
                + " | ".join(model_errors)
                + ". Try converting the newer model with: python -m src.cv.convert_joel_model"
            )

        encoder_candidates: list[Path]
        if encoder_path is not None:
            encoder_candidates = [encoder_path]
        else:
            if loaded_model_path.name == "best_graph_siamese_simease2000.keras":
                preferred_encoders = [JOEL_535_DIR / "wlasl_label_encoder_simease2000.pkl"]
            else:
                preferred_encoders = []

            existing_preferred = [path for path in preferred_encoders if path.exists()]
            existing_generic = [path for path in ENCODER_PATHS if path.exists()]
            encoder_candidates = existing_preferred + [p for p in existing_generic if p not in existing_preferred]
            if not encoder_candidates:
                encoder_candidates = ENCODER_PATHS

        encoder_errors: list[str] = []
        loaded_encoder: Any | None = None
        loaded_encoder_path: Path | None = None
        for candidate in encoder_candidates:
            try:
                loaded_encoder = _load_label_encoder(candidate)
                loaded_encoder_path = candidate
                break
            except Exception as exc:
                encoder_errors.append(f"{candidate}: {exc}")

        if loaded_encoder is None or loaded_encoder_path is None:
            raise RuntimeError("Failed to load Joel label encoder. " + " | ".join(encoder_errors))

        self.model = loaded_model
        self.label_encoder = loaded_encoder
        self.model_path = loaded_model_path
        self.encoder_path = loaded_encoder_path
        self.confidence_threshold = confidence_threshold
        self.buffer_size = buffer_size
        self.prediction_buffer = deque(maxlen=buffer_size)

        self.mode = "classifier"
        self.reference_labels: np.ndarray | None = None
        self.reference_samples: np.ndarray | None = None

        input_shape = self.model.input_shape
        if isinstance(input_shape, list) and len(input_shape) == 2 and len(input_shape[0]) == 4:
            self.mode = "siamese"
            self.sequence_len = int(input_shape[0][1])
            self.feature_dim = int(input_shape[0][3])
            self.frame_buffer: deque[np.ndarray] = deque(maxlen=0)
            self.hands = object()
            self.hands_init_error = None
            self._load_siamese_reference_bank()
        elif isinstance(input_shape, tuple) and len(input_shape) == 3:
            self.mode = "classifier"
            self.sequence_len = int(input_shape[1])
            self.feature_dim = int(input_shape[2])
            self.frame_buffer = deque(maxlen=self.sequence_len)

            self.hands, self.hands_init_error = _create_hands_tracker(preferred_hands_class=MP_HANDS_CLASS)
            if self.hands_init_error is None:
                self.hands_init_error = MP_HANDS_IMPORT_ERROR
        else:
            raise RuntimeError(f"Unexpected Joel model input shape: {input_shape}")

    def _load_siamese_reference_bank(self) -> None:
        refs = SIA_REFERENCE_PATHS.get(self.model_path.name)
        if refs is None:
            raise RuntimeError(f"No reference bank configured for siamese model: {self.model_path.name}")

        x_path, y_path = refs
        if not x_path.exists() or not y_path.exists():
            raise RuntimeError(f"Missing siamese reference arrays: {x_path} | {y_path}")

        x_data = np.load(str(x_path), mmap_mode="r")
        y_data = np.load(str(y_path), allow_pickle=True)

        if getattr(y_data, "dtype", None) is not None and y_data.dtype.kind in {"i", "u"}:
            y_labels = np.array(self.label_encoder.inverse_transform(y_data.astype(int)), dtype=object)
        else:
            y_labels = y_data.astype(object)

        unique_labels, first_idx = np.unique(y_labels, return_index=True)
        self.reference_labels = unique_labels
        self.reference_samples = np.array(x_data[first_idx], dtype=np.float32)

    def reset_state(self) -> None:
        self.prediction_buffer.clear()
        self.frame_buffer.clear()

    def _extract_75x3_sequence(self, video_path: str) -> np.ndarray:
        holistic_module = getattr(MP_SOLUTIONS, "holistic", None) if MP_SOLUTIONS is not None else None
        if holistic_module is None:
            raise RuntimeError("MediaPipe not initialized")
            return np.zeros((0, 75, 3), dtype=np.float32)

        holistic_cls = getattr(holistic_module, "Holistic", None)
        if holistic_cls is None:
            return np.zeros((0, 75, 3), dtype=np.float32)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        frames: list[np.ndarray] = []
        with holistic_cls(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = holistic.process(rgb)
                frame_coords = np.zeros((75, 3), dtype=np.float32)

                if res.pose_landmarks is not None:
                    for i, lm in enumerate(res.pose_landmarks.landmark[:33]):
                        frame_coords[i] = [lm.x, lm.y, lm.z]

                if res.left_hand_landmarks is not None:
                    for i, lm in enumerate(res.left_hand_landmarks.landmark[:21]):
                        frame_coords[33 + i] = [lm.x, lm.y, lm.z]

                if res.right_hand_landmarks is not None:
                    for i, lm in enumerate(res.right_hand_landmarks.landmark[:21]):
                        frame_coords[54 + i] = [lm.x, lm.y, lm.z]

                frames.append(frame_coords)

        cap.release()

        if not frames:
            return np.zeros((0, 75, 3), dtype=np.float32)
        return np.stack(frames, axis=0)

    def _preprocess_siamese_sequence(self, seq_3d: np.ndarray) -> np.ndarray:
        if seq_3d.size == 0:
            return np.zeros((self.sequence_len, 75, 9), dtype=np.float32)

        center = seq_3d[:, 11:12, :]
        seq_3d = seq_3d - center
        shoulder_dist = np.linalg.norm(seq_3d[:, 11] - seq_3d[:, 12], axis=-1)
        scale = float(np.mean(shoulder_dist) + 1e-6)
        seq_3d = seq_3d / scale

        vel = np.zeros_like(seq_3d)
        vel[1:] = seq_3d[1:] - seq_3d[:-1]

        acc = np.zeros_like(seq_3d)
        acc[1:] = vel[1:] - vel[:-1]

        kinematic = np.concatenate([seq_3d, vel, acc], axis=-1)
        if kinematic.shape[0] >= self.sequence_len:
            return kinematic[: self.sequence_len].astype(np.float32)

        pad_width = self.sequence_len - kinematic.shape[0]
        return np.pad(
            kinematic,
            ((0, pad_width), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0.0,
        ).astype(np.float32)

    @staticmethod
    def _normalize_landmarks(landmarks) -> np.ndarray:
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
        wrist = coords[0]
        coords = coords - wrist
        max_val = np.max(np.abs(coords))
        if max_val > 0:
            coords = coords / max_val
        return coords.flatten()

    def _to_model_feature(self, hand_landmarks) -> np.ndarray:
        normalized = self._normalize_landmarks(hand_landmarks)
        if normalized.size < self.feature_dim:
            padded = np.zeros((self.feature_dim,), dtype=np.float32)
            padded[: normalized.size] = normalized
            return padded
        return normalized[: self.feature_dim].astype(np.float32)

    def _predict_sequence(self, sequence: np.ndarray) -> tuple[str | None, float]:
        prediction = self.model.predict(np.array([sequence]), verbose=0)
        class_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        if confidence < self.confidence_threshold:
            return None, confidence

        label = self.label_encoder.inverse_transform([class_idx])[0]
        return str(label), confidence

    def analyze_video(self, video_path: str, expected: str | None = None):

        # =========================================================
        # 🔥 SIAMESE MODE (MAIN PIPELINE)
        # =========================================================
        if self.mode == "siamese":

            seq_3d = self._extract_75x3_sequence(video_path)
            frames_processed = int(seq_3d.shape[0])

            if frames_processed == 0:
                return {
                    "predicted_signs": [],
                    "confidences": [],
                    "frames_processed": 0,
                    "warning": "No landmarks detected",
                }

            query = self._preprocess_siamese_sequence(seq_3d)

            # 🔥 BUILD EMBEDDING MODEL ONCE
            if not hasattr(self, "embedding_model"):
                self.embedding_model = tf.keras.Model(
                    inputs=self.model.input[0],
                    outputs=self.model.get_layer("unit_normalization_1").output
                )

            # 🔥 QUERY EMBEDDING
            query_emb = self.embedding_model.predict(query[np.newaxis, ...], verbose=0)
            query_emb /= np.linalg.norm(query_emb, axis=1, keepdims=True)

            # 🔥 REFERENCE EMBEDDINGS (PRECOMPUTED)
            if not hasattr(self, "ref_embeddings"):
                print("⚡ Loading embeddings from JSON (FAST)...")

                json_path = Path(__file__).resolve().parents[2] / "artifacts/joel/asl_reference_library.json"

                with open(json_path, "r") as f:
                    data = json.load(f)

                self.reference_labels = list(data.keys())
                self.ref_embeddings = np.array(list(data.values()), dtype=np.float32)

                # normalize once
                self.ref_embeddings /= np.linalg.norm(self.ref_embeddings, axis=1, keepdims=True)

                print("✅ Loaded embeddings:", self.ref_embeddings.shape)

            # 🔥 FAST COSINE SIMILARITY
            if expected:
                expected = expected.upper()

                filtered_indices = [
                    i for i, label in enumerate(self.reference_labels)
                    if expected in label.upper()
                ]

                if filtered_indices:
                    ref_emb = self.ref_embeddings[filtered_indices]
                    ref_labels = [self.reference_labels[i] for i in filtered_indices]
                else:
                    ref_emb = self.ref_embeddings
                    ref_labels = self.reference_labels
            else:
                ref_emb = self.ref_embeddings
                ref_labels = self.reference_labels
            # 🔥 TOP K
            scores = np.dot(ref_emb, query_emb.T).reshape(-1)
            top_k = 5
            idx = np.argsort(scores)[::-1][:top_k]

            predicted_signs = [str(ref_labels[i]) for i in idx]
            confidences = [float(scores[i]) for i in idx]

            return {
                "predicted_signs": predicted_signs,
                "confidences": confidences,
                "frames_processed": frames_processed,
                "model_type": "embedding_similarity_fast",
            }
        # =========================================================
        # 🔥 CLASSIFIER MODE (UNCHANGED)
        # =========================================================

        if self.hands is None:
            self.hands, self.hands_init_error = _create_hands_tracker(preferred_hands_class=None)

        hands_tracker = self.hands

        if hands_tracker is None:
            return {
                "predicted_signs": [],
                "confidences": [],
                "frames_processed": 0,
                "warning": "MediaPipe not available",
                "error": self.hands_init_error,
            }

        process_fn = getattr(hands_tracker, "process", None)

        if process_fn is None:
            return {
                "predicted_signs": [],
                "confidences": [],
                "frames_processed": 0,
                "warning": "MediaPipe process not callable",
            }

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        detected_signs = []
        detected_confidences = []
        last_emitted = None
        frame_count = 0

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = process_fn(rgb)

            if not results.multi_hand_landmarks:
                self.prediction_buffer.clear()
                self.frame_buffer.clear()
                continue

            for hand_landmarks in results.multi_hand_landmarks:
                feature = self._to_model_feature(hand_landmarks)
                self.frame_buffer.append(feature)

                if len(self.frame_buffer) < self.sequence_len:
                    continue

                sequence = np.stack(list(self.frame_buffer), axis=0)
                label, confidence = self._predict_sequence(sequence)

                if label is None:
                    self.prediction_buffer.append("Unknown")
                else:
                    self.prediction_buffer.append(label)

                if len(self.prediction_buffer) == self.buffer_size:
                    most_common = Counter(self.prediction_buffer).most_common(1)[0][0]
                    if most_common != "Unknown" and most_common != last_emitted:
                        detected_signs.append(most_common)
                        detected_confidences.append(confidence)
                        last_emitted = most_common

        cap.release()

        return {
            "predicted_signs": detected_signs,
            "confidences": detected_confidences,
            "frames_processed": frame_count,
        }