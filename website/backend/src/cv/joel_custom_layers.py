from __future__ import annotations

import numpy as np
import tensorflow as tf
import keras


@keras.saving.register_keras_serializable(package="Custom")
class SkeletonAugmentation(keras.layers.Layer):
    def __init__(self, noise_factor: float = 0.0, scale_range: tuple[float, float] | list[float] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.noise_factor = noise_factor
        self.scale_range = tuple(scale_range) if scale_range is not None else None

    def call(self, inputs, training=None):
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"noise_factor": self.noise_factor, "scale_range": self.scale_range})
        return config


@keras.saving.register_keras_serializable(package="Custom")
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, maxlen: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.coord_proj = keras.layers.Dense(embed_dim)

    def call(self, x, training=None):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.coord_proj(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({"maxlen": self.maxlen, "embed_dim": self.embed_dim})
        return config


@keras.saving.register_keras_serializable(package="Custom")
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

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
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate})
        return config


@keras.saving.register_keras_serializable(package="Custom")
class LateDropout(keras.layers.Layer):
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
        config.update({"rate": self.rate, "start_step": self.start_step})
        return config


@keras.saving.register_keras_serializable(package="Custom")
class SpatialGraphConv(keras.layers.Layer):
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.A = self._build_adjacency_matrix()
        self.dense = keras.layers.Dense(units, activation="relu")

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


@keras.saving.register_keras_serializable(package="Custom")
class MambaStyleBlock(keras.layers.Layer):
    def __init__(self, d_model: int, kernel_size: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.norm = keras.layers.LayerNormalization()
        self.conv1d = keras.layers.Conv1D(filters=d_model, kernel_size=kernel_size, padding="causal", activation="silu")
        self.ssm_scan = keras.layers.GRU(d_model, return_sequences=True)
        self.out_proj = keras.layers.Dense(d_model)

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


@keras.saving.register_keras_serializable(package="Custom")
class AbsDiffLayer(keras.layers.Layer):
    def call(self, inputs, training=None):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            return tf.abs(inputs[0] - inputs[1])
        return tf.abs(inputs)
