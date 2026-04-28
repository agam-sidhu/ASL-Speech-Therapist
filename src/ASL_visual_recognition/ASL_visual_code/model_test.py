import numpy as np
import json
import os
import cv2
import mediapipe as mp
from mediapipe.python.solutions import holistic as mp_holistic # Explicit MediaPipe bypass
import tensorflow as tf
import keras # Native Keras 3
from keras import layers

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
SCRIPT_DIR = os.getcwd() if '__file__' not in globals() else os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(SCRIPT_DIR, "00414.mp4")           # Input MP4
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "my_test_video.npy")  # Intermediate NPY
LIBRARY_PATH = os.path.join(SCRIPT_DIR, "asl_reference_library.json")
ENCODER_PATH = os.path.join(SCRIPT_DIR, "best_graph_simease2000.keras")
TF_ENABLE_ONEDNN_OPTS=0

MAX_FRAMES = 100

# ==========================================
# 2. MEDIAPIPE EXTRACTION DEFINITIONS
# ==========================================
def extract_landmarks(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([pose, lh, rh]).flatten()

def process_video_to_npy(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frames_data = []
    print(f"🎥 Processing video: {os.path.basename(video_path)}...")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            keypoints = extract_landmarks(results)
            frames_data.append(keypoints)
    cap.release()
    sequence_array = np.array(frames_data)
    np.save(output_path, sequence_array)
    print(f"✅ Extracted {sequence_array.shape[0]} frames of skeleton data to {os.path.basename(output_path)}")

# ==========================================
# 3. KINEMATIC PREPROCESSING
# ==========================================
def normalize_skeleton(seq_3d):
    center = seq_3d[:, 11:12, :]
    seq_3d = seq_3d - center
    shoulder_dist = np.linalg.norm(seq_3d[:, 11] - seq_3d[:, 12], axis=-1)
    scale = np.mean(shoulder_dist) + 1e-6
    return seq_3d / scale

def process_seq(seq, max_frames=MAX_FRAMES):
    frames_count = seq.shape[0]
    if seq.ndim == 2: 
        seq_3d = seq.reshape((frames_count, 75, 3))
    else:
        seq_3d = seq
    seq_3d = normalize_skeleton(seq_3d)
    vel = np.zeros_like(seq_3d)
    vel[1:] = seq_3d[1:] - seq_3d[:-1]
    acc = np.zeros_like(seq_3d)
    acc[1:] = vel[1:] - vel[:-1]
    kinematic_seq = np.concatenate([seq_3d, vel, acc], axis=-1)
    if frames_count >= max_frames:
        kinematic_seq = kinematic_seq[:max_frames]
    else:
        pad_width = max_frames - frames_count
        kinematic_seq = np.pad(kinematic_seq, ((0, pad_width), (0, 0), (0, 0)), mode='constant', constant_values=0.0)
    return kinematic_seq

# ==========================================
# 4. CUSTOM LAYER BLUEPRINTS
# ==========================================
def build_adjacency_matrix():
    NUM_JOINTS = 75
    A = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.float32)
    pose_edges = [
        (0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), (9,10),
        (11,12), (11,13), (13,15), (15,17), (15,19), (15,21), (17,19),
        (12,14), (14,16), (16,18), (16,20), (16,22), (18,20),
        (11,23), (12,24), (23,24), (23,25), (25,27), (27,29), (27,31), (29,31),
        (24,26), (26,28), (28,30), (28,30), (30,32)
    ]
    hand_edges = [
        (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
        (5,9), (9,10), (10,11), (11,12), (9,13), (13,14), (14,15), (15,16),
        (13,17), (0,17), (17,18), (18,19), (19,20)
    ]
    edges = pose_edges + [(u+33, v+33) for u, v in hand_edges] + [(u+54, v+54) for u, v in hand_edges] + [(15, 33), (16, 54)]

    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0

    I = np.eye(NUM_JOINTS)
    A = A + I
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
    D_mat = np.diag(D_inv_sqrt)
    return tf.constant(np.dot(D_mat, np.dot(A, D_mat)), dtype=tf.float32)

@keras.saving.register_keras_serializable()
class SpatialGraphConv(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.A = build_adjacency_matrix()
        self.dense = layers.Dense(units, activation='relu')
    def call(self, inputs):
        h = self.dense(inputs)
        return tf.einsum('vw, btwc -> btvc', self.A, h)

@keras.saving.register_keras_serializable()
class MambaStyleBlock(layers.Layer):
    def __init__(self, d_model, kernel_size=4, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.norm = layers.LayerNormalization()
        self.conv1d = layers.Conv1D(filters=d_model, kernel_size=kernel_size, padding='causal', activation='silu')
        self.ssm_scan = layers.GRU(d_model, return_sequences=True)
        self.out_proj = layers.Dense(d_model)
    def call(self, inputs):
        x = self.norm(inputs)
        x_conv = self.conv1d(x)
        x_ssm = self.ssm_scan(x_conv)
        out = self.out_proj(x_ssm)
        return inputs + out

@keras.saving.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# ==========================================
# 5. MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    
    if os.path.exists(VIDEO_PATH) and not os.path.exists(OUTPUT_PATH):
        process_video_to_npy(VIDEO_PATH, OUTPUT_PATH)

    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"❌ Model not found at {ENCODER_PATH}. Move it to this folder!")
        
    print("\n🔄 Loading AI Model and Library...")

    # ==========================================
    # 🚨 KERAS 3 LAMBDA SHAPE BYPASS PATCH
    # ==========================================
    original_compute = keras.layers.Lambda.compute_output_shape
    def bypassed_compute(self, input_shape):
        try:
            return original_compute(self, input_shape)
        except NotImplementedError:
            # Fallback if Keras 3 cannot infer the shape of the custom math
            return input_shape[0] if isinstance(input_shape, (list, tuple)) else input_shape
    keras.layers.Lambda.compute_output_shape = bypassed_compute
    # ==========================================
    
    custom_objs = {
        'SpatialGraphConv': SpatialGraphConv,
        'MambaStyleBlock': MambaStyleBlock,
        'TransformerBlock': TransformerBlock
    }
    
    # Load architecture and weights naturally
    full_siamese_model = keras.models.load_model(
        ENCODER_PATH, 
        custom_objects=custom_objs, 
        compile=False, 
        safe_mode=False
    )
    
    # Extract the twin encoder
    encoder = full_siamese_model.get_layer('Graph_Encoder')
    print("✅ Model loaded successfully!")

    with open(LIBRARY_PATH, 'r') as f:
        reference_library = json.load(f)

    if os.path.exists(OUTPUT_PATH):
        print("\n🔍 Analyzing skeleton sequence and running Siamese comparison...")
        raw_skeleton = np.load(OUTPUT_PATH)
        processed_skeleton = process_seq(raw_skeleton)
        processed_skeleton = np.expand_dims(processed_skeleton, axis=0)
        
        # 1. Compute user embedding ONCE
        user_embedding = encoder.predict(processed_skeleton, verbose=0)
        
        user_embedding_flat = user_embedding[0] 

        results = []
        for word, ref_data in reference_library.items():
            
            # 2. Load the reference embedding (It's ALREADY a fingerprint in the JSON!)
            ref_embedding_flat = np.array(ref_data)

            # 3. Siamese distance (Calculate the difference between the two arrays)
            distance = np.mean(np.abs(user_embedding_flat - ref_embedding_flat))

            # 4. Convert distance → similarity (Closer to 1.0 means a better match)
            similarity = np.exp(-distance)

            results.append((word, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Output Results
        best_match_id, best_score = results[0]
        
        print("\n==================================")
        print("🎯 PREDICTION RESULTS")
        print("==================================")
        print(f"Predicted Signage ID : {best_match_id}")
        print(f"Match Confidence     : {best_score:.4f}")
        
        if best_score > 0.70:
            print("Status               : ✅ STRONG MATCH")
        else:
            print("Status               : ❌ POOR MATCH (Try signing again)")
        print("==================================\n")