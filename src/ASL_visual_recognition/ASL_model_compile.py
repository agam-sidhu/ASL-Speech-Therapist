import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import PyDataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder


# --- 1. Load the Extracted Data ---
print("Loading data from disk...")
# Adjust filenames if you named them differently in the previous step
X_raw = np.load("X_data_standardized.npy", allow_pickle=True)
y_raw = np.load("y_labels.npy", allow_pickle=True)

# --- 2. Encode the Labels ---
# Neural networks can't read strings like "hello". We must convert them to integers.
print("Encoding labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
NUM_CLASSES = len(le.classes_)

# Save the encoder so we can translate numbers back to words in real-time later!
joblib.dump(le, "wlasl_label_encoder.pkl") 

# --- 3. Train / Validation Split ---
X_train, X_val, y_train, y_val = train_test_split(
    X_raw, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Training samples: {len(X_train)} | Validation samples: {len(X_val)}")

# --- 4. The Data Generator ---
class KinematicDataGenerator(PyDataset): # <--- Inherit from PyDataset now
    def __init__(self, X_data, y_data, batch_size=32, max_frames=100, **kwargs):
        super().__init__(**kwargs) # <--- Required by PyDataset initialization
        self.X = X_data
        self.y = y_data
        self.batch_size = batch_size
        self.max_frames = max_frames

    def __len__(self):
        """Returns the number of batches per epoch."""
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        """Builds and returns one batch of data."""
        batch_x = self.X[index * self.batch_size : (index + 1) * self.batch_size]
        batch_y = self.y[index * self.batch_size : (index + 1) * self.batch_size]
        
        processed_x = []
        
        for seq in batch_x:
            # 1. Un-flatten the sequence! 
            # From (frames, 225) -> (frames, 75 joints, 3 coords)
            frames_count = seq.shape[0]
            seq_3d = seq.reshape((frames_count, 75, 3)) 
            
            # 2. Calculate Kinematics on the 3D data
            vel = np.zeros_like(seq_3d)
            vel[1:] = seq_3d[1:] - seq_3d[:-1]
            
            acc = np.zeros_like(seq_3d)
            acc[1:] = vel[1:] - vel[:-1]
            
            # 3. Concatenate the features: (frames, 75, 9)
            kinematic_seq = np.concatenate([seq_3d, vel, acc], axis=-1)
            
            # 4. Dynamic Padding (Padding the frame dimension only)
            if frames_count >= self.max_frames:
                kinematic_seq = kinematic_seq[:self.max_frames]
            else:
                pad_width = self.max_frames - frames_count
                # Pad only the 0th axis (frames). Leave joints and features alone.
                kinematic_seq = np.pad(kinematic_seq, ((0, pad_width), (0, 0), (0, 0)), mode='constant', constant_values=0.0)
                
            processed_x.append(kinematic_seq)
            
        return np.array(processed_x), np.array(batch_y)
    
# --- 5. Instantiate Generators ---
MAX_FRAMES = 100 # Change this based on your dataset's average length
BATCH_SIZE = 32

train_gen = KinematicDataGenerator(X_train, y_train, batch_size=BATCH_SIZE, max_frames=MAX_FRAMES)
val_gen = KinematicDataGenerator(X_val, y_val, batch_size=BATCH_SIZE, max_frames=MAX_FRAMES)

print(f"Generators ready! Input feature size per frame: {X_raw[0].shape[-1] * 3}")

# Ask the generator for batch index 0 (the very first batch)
batch_x, batch_y = train_gen[0]

print("✅ Generator Test Successful!")
print(f"Batch X (Data) shape: {batch_x.shape}") 
print(f"Batch y (Labels) shape: {batch_y.shape}")

# --- 1. The Adjacency Matrix (Our Rulebook) ---
def build_adjacency_matrix():
    NUM_JOINTS = 75
    A = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.float32)
    
    # 1. Define the connection maps
    # --- MediaPipe Pose Connections (0-32) ---
    pose_edges = [
        (0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), (9,10), # Face
        (11,12), (11,13), (13,15), (15,17), (15,19), (15,21), (17,19),  # Left Arm & Hand Base
        (12,14), (14,16), (16,18), (16,20), (16,22), (18,20),           # Right Arm & Hand Base
        (11,23), (12,24), (23,24),                                      # Torso
        (23,25), (25,27), (27,29), (27,31), (29,31),                    # Left Leg
        (24,26), (26,28), (28,30), (28,32), (30,32)                     # Right Leg
    ]
    
    # --- MediaPipe Hand Connections (Internal 0-20 mapping) ---
    hand_edges = [
        (0,1), (1,2), (2,3), (3,4),         # Thumb
        (0,5), (5,6), (6,7), (7,8),         # Index Finger
        (5,9), (9,10), (10,11), (11,12),    # Middle Finger
        (9,13), (13,14), (14,15), (15,16),  # Ring Finger
        (13,17), (0,17), (17,18), (18,19), (19,20) # Pinky & Palm Base
    ]

    # 2. Apply the edges to the matrix
    edges = []
    
    # Add Pose
    edges.extend(pose_edges)
    
    # Add Left Hand (Offset by 33)
    for u, v in hand_edges:
        edges.append((u + 33, v + 33))
        
    # Add Right Hand (Offset by 54)
    for u, v in hand_edges:
        edges.append((u + 54, v + 54))
        
    # Add Bridges (Body Wrist to Hand Root)
    edges.append((15, 33)) # Left Wrist -> Left Hand
    edges.append((16, 54)) # Right Wrist -> Right Hand
    
    # 3. Populate the Matrix (Undirected graph = both directions)
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
        
    # 4. Add Self-Loops (Identity Matrix)
    # A joint needs to be connected to itself so it remembers its own position
    I = np.eye(NUM_JOINTS)
    A = A + I
    
    # 5. Graph Normalization (Crucial for Neural Networks)
    # If we don't normalize, joints with many connections (like the wrist) 
    # will "shout" over joints with few connections (like a fingertip).
    D = np.sum(A, axis=1) # Degree matrix (how many connections each joint has)
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0 # Handle division by zero
    
    # D^(-1/2) * A * D^(-1/2)
    D_mat = np.diag(D_inv_sqrt)
    A_normalized = np.dot(D_mat, np.dot(A, D_mat))
    
    # Convert to a fixed TensorFlow Constant so it can be baked into the Keras model
    return tf.constant(A_normalized, dtype=tf.float32)

# --- 2. The Spatial Graph Layer (The Eyes) ---
class SpatialGraphConv(layers.Layer):
    # Removed adjacency_matrix from arguments to avoid serialization errors
    def __init__(self, units, **kwargs): 
        super().__init__(**kwargs)
        self.units = units
        self.A = build_adjacency_matrix() # Fetch the rulebook internally!
        self.dense = layers.Dense(units, activation='relu')

    def call(self, inputs):
        h = self.dense(inputs) 
        out = tf.einsum('vw, btwc -> btvc', self.A, h)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

# --- 3. The Mamba-Style SSM Block (The Flow) ---
class MambaStyleBlock(layers.Layer):
    def __init__(self, d_model, kernel_size=4, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.kernel_size = kernel_size # Save this for the config
        self.norm = layers.LayerNormalization()
        
        self.conv1d = layers.Conv1D(
            filters=d_model, 
            kernel_size=kernel_size, 
            padding='causal', 
            activation='silu' 
        )
        self.ssm_scan = layers.GRU(d_model, return_sequences=True)
        self.out_proj = layers.Dense(d_model)

    def call(self, inputs):
        x = self.norm(inputs)
        x_conv = self.conv1d(x)
        x_ssm = self.ssm_scan(x_conv)
        out = self.out_proj(x_ssm)
        return inputs + out

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "kernel_size": self.kernel_size
        })
        return config

# --- 4. The Transformer Block (The Global Reasoner) ---
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        # Save variables for the config
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"), 
            layers.Dense(embed_dim)
        ])
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

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })
        return config

# --- 5. Assemble the Full Model ---
def build_graph_jamba_model(max_frames=100, num_joints=75, features=9, num_classes=2000):
    inputs = keras.Input(shape=(max_frames, num_joints, features))
    
    # Notice we don't pass adjacency_matrix=A anymore! The layer gets it directly.
    x = SpatialGraphConv(units=64)(inputs)
    
    x = layers.Reshape((max_frames, num_joints * 64))(x)
    x = layers.Dense(256, activation='relu')(x)
    
    x = MambaStyleBlock(d_model=256)(x)
    x = MambaStyleBlock(d_model=256)(x) 
    
    x = TransformerBlock(embed_dim=256, num_heads=4, ff_dim=512)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="Graph_Jamba_ASL")
    return model

# --- 6. Instantiate and Compile ---
NUM_CLASSES = len(le.classes_) # Dynamically set to match your LabelEncoder
model = build_graph_jamba_model(num_classes=NUM_CLASSES)

# Since we used LabelEncoder (integers) instead of One-Hot Encoding, 
# we MUST use sparse_categorical_crossentropy
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Define your save paths
MODEL_FILE = "best_graph_jamba.keras"
ENCODER_FILE = "label_encoder.pkl"

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    patience=40, # Graph models can take a bit longer to find deep minima, 80 is great here
    restore_best_weights=True, 
    verbose=1
)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=MODEL_FILE, 
    monitor='val_accuracy', 
    save_best_only=True, 
    verbose=1
)

# Optional but highly recommended: Reduce learning rate when learning stalls
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", 
    factor=0.5, 
    patience=20, 
    verbose=1
)

print("\n🚀 Starting Graph-Jamba Training...\n")

# Feed the generators directly into fit!
history = model.fit(
    train_gen,
    epochs=200,
    validation_data=val_gen,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# Save the encoder so you know which integer corresponds to which sign during inference
joblib.dump(le, ENCODER_FILE)
print(f"\n✅ Training Complete! Model saved as '{MODEL_FILE}' and Encoder saved as '{ENCODER_FILE}'.")

# --- 5. Plotting Training History ---
print("\n📊 Generating Training Graphs...")
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()