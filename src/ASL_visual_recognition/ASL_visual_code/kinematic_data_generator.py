import numpy as np
import keras
from keras.utils import PyDataset
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

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