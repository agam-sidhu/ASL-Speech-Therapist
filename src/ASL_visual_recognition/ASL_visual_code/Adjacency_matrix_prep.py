import numpy as np
import tensorflow as tf

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

# Generate our rulebook
adjacency_matrix = build_adjacency_matrix()
print("✅ Adjacency Matrix Built! Shape:", adjacency_matrix.shape)