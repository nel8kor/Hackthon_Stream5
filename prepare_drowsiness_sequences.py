"""
This script processes validation data for a drowsiness detection task by filtering, relabeling, and preparing sequential data.
Workflow:
1. Loads image data (`X_val.npy`) and corresponding labels (`y_val.npy`).
2. Filters samples to include only 'distracted' (label 1) and 'drowsy' (label 2) classes.
3. Relabels 'distracted' as 0 and 'drowsy' as 1 for binary classification.
4. One-hot encodes the binary labels.
5. Constructs sequences of 10 consecutive frames for temporal modeling.
6. Assigns the label of the middle frame in each sequence as the sequence label.
7. Saves the resulting sequences and labels as NumPy arrays (`X_seq.npy`, `y_seq.npy`).
Outputs:
- `X_seq.npy`: Array of shape (num_sequences, 10, height, width, channels) containing frame sequences.
- `y_seq.npy`: Array of shape (num_sequences, 2) containing one-hot encoded labels for each sequence.
"""
import numpy as np
from tensorflow.keras.utils import to_categorical # type: ignore

# Load data
X = np.load("X_val.npy")  # shape: (num_samples, height, width, channels)
y = np.load("y_val.npy")  # shape: (num_samples,), values: 0, 1, 2

# Filter for distracted (1) and drowsy (2)
mask = (y == 1) | (y == 2)
X_filtered = X[mask]
y_filtered = y[mask]

# Relabel to 0 and 1: 0 = distracted, 1 = drowsy
y_binary = np.where(y_filtered == 1, 0, 1)

# One-hot encode: 0 → [1,0] (distracted), 1 → [0,1] (drowsy)
y_onehot = to_categorical(y_binary, num_classes=2)

print(f"X shape: {X_filtered.shape}, y shape: {y_onehot.shape}")

# Create sequences of 10 frames
sequence_length = 10
num_samples = len(X_filtered) - sequence_length + 1

X_seq = np.array([X_filtered[i:i+sequence_length] for i in range(num_samples)])
y_seq_indices = np.array([y_binary[i+sequence_length//2] for i in range(num_samples)])
y_seq_onehot = to_categorical(y_seq_indices, num_classes=2)

np.save("X_seq.npy", X_seq)
np.save("y_seq.npy", y_seq_onehot)

print(f"X_seq shape: {X_seq.shape}, y_seq shape: {y_onehot.shape}")