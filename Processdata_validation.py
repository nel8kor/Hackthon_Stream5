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