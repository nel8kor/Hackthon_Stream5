#!/usr/bin/env py
# -*- coding: utf-8 -*-
# To view the frames and labels of the dataset
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Load the file
X = np.load('X_val.npy')
y = np.load('y_val.npy')

print("X shape:", X.shape)
print("y shape:", y.shape)
print("X dtype:", X.dtype)
print("y dtype:", y.dtype)

frame_count = X.shape[0]

print(f"Total frames: {frame_count}")


# for i in range(30):  # First 10 frames
plt.imshow(X[11000])
plt.title(f"Label: {np.argmax(y[11000])}")
plt.axis('off')
plt.show()
