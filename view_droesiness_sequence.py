#!/usr/bin/env py
# -*- coding: utf-8 -*-
# To view the frames and labels of the dataset
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from keras.utils import to_categorical # type: ignore

# Load the file
X = np.load('saved_drowsy_npy\X_val.npy')
y = np.load('saved_drowsy_npy\y_val.npy')

#y = to_categorical(y, num_classes=3)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("X dtype:", X.dtype)
print("y dtype:", y.dtype)

frame_count = X.shape[0]

print(f"Total frames: {frame_count}")

# sample_range = (371, 471)

# for i in range(sample_range[0], sample_range[1]):  # First 30 frames
# 	plt.imshow(X[i])
# 	plt.title(f"Label: {np.argmax(y[i])}")
# 	plt.axis('off')
# 	plt.show()
sample_range = (0, 1)  # reduced range for quick testing

for i in range(sample_range[0], sample_range[1]):
    for t in range(10):  # show all 10 frames in the sequence
        plt.imshow(X[i][t].astype(np.uint8))  # ensure data is displayable
        plt.title(f"Sequence {i}, Frame {t}, Label: {y[i]}")
        plt.axis('off')
        plt.show()

