#!/usr/bin/env py
# -*- coding: utf-8 -*-
"""
Builds and compiles a 3D Convolutional Neural Network (3D CNN) model for video classification tasks.
The model consists of three convolutional blocks with Conv3D, MaxPooling3D, and BatchNormalization layers,
followed by a dense classification head. The network is designed to process video clips or sequences of images.
Args:
    input_shape (tuple): Shape of the input data in the format (frames, height, width, channels).
                            Default is (10, 240, 320, 3).
    num_classes (int): Number of output classes for classification. Default is 2.
Returns:
    keras.models.Sequential: A compiled Keras Sequential model ready for training.
"""
import numpy as np
from keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import matplotlib.pyplot as plt

def build_3d_cnn(input_shape=(10, 240, 320, 3), num_classes=2):
    model = Sequential()

    # Block 1
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(BatchNormalization())

    # Block 2
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(BatchNormalization())

    # Block 3
    model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))  # reduce temporal dimension
    model.add(BatchNormalization())

    # Classification Head
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

    return model


# Load data
X = np.load("X.npy")  # shape: (num_samples, height, width, channels)
y = np.load("y.npy")  # shape: (num_samples,), values: 0, 1, 2

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


print(f"X_seq shape: {X_seq.shape}, y_seq shape: {y_seq_onehot.shape}")

model = build_3d_cnn(input_shape=(10, 120, 160, 3), num_classes=2)

history = model.fit(
    X_seq, y_seq_onehot,
    validation_split=0.2,
    epochs=10,
    batch_size=2,
    shuffle=True
)

model.save('model.h5')

#Training data plot
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_accuracy.png')
#plt.show()