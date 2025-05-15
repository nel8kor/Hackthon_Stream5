#!/usr/bin/env py
# -*- coding: utf-8 -*-
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Load the file
X = np.load('X.npy')
y = np.load('y.npy')

print("X shape:", X.shape)
print("y shape:", y.shape)
print("X dtype:", X.dtype)
print("y dtype:", y.dtype)

#Training the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    GlobalAveragePooling2D(),  # replaces Flatten safely
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history=model.fit(X, y, epochs=50, batch_size=32, validation_split=0.1)
model.save('model.h5')

#Training data plot
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()