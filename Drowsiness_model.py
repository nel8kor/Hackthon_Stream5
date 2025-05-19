#!/usr/bin/env py
# -*- coding: utf-8 -*-
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# Load the file
X = np.load('X.npy')
y = np.load('y.npy')

#X, y = shuffle(X, y, random_state=42)


print("X shape:", X.shape)
print("y shape:", y.shape)
print("X dtype:", X.dtype)
print("y dtype:", y.dtype)

#Training the model
model = Sequential([
    Input(shape=(240, 320, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(),
    BatchNormalization(),
    GlobalAveragePooling2D(),  # replaces Flatten safely
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.1)

model.save('model.h5')

#Training data plot
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_accuracy.png')
#plt.show()