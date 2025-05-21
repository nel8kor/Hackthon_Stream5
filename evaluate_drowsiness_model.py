#!/usr/bin/env py
"""
This script evaluates a trained drowsiness detection model on a test dataset.
Workflow:
1. Loads test data (`X_seq.npy` for features and `y_seq.npy` for labels).
2. Loads a pre-trained Keras model (`model.h5`).
3. Evaluates the model on the test data and prints the validation accuracy.
4. Generates class predictions for the test data.
5. Handles both categorical and integer label formats for ground truth.
6. Prints a classification report (precision, recall, f1-score, support) for each class ("Distracted", "Drowsy").
7. Saves the classification report to a text file (`classification_report.txt`).
8. Computes and visualizes the confusion matrix as a heatmap, saving it as `confusion_matrix.png`.
Dependencies:
- numpy
- matplotlib
- seaborn
- keras
- scikit-learn
Intended for use in evaluating sequence-based drowsiness detection models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model # type: ignore
from keras.utils import to_categorical # type: ignore
from sklearn.metrics import classification_report, confusion_matrix

# Load test data
X_seq = np.load('saved_drowsy_npy\X_seq.npy')
y_seq = np.load('saved_drowsy_npy\y_seq.npy')

print(f"X_Seq shape: {X_seq.shape}")
print(f"Y_seq shape: {y_seq.shape}")

# Load model
model = load_model('model.h5')
loss, accuracy = model.evaluate(X_seq, y_seq, verbose=1)
print(f"Validation Accuracy: {accuracy*100:.2f}%")

# Predict
y_pred_probs = model.predict(X_seq)
y_pred_classes = np.argmax(y_pred_probs, axis=1)  # ✅ Correct way to get class predictions

if len(y_seq.shape) == 2 and y_seq.shape[1] == 2:
    y_seq_classes = np.argmax(y_seq, axis=1)
else:
    y_seq_classes = y_seq


# Confirm shape
print(f"y_seq_classes shape: {y_seq_classes.shape}, y_pred_classes shape: {y_pred_classes.shape}")


# Classification report
print(classification_report(y_seq_classes, y_pred_classes, target_names=["Distracted", "Drowsy"]))

#Save the classification report to a text file
with open("classification_report.txt", "w") as f:
    f.write(classification_report(y_seq_classes, y_pred_classes, target_names=["Distracted", "Drowsy"]))    

# Confusion matrix
cm = confusion_matrix(y_seq_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Distracted", "Drowsy"],
            yticklabels=["Distracted", "Drowsy"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

