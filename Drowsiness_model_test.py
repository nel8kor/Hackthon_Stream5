#!/usr/bin/env py

from sklearn.metrics import classification_report, confusion_matrix # type: ignore
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

#load the test data
X_val = np.load('X_Val.npy')
y_val = np.load('y_Val.npy')


#load the model
model = load_model('model.h5')
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {accuracy*100:.2f}%")

print('\n*********************\n')

print(confusion_matrix(y_true, y_pred_classes))
print(classification_report(y_true, y_pred_classes, target_names=["awake","drowsy"]))
