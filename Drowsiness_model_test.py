#!/usr/bin/env py

from sklearn.metrics import classification_report, confusion_matrix # type: ignore
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import seaborn as sns
import matplotlib.pyplot as plt

#load the test data
X_val = np.load('X_Val.npy')
y_val = np.load('y_Val.npy')

print(f"Y shape:{y_val.shape}")


#load the model
model = load_model('model.h5')
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

loss, accuracy = model.evaluate(X_val, y_val, batch_size=16)
print(f"Validation Accuracy: {accuracy*100:.2f}%")

print('\n*********************\n')

print(confusion_matrix(y_true, y_pred_classes))
print(classification_report(y_true, y_pred_classes, labels=[0, 1, 2], target_names=["awake","distracted","drowsy"]))

cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['awake','distracted','drowsy'], yticklabels=['awake','distracted','drowsy'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
print('\n..............saving confusion matrix as png..............\n')
plt.savefig('confusion_matrix.png')
print('\n..............confusion matrix saved as png..............\n')
plt.show()

