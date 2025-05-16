import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

model = load_model('model.h5')
cap = cv2.VideoCapture(0)

#sequence = []
frame_size = (64, 64)  # Must match model input
THRESHOLD = 10  # Frames

counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # face = cv2.resize(frame, frame_size)
    try:
        face = cv2.resize(frame, frame_size)
    except Exception as e:
        print(f"Resize failed: {e}")
        continue
    face = face / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)


    pred = model.predict(face)[0]
    label = np.argmax(pred)

    if label == 1:  # 1 = drowsy
        counter += 1
    else:
        counter = 0
        
    labels = ['awake', 'drowsy']

    if counter > THRESHOLD:
        cv2.putText(frame, labels[label], (30, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()