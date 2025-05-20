import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from collections import deque
import os
from datetime import datetime

# Load the trained model
model = load_model('model.h5')

# Create output folder for saved frames
output_dir = "classified_frames"
os.makedirs(output_dir, exist_ok=True)
save_dir = "saved_drowsy_npy"
os.makedirs(save_dir, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Sequence buffer
sequence_length = 10
frame_buffer = deque(maxlen=sequence_length)

X, y = [], []

print("⏳ Starting real-time detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to model input
    resized_frame = cv2.resize(frame, (120, 160))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Normalize if needed (adjust if your training used different normalization)
    normalized_frame = rgb_frame.astype(np.float32) / 255.0

    # Add frame to buffer
    frame_buffer.append(normalized_frame)
    
    # If we have enough frames for a sequence
    if len(frame_buffer) == sequence_length:
        input_sequence = np.array(frame_buffer).reshape(1, sequence_length, 120, 160, 3)
        prediction = model.predict(input_sequence, verbose=0)
        pred_class = np.argmax(prediction)

        # Draw prediction on live frame
        label = "Drowsy" if pred_class == 1 else "Distracted"
        color = (0, 0, 255) if pred_class == 1 else (0, 255, 255)
        cv2.putText(frame, f"Detected: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Save all frames if drowsy
        if pred_class == 1:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            for i, f in enumerate(frame_buffer):
                save_path = os.path.join(output_dir, f"drowsy_{timestamp}_{i}.jpg")
                bgr_frame = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, bgr_frame)
            # Convert list of frames to numpy array
            drowsy_sequence = np.array(frame_buffer)  # shape: (10, height, width, 3)
            labels=np.array(pred_class)
            X.append(drowsy_sequence)
            y.append(labels) 
            #np.save(save_path, data)
            print(f"🟥 Drowsy state detected — saved 10 frames at {timestamp}")
            # print(f"🟥 Drowsy state detected — saved sequence and label to {save_path}")

    # Show the live feed
    cv2.imshow("Real-Time Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
X_val = np.array(X)
y_val = np.array(y)
np.save(os.path.join(save_dir, "X_val.npy"), X_val)
np.save(os.path.join(save_dir, "y_val.npy"), y_val)
print(f"🟩 Saved {len(X_val)} frames and labels to {save_dir}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
