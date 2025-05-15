import numpy as np
import cv2
import time

# Load pre-saved frame data
frames = np.load('X.npy')  # shape: (num_frames, height, width, channels)

fps = 15  # Adjust as needed
delay = int(1000 / fps)

for frame in frames:
    frame_bgr = (frame * 255).astype('uint8')  # If normalized to 0-1
    cv2.imshow("Simulated Video", frame_bgr)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()