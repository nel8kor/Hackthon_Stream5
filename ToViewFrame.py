import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Load the file
X = np.load('X.npy')
y = np.load('y.npy')

print("X shape:", X.shape)
print("y shape:", y.shape)
print("X dtype:", X.dtype)
print("y dtype:", y.dtype)

frame_count = X.shape[0]

print(f"Total frames: {frame_count}")


for i in range(2):  # First 10 frames
    plt.imshow(X[i])
    plt.title(f"Label: {np.argmax(y[i])}")
    plt.axis('off')
    plt.show()
