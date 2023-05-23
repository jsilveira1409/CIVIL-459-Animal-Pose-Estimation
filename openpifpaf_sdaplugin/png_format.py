import cv2
import numpy as np
from sda import SDA


# Load the image and alpha channel
image = cv2.imread('image.png', cv2.IMREAD_UNCHANGED)
alpha = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# Normalize alpha channel to range [0, 1]
alpha_normalized = alpha.astype(float) / 255

# Split image into color channels
b, g, r = cv2.split(image)

# Apply transparency by multiplying color channels with normalized alpha
b = (b * alpha_normalized).astype(np.uint8)
g = (g * alpha_normalized).astype(np.uint8)
r = (r * alpha_normalized).astype(np.uint8)

# Merge color channels back together
result = cv2.merge((b, g, r))

# Save or display the resulting image
cv2.imwrite('result.png', result)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
