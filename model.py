import cv2
import numpy as np

# Step 1: Read the Image
image = cv2.imread('training/016z050pf.jpg')

#cropping to only include green background
image = image[320:, 750:1800]

# Step 2: Convert to HSV Color Space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Step 3: Create a Mask for the Green Background
# originally 35,85
lower_green = np.array([35, 40, 40])   # Lower bound for green in HSV
upper_green = np.array([85, 255, 255]) # Upper bound for green in HSV
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

# Step 4: Invert the Mask
mask_inv = cv2.bitwise_not(green_mask)

# Step 5: Apply the Mask to Extract the Subject
subject = cv2.bitwise_and(image, image, mask=mask_inv)

# Step 6: Optionally, Refine the Mask (example of erosion and dilation)
kernel = np.ones((5, 5), np.uint8)
# mask_inv = cv2.erode(mask_inv, kernel, iterations=1)
# mask_inv = cv2.dilate(mask_inv, kernel, iterations=1)
# # Optional: Apply Gaussian Blur to smooth edges
# mask_inv = cv2.GaussianBlur(mask_inv, (5, 5), 0)
# subject = cv2.bitwise_and(image, image, mask=mask_inv)

# subject = cv2.erode(subject, kernel, iterations=3)
# subject = cv2.dilate(subject, kernel, iterations=3)

imS = cv2.resize(subject, (960, 540))

# Convert the image to grayscale using the V (value) channel
value_channel = imS[:, :, 2]

# Create a binary image where non-black pixels (value > 0) are turned white
_, binary_image = cv2.threshold(value_channel, 1, 255, cv2.THRESH_BINARY)

cv2.imwrite('thresholded.jpg', binary_image)
# Resize image
# Save or display the result
cv2.imshow('Extracted Subject', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
