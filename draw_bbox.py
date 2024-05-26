import cv2
import numpy as np

# Create a black image of size 1024x1024 with 3 channels (BGR)
image = np.zeros((1024, 1024, 3), dtype=np.uint8)

# Define the bounding box values
bbox = [932, 201, 8, 12]

# Calculate the bottom-right corner from the top-left corner and width, height
top_left = (bbox[0], bbox[1])
bottom_right = (bbox[0] + bbox[2], bbox[1] + bbox[3])

# Draw a blue rectangle on the image
# cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255), -1)
cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), -1)

# Save or display the image to verify the result
# cv2.imwrite('output.png', image)
# To display the image, you can use the following line (uncomment if you want to see it in a window):
cv2.imshow('Image with Blue Rectangle', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
