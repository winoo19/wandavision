import cv2
import numpy as np

img = cv2.imread("images/patterns/green_square.jpg")

# Plot the image
cv2.imshow("Image", img)

# Apply a blur to reduce noise
blur = cv2.GaussianBlur(img, (5, 5), 0)

cv2.imshow("Blur", blur)

# segment by color
color = np.array([92, 130, 24])
thresh = cv2.inRange(blur, color - 30, color + 30)

cv2.imshow("Thresh", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
