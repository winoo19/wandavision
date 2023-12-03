# Script to get the color of a pixel on the screen when clicking on it
import numpy as np
import pyautogui
import time

for i in range(3):
    print(3 - i)
    time.sleep(1)

# Get screenshot
img = pyautogui.screenshot()

# Get mouse position
pos = pyautogui.position()

# Convert to numpy array
img = np.array(img)

# Convert RGB to BGR
img = img[:, :, ::-1].copy()

# Get color of pixel
color = img[pos[1], pos[0]]

print(color)
