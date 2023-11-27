import cv2
from picamera2 import Picamera2

cv2.startWindowThread()

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "XRGB8888"},
)
picam2.configure(config)
picam2.start()

while True:
    im = picam2.capture_array()

    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    cv2.imshow("preview", grey)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
