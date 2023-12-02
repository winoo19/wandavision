from picamera2 import Picamera2, Preview
import time
import sys
import os

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 selfie.py <filename>")
        sys.exit(1)

    path_name = sys.argv[1]
    path = os.path.dirname(path_name)
    if path and not os.path.exists(path):
        os.makedirs(path)

    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "XRGB8888"}
    )
    picam2.configure(camera_config)
    picam2.start_preview()

    picam2.start()
    time.sleep(2)
    picam2.capture_file(path_name)
