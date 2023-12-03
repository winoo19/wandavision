import cv2
from time import perf_counter, sleep


def capture_aruco(aruco_dict):
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    cap = cv2.VideoCapture(0)

    FRAME_RATE = 10

    while True:
        start = perf_counter()
        ret, frame = cap.read()
        corners, ids, rejected = aruco_detector.detectMarkers(frame)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            print(ids)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        end = perf_counter()
        time_taken = end - start
        time_to_sleep = 1 / FRAME_RATE - time_taken
        if time_to_sleep > 0:
            sleep(time_to_sleep)

    cap.release()
    cv2.destroyAllWindows()


def load_aruco(aruco_dict_index):
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_index)
    img = cv2.aruco.generateImageMarker(aruco_dict, 1, 200)

    cv2.imwrite(f"./aruco_markers/{aruco_dict_index}.jpg", img)


if __name__ == "__main__":
    aruco_dict_index = cv2.aruco.DICT_4X4_50
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_index)
    # load_aruco(aruco_dict_index)

    capture_aruco(aruco_dict)

    # help(cv2.aruco)
