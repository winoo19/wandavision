import cv2


def capture_aruco(aruco_dict):
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        corners, ids, rejected = aruco_detector.detectMarkers(frame)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            print(ids)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

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
