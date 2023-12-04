import threading
import cv2
from detect_pattern.detect import get_picam2
import numpy as np
from gestures_chat.curve_matching import frdist_invariant
from pydub import AudioSegment
from pydub.playback import play


class PatternInterpreter:
    """
    Interprets patterns drawn by the user

    Attributes
    ----------
    picam2 : Picamera2
        The picamera2 object
    aruco_detector : cv2.aruco.ArucoDetector
        The aruco detector
    n_lags : int
        The number of lags to use for smoothing
    points : list
        The points of the current pattern
    patterns : dict
        The patterns database

    Methods
    -------
    get_aruco_detector()
        Constructs an aruco detector
    get_picam2()
        Constructs a picam2 object
    add_point(point, threshold_min=5, threshold_max=70)
        Adds a points to the curve if the last point is at least threshold
        pixels away from the new point
    match_pattern()
        Matches the current pattern to the pattern database
    chat(aruco_target_id=0)
        Starts a conversation with the user
    """

    def __init__(self, picam=None):
        if picam is None:
            self.picam2 = get_picam2()
        else:
            self.picam2 = picam
        self.aruco_detector = self.get_aruco_detector()

        self.n_lags = 15

        self.points = []

        self.patterns = {
            "circle": {
                "points": np.load("trajectories/circle0.npy"),
                "threshold": 0.45,
                "message": "Hello!",
                "sound": AudioSegment.from_wav("audio/dar_cera_pulir_cera.wav"),
            },
            "heart": {
                "points": np.load("trajectories/heart0.npy"),
                "threshold": 0.45,
                "message": "I love you too!",
                "sound": AudioSegment.from_wav("audio/iloveu2.wav"),
            },
            "infinity": {
                "points": np.load("trajectories/inf0.npy"),
                "threshold": 0.45,
                "message": "Forever is a long time...",
                "sound": AudioSegment.from_wav("audio/forever_long_time.wav"),
            },
            "thunder": {
                "points": np.load("trajectories/thunder0.npy"),
                "threshold": 0.35,
                "message": "I'm Thor! God of Thunder!",
                "sound": AudioSegment.from_wav("audio/thor.wav"),
            },
        }

    @staticmethod
    def get_aruco_detector():
        """
        Constructs an aruco detector

        Returns
        -------
        cv2.aruco.ArucoDetector
        """
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_detector = cv2.aruco.ArucoDetector(
            aruco_dict, cv2.aruco.DetectorParameters()
        )
        return aruco_detector

    def add_point(self, point, threshold_min=5, threshold_max=70):
        """
        Adds a points to the curve if the last point is at least threshold
        pixels away from the new point

        Parameters
        ----------
        point : np.array
            The point to add
        threshold_min : int
            The minimum distance between the last point and the new point
            for it to count as a distinct point
        threshold_max : int
            The maximum distance between the last point and the new point
            for it not to count as an outlier

        Returns
        -------
        None
        """
        if len(self.points) == 0:
            self.points.append(point)
            return

        last_point = self.points[-1]
        if threshold_max > np.linalg.norm(point - last_point) > threshold_min:
            self.points.append(point)

    def match_pattern(self):
        """
        Matches the current pattern to the pattern database

        Returns
        -------
        str
            The name of the closest pattern
        """
        # Normalize points and invert y axis
        pattern = np.array(self.points)
        pattern[:, 0] = 640 - pattern[:, 0]
        pattern[:, 1] = 480 - pattern[:, 1]

        pattern -= np.mean(pattern, axis=0)
        pattern /= np.max(np.abs(pattern), axis=0)

        # Find the closest pattern
        min_dist = np.inf
        closest_pattern = None
        for pattern_name, pattern_data in self.patterns.items():
            pattern_points = pattern_data["points"]

            dist = frdist_invariant(pattern, pattern_points)
            if dist < min_dist:
                min_dist = dist
                closest_pattern = pattern_name

        print("Closest pattern:", closest_pattern)
        print("Distance to closest pattern:", min_dist)

        return (
            closest_pattern
            if closest_pattern is not None
            and min_dist < self.patterns[closest_pattern]["threshold"]
            else None
        )

    def chat(self, aruco_target_id: int = 0):
        """
        Starts a conversation with the user

        Parameters
        ----------
        aruco_target_id : int
            The id of the aruco marker to track

        Returns
        -------
        None
        """
        cv2.startWindowThread()
        self.picam2.start()

        previous_state = 0
        state = 0  # 0 No aruco detected, 1 Aruco detected
        lagged_states = [0] * self.n_lags

        while True:
            im = self.picam2.capture_array()

            # Convert array to cv2 image without altering colors
            frame = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            corners, ids, _ = self.aruco_detector.detectMarkers(frame)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                new_state = 1 if aruco_target_id in ids else 0

                if state == 1:
                    # Get center of aruco
                    center = np.mean(corners[0][0], axis=0)
                    self.add_point(center)
            else:
                new_state = 0

            # Update lagged states
            lagged_states.pop(0)
            lagged_states.append(new_state)

            # Update state
            state = 1 if all(lagged_states) else state
            state = 0 if not any(lagged_states) else state

            # Detect edges
            if previous_state != state:
                if state == 1:
                    print("Aruco detected")
                else:
                    print("Aruco disappeared")
                    # Match pattern
                    pattern = self.match_pattern()
                    if pattern is not None:
                        sound = self.patterns[pattern]["sound"]
                        threading.Thread(
                            target=play, args=(sound,), daemon=True
                        ).start()
                        print(self.patterns[pattern]["message"])
                    else:
                        print("I don't know what you mean!!!")

                    self.points = []

            # Update previous state
            previous_state = state

            # Plot points
            # for point in self.points:
            #     cv2.circle(frame, tuple(point.astype(int)), 5, (0, 0, 255), -1)

            # Plot smooth lne of points
            if len(self.points) > 1:
                # Linear fade out of size. Thicker the newer the point.
                for i in range(len(self.points) - 1):
                    size = 4 * (i / len(self.points))
                    cv2.line(
                        frame,
                        tuple(self.points[i].astype(int)),
                        tuple(self.points[i + 1].astype(int)),
                        (0, 0, 255),
                        int(size) + 2,
                    )

            # mirror
            cv2.imshow("frame", cv2.flip(frame, 1))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    pattern_interpreter = PatternInterpreter()
    pattern_interpreter.chat(aruco_target_id=0)
