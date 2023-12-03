import cv2
from picamera2 import Picamera2
import numpy as np


class Figure:
    def __init__(self, figure_type, color_name, color, n_vertices, color_threshold=30):
        self.figure_type = figure_type
        self.color_name = color_name
        self.color = np.array(color)
        self.n_vertices = n_vertices
        self.color_threshold = color_threshold

    def __eq__(self, other):
        if other is None:
            return False
        return (
            self.figure_type == other.figure_type
            and self.color_name == other.color_name
        )

    def __repr__(self):
        return f"Figure({self.figure_type}, {self.color_name}, {self.color}, {self.n_vertices})"

    def __str__(self):
        return f"{self.color} {self.figure_type}"

    def detect(self, img):
        """
        Returns True if the image contains the figure
        """

        # Remove alpha channel
        img_rgb = img[:, :, :3]

        # Blur without changing color
        blur = cv2.GaussianBlur(img_rgb, (5, 5), 0)

        # Use mask with euclidean distance
        mask = np.linalg.norm(blur - self.color, axis=-1) < self.color_threshold
        thresh = np.zeros_like(mask, dtype=np.uint8)
        thresh[mask] = 255

        # Erode to remove noise
        ks = 5
        kernel = np.ones((ks, ks), np.uint8)
        eroded = cv2.erode(thresh, kernel, iterations=1)

        # Dilate to recover original size
        dilated = cv2.dilate(eroded, kernel, iterations=1)

        # Find the contours
        contours, _ = cv2.findContours(
            dilated.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        # Match with all contours
        for contour in contours:
            corners = cv2.approxPolyDP(
                contour, 0.01 * cv2.arcLength(contour, True), True
            )

            # Match figure type
            if len(corners) == self.n_vertices:
                return True

        return False

    def plot_on_image(self, img, center, size):
        """
        Plots a figure on the image
        """

        # Get vertices
        vertices = []
        for i in range(self.n_vertices):
            angle = 2 * np.pi * i / self.n_vertices
            # Add up angle to even number of vertices
            angle += ((self.n_vertices + 1) % 2) * np.pi / self.n_vertices
            # First vertex is at the top
            angle -= np.pi / 2
            x = int(center[0] + size * np.cos(angle))
            y = int(center[1] + size * np.sin(angle))
            vertices.append((x, y))

        # Plot figure
        c = tuple(self.color.tolist())
        cv2.fillPoly(img, np.array([vertices]), c)


def get_picam2():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "XRGB8888"},
    )
    picam2.configure(config)
    return picam2


def enter_password(password: list, valid_figures: list):
    global picam2
    picam2.start()

    password_is_correct = None

    # Stable flanks
    prev_state = None  # Detected figure
    state = None  # Detected figure
    n_lags = 10
    lagged_states = []  # Detected figures

    sequence = []
    next_character = 0

    while cv2.waitKey(1) != ord("q") and password_is_correct is None:
        # Capture image
        im = picam2.capture_array()

        # Convert to RGB
        frame = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Detect figures
        figure_found = None
        for figure in valid_figures:
            if figure.detect(frame):
                figure_found = figure
                # print("Detected!", lagged_states)
                break

        lagged_states.append(figure_found)
        if len(lagged_states) > n_lags:
            lagged_states.pop(0)

        # Update state
        if len(lagged_states) == n_lags:
            f0 = lagged_states[0]
            # If all figures are the same
            if all(f0 == f for f in lagged_states):
                state = f0

        # Detect edges
        if prev_state != state:
            if state is not None:
                print(f"{state} detected")
                sequence.append(state)
                next_character += 1
            else:
                print("Figure disappeared")

        print("State:", state, "Prev:", prev_state)

        # Update previous state
        prev_state = state

        # Check password
        if len(sequence) == len(password):
            if sequence == password:
                print("Correct password")
                password_is_correct = True
            else:
                print("Incorrect password")
                password_is_correct = False

        # Plot sequence
        for i, figure in enumerate(sequence):
            figure.plot_on_image(frame, (50 + 70 * i, 50), 30)

        cv2.imshow("preview", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    # If password is correct, show camera with green border for 5 seconds
    seconds = 2
    color = (0, 255, 0) if password_is_correct else (0, 0, 255)
    start_time = cv2.getTickCount()
    while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < seconds:
        im = picam2.capture_array()
        frame = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2RGB)
        cv2.rectangle(frame, (0, 0), (640, 480), color, 10)
        cv2.imshow("preview", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()

    return password_is_correct


def authenticate():
    global password, valid_figures

    while not enter_password(password, valid_figures):
        print("Incorrect password. Try again.")


if __name__ == "__main__":
    picam2 = get_picam2()
    valid_figures = [
        Figure("triangle", "red", (0, 0, 178), 3),
        Figure("triangle", "yellow", (0, 255, 255), 3),
        Figure("quadrilateral", "green", (30, 125, 15), 4),
        Figure("pentagon", "blue", (249, 54, 0), 5),
    ]

    password = [
        valid_figures[2],
        valid_figures[0],
        valid_figures[3],
    ]
    authenticate()
