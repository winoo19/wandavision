import cv2
from picamera2 import Picamera2
import numpy as np


class Figure:
    def __init__(self, figure_type, color_name, color, n_vertices):
        self.figure_type = figure_type
        self.color_name = color_name
        self.color = np.array(color, dtype=np.uint8)
        self.n_vertices = n_vertices

    def __eq__(self, other):
        return self.figure_type == other.figure_type and self.color == other.color

    def __repr__(self):
        return f"{self.color} {self.figure_type}"

    def __str__(self):
        return f"{self.color} {self.figure_type}"

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


def get_colour_name(requested_colour):
    """
    Only primary colors
    """
    colors = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }

    min_colours = {}
    for key, value in colors.items():
        r_c, g_c, b_c = value
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = key

    return min_colours[min(min_colours.keys())]


def get_picam2():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "XRGB8888"},
    )
    picam2.configure(config)
    return picam2


def detect_figure(img, target_figure, color_threshold=50):
    """
    Returns True if the image contains the target figure with the target color
    """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image
    _, thresh = cv2.threshold(blur, 95, 255, cv2.THRESH_BINARY_INV)

    # Find the contours
    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    # Get all corners of each contour found
    figures = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        figures.append(approx)

    for corners in figures:
        # Get type of figure and color
        if len(corners) == 3:
            figure_type = "triangle"
        elif len(corners) == 4:
            figure_type = "quadrilateral"
        elif len(corners) == 5:
            figure_type = "pentagon"
        else:
            figure_type = "unknown"

        if figure_type == "unknown":
            continue

        # Get color of the center of the figure
        # Center is the mean of all corners
        center = np.mean(corners, axis=0)
        center = np.array(center, dtype=np.int32)
        color = img[center[0][1], center[0][0]]

        color_name = get_colour_name(color)

        # print(f"Detected {color_name} {figure_type}")
        # print(np.linalg.norm(color - target_figure.color))

        if (
            figure_type == target_figure.figure_type
            and np.linalg.norm(color - target_figure.color) < color_threshold
        ):
            return True

    return False


def enter_password(password: list, valid_figures: list):
    print("Password:")
    for figure in password:
        print(figure)

    picam2 = get_picam2()
    picam2.start()

    sequence = []
    next_character = 0

    while cv2.waitKey(1) != ord("q"):
        im = picam2.capture_array()

        # Convert array to cv2 image without altering colors
        temp = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)

        # Detect figures
        if detect_figure(frame, password[next_character]):
            sequence.append(password[next_character])
            next_character += 1

        if len(sequence) == len(password):
            if sequence == password:
                print("Correct password")
            else:
                print("Incorrect password")
            sequence = []
            next_character = 0

        # Plot sequence
        for i, figure in enumerate(sequence):
            figure.plot_on_image(frame, (50 + 100 * i, 50), 30)

        cv2.imshow("preview", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    password = [
        Figure("quadrilateral", "green", (92, 130, 24), 4),
        Figure("triangle", "red", (0, 0, 178), 3),
        Figure("pentagon", "blue", (249, 54, 0), 5),
    ]

    valid_figures = [
        Figure("triangle", "red", (0, 0, 178), 3),
        Figure("quadrilateral", "green", (92, 130, 24), 4),
        Figure("pentagon", "blue", (249, 54, 0), 5),
    ]

    enter_password(password, valid_figures)
