import cv2
import numpy as np
import webcolors


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def closest_primary_colour(requested_colour):
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


def get_colour_name(requested_colour, primary=True):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        if primary:
            closest_name = closest_primary_colour(requested_colour)
        else:
            closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


def detect(image_path: str):
    """
    Detect only one of triangles, squares, pentagons.
    Get the color of the figure.
    """

    # Read the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Show blurred image for 1 second
    cv2.imshow("blur", blur)
    cv2.waitKey(1000)

    # Threshold the image
    ret, thresh = cv2.threshold(blur, 95, 255, cv2.THRESH_BINARY_INV)

    # Show thresholded image for 1 second
    cv2.imshow("thresh", thresh)
    cv2.waitKey(1000)

    # Find the contours
    contours, hierarchy = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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

        color_name = get_colour_name(color[::-1])[1]

        print("Found figure of type {} with color {}".format(figure_type, color_name))

    # Display the image
    cv2.imshow("img", img)
    cv2.waitKey(30000)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect("images/patterns/pentagon_green.jpg")
