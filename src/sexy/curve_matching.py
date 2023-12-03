import numpy as np
import pygame
import scipy.optimize
import matplotlib.pyplot as plt


def get_start_index(points1: np.ndarray, points2: np.ndarray, n_start_points: int = 5):
    """
    Find the closest points in points2 to the first point in points1

    Parameters
    ----------
    points1 : np.ndarray
        numpy array of shape (n_points, 2) containing the coordinates of the points
    points2 : np.ndarray
        numpy array of shape (n_points, 2) containing the coordinates of the points
    n_start_points : int
        Number of points to be used to find the closest points

    Returns
    -------
    int
        The index of the closest point in points2 to the first point in points1
    """

    # Make sure n_start_points is not larger than the number of points
    if n_start_points > points1.shape[0] or n_start_points > points2.shape[0]:
        n_start_points = min(points1.shape[0], points2.shape[0])

    first_points1 = points1[:n_start_points]

    # Make points2 length divisible by n_start_points, removing extra points
    n_points2 = points2.shape[0]
    n_points2 = n_points2 - n_points2 % n_start_points
    points2_reshaped = points2[:n_points2].reshape(-1, n_start_points, 2)

    # Find the closest points
    min_mean_dist = np.inf
    min_index = 0
    for i in range(points2_reshaped.shape[0]):
        mean_dist = np.mean(np.linalg.norm(points2_reshaped[i] - first_points1, axis=1))
        if mean_dist < min_mean_dist:
            min_mean_dist = mean_dist
            min_index = i

    return min_index * n_start_points


def frdist_invariant(points1: np.ndarray, points2: np.ndarray):
    """
    Invariant to direction of points

    Parameters
    ----------
    points1 : np.ndarray
        numpy array of shape (n_points, 2) containing the coordinates of the points
    points2 : np.ndarray
        numpy array of shape (n_points, 2) containing the coordinates of the points

    Returns
    -------
    float
        The Fréchet distance between the two curves
    """
    # get closest points
    start_index = get_start_index(points1, points2)
    points2 = np.roll(points2, -start_index, axis=0)
    frdist1 = frdist(points1, points2)

    points2 = points2[::-1]
    start_index = get_start_index(points1, points2)
    points2 = np.roll(points2, -start_index, axis=0)
    frdist2 = frdist(points1, points2)

    return min(frdist1, frdist2)


def frdist(points1: np.ndarray, points2: np.ndarray):
    """
    The Fréchet distance is the smallest of the maximum pairwise distances.

    Parameters
    ----------
    points1 : np.ndarray
        numpy array of shape (n_points, 2) containing the coordinates of the points
    points2 : np.ndarray
        numpy array of shape (n_points, 2) containing the coordinates of the points

    Returns
    -------
    float
        The Fréchet distance between the two curves
    """

    # Calculate the distance matrix
    D = np.linalg.norm(points1[:, None] - points2, axis=2)

    # Initialize the matrix
    M = np.zeros_like(D)

    # Fill the first row
    M[0, 0] = D[0, 0]
    for i in range(1, M.shape[1]):
        M[0, i] = max(M[0, i - 1], D[0, i])

    # Fill the first column
    for i in range(1, M.shape[0]):
        M[i, 0] = max(M[i - 1, 0], D[i, 0])

    # Fill the rest of the matrix
    for i in range(1, M.shape[0]):
        for j in range(1, M.shape[1]):
            M[i, j] = max(min(M[i - 1, j], M[i - 1, j - 1], M[i, j - 1]), D[i, j])

    return M[-1, -1]


def normalize_points(points):
    """
    Normalizes a set of points

    Parameters
    ----------
    points : np.ndarray
        numpy array of shape (n_points, 2) containing the coordinates of the points

    Returns
    -------
    np.ndarray
        The normalized points
    """
    points = points - np.mean(points, axis=0)
    points = points / np.max(np.abs(points), axis=0)

    return points


def create_curve(threshold=5.0):
    """
    Opens a pygame window and waits for the user to draw a curve with the mouse.
    Only add point if distance between last point and current point is larger than threshold.

    Parameters
    ----------
    threshold : float
        The minimum distance between the last point and the new point for it to count as a distinct point

    Returns
    -------
    np.ndarray
        The normalized points
    """

    # Initialize pygame
    pygame.init()
    WIDTH = 640
    HEIGHT = 480
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Draw a curve with the mouse")
    screen.fill((255, 255, 255))
    pygame.display.flip()

    # Initialize variables
    points = []
    n = 0
    running = True

    # Main loop
    while running:
        # Events
        for event in pygame.event.get():
            # Quit
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                points = []
            # key pressed
            elif event.type == pygame.KEYDOWN:
                if event.key == 8:
                    points = []
                elif event.key == 13:
                    running = False

        pygame.draw.rect(screen, (255, 255, 255), (0, 0, WIDTH, HEIGHT))

        # If mouse is pressed
        if pygame.mouse.get_pressed()[0]:
            # Get mouse position
            pos = pygame.mouse.get_pos()
            pos = (pos[0], HEIGHT - pos[1])
            # If points is empty, add point
            if len(points) == 0:
                points.append(pos)
                n += 1
            # If distance between last point and current point is larger than threshold, add point
            elif np.linalg.norm(np.array(pos) - np.array(points[-1])) > threshold:
                points.append(pos)
                n += 1

        # Draw points
        for point in points:
            pygame.draw.circle(screen, (0, 0, 0), (point[0], HEIGHT - point[1]), 3)

        # Update screen
        pygame.display.flip()

    # Close pygame
    pygame.quit()

    # Convert points to numpy array
    points = np.array(points).reshape(-1, 2)

    # Normalize points
    return normalize_points(points)


class GenericCurve:
    """
    Class for storing a curve pattern

    Attributes
    ----------
    points : np.ndarray
        numpy array of shape (n_points, 2) containing the coordinates of the points

    Methods
    -------
    save_pattern(filename)
        Saves the pattern to a file
    load_pattern(filename)
        Loads the pattern from a file
    """

    def __init__(self, filename=None):
        self.points = None

        if filename is not None:
            self.load_pattern(filename)
        else:
            self.points = create_curve()

    def save_pattern(self, filename):
        np.save(filename, self.points)

    def load_pattern(self, filename):
        self.points = np.load(filename)


if __name__ == "__main__":
    ##############################
    ###     CREATE_PATTERN    ####
    # pattern = GenericCurve()
    # pattern.save_pattern("patterns/thunder0.npy")
    ##############################

    ##############################
    ###     TEST PATTERN      ####
    pattern = GenericCurve("patterns/circle0.npy")
    points = create_curve()

    similarity = frdist_invariant(pattern.points, points)

    print("Fréchet distance:", similarity)

    # Plot pattern and curve
    plt.plot(pattern.points[:, 0], pattern.points[:, 1], label="pattern")
    plt.scatter(
        points[:, 0],
        points[:, 1],
        label="curve",
        c="g" if similarity < 0.7 else "r",
    )
    plt.gca().set_aspect("equal", adjustable="box")

    # Limit the plot to the area around the curve
    plt.xlim(np.min(points[:, 0]) - 0.1, np.max(points[:, 0]) + 0.1)
    plt.ylim(np.min(points[:, 1]) - 0.1, np.max(points[:, 1]) + 0.1)
    plt.legend()
    plt.show()
    ##############################
