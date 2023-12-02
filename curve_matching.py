import numpy as np
import pygame
import scipy.optimize
import matplotlib.pyplot as plt


def frdist(points1: np.ndarray, points2: np.ndarray):
    """
    The Fréchet distance is the smallest of the maximum pairwise distances.
    :param points1: numpy array of shape (n_points, 2) containing the coordinates of the points
    :param points2: numpy array of shape (n_points, 2) containing the coordinates of the points
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


class Curve:
    def __init__(self):
        self.n_params = None
        self.params = None
        self.mse = None

    def func(self, n_points):
        raise NotImplementedError

    def error_func(self, points, params):
        raise NotImplementedError

    def improve_mse(self, mse):
        """
        Improves the mse, increasing it if the parameters are not valid for example
        :param mse: current mse
        :return: new mse
        """
        return mse

    def fit_curve(self, points):
        """
        Fits the circle to a set of points
        :param points: numpy array of shape (n_points, 2) containing the coordinates of the points
        :return: center and radius of the fitted circle
        """

        result = scipy.optimize.least_squares(
            lambda params: self.error_func(points, params), np.zeros(self.n_params)
        )

        self.params = result.x
        self.mse = self.improve_mse(result.cost)

        return self.params, self.mse


class CircleCurve(Curve):
    def __init__(self):
        self.n_params = 3

    def func(self, n_points):
        """
        Creates a circle with n_points points
        :param n_points: number of points to be drawn
        :return: numpy array of shape (n_points, 2) containing the coordinates of the points
        """
        if self.params is None:
            raise Exception("No parameters found. Please fit the curve first.")
        t = np.linspace(0, 2 * np.pi, n_points)
        return np.array(
            [
                self.params[0] + self.params[2] * np.cos(t),
                self.params[1] + self.params[2] * np.sin(t),
            ]
        )

    @staticmethod
    def error_func(points, params):
        """
        Calculates the error between a set of points and a circle
        :param points: numpy array of shape (n_points, 2) containing the coordinates of the points
        :param params: numpy array of shape (3,) containing the parameters of the circle
        :return: numpy array of shape (n_points,) containing the error between the points and the circle
        """
        return (
            (points[:, 0] - params[0]) ** 2
            + (points[:, 1] - params[1]) ** 2
            - params[2] ** 2
        )


class LemniscateCurve(Curve):
    def __init__(self):
        self.n_params = 4

    def func(self, n_points):
        """
        Creates a lemniscate with n_points points
        :param n_points: number of points to be drawn
        :return: numpy array of shape (n_points, 2) containing the coordinates of the points
        """
        if self.params is None:
            raise Exception("No parameters found. Please fit the curve first.")

        t = np.linspace(0, 2 * np.pi, n_points)
        return np.array(
            [
                self.params[0]
                + self.params[2] * np.sqrt(2) * np.cos(t) / (1 + np.sin(t) ** 2),
                self.params[1]
                + self.params[2]
                * np.sqrt(2)
                * np.cos(t)
                * np.sin(t)
                / (self.params[3] * (1 + np.sin(t) ** 2)),
            ]
        )

    @staticmethod
    def error_func(points, params):
        """
        E = ((x-a)**2+d**2(y-b)**2)**2-2c**2((x-a)**2−d**2(y-b)**2)
        :param points: numpy array of shape (n_points, 2) containing the coordinates of the points
        :param params: numpy array of shape (4,) containing the parameters of the lemniscate
        :return: numpy array of shape (n_points,) containing the error between the points and the lemniscate
        """
        a, b, c, d = params
        x = points[:, 0] - a
        y = points[:, 1] - b
        return (x**2 + d**2 * y**2) ** 2 - 2 * c**2 * (x**2 - d**2 * y**2)

    def improve_mse(self, mse):
        """
        Improves the mse, increasing it if the parameters are not valid for example
        :param mse: current mse
        :return: new mse
        """
        if abs(self.params[3]) < 0.175 or abs(self.params[3]) > 5:
            return 100
        return mse


class GenericCurve(Curve):
    def __init__(self, filename=None):
        self.points = None

        if filename is not None:
            self.load_pattern(filename)

    def create_curve(self, threshold=5.0):
        """
        Opens a pygame window and waits for the user to draw a curve with the mouse.
        Only add point if distance between last point and current point is larger than threshold.
        :param n_points: number of points to be drawn
        :return: numpy array of shape (n_points, 2) containing the coordinates of the points
        """

        # Initialize pygame
        pygame.init()
        WIDTH = 800
        HEIGHT = 800
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("DEFINE PATTERN")
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
                elif event.type == pygame.MOUSEBUTTONUP:
                    running = False

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
        points = points - np.mean(points, axis=0)
        points = points / np.max(np.abs(points), axis=0)

        self.points = points

        return points

    def frechet_dist(self, other_points):
        return frdist(self.points, other_points)

    def save_pattern(self, filename):
        np.save(filename, self.points)

    def load_pattern(self, filename):
        self.points = np.load(filename)


class CurveMatching:
    def __init__(self):
        self.points = None

        self.curves = {
            "circle": CircleCurve,
            "lemniscate": LemniscateCurve,
        }

    def create_curve(self, threshold=5.0):
        """
        Opens a pygame window and waits for the user to draw a curve with the mouse.
        Only add point if distance between last point and current point is larger than threshold.
        :param n_points: number of points to be drawn
        :return: numpy array of shape (n_points, 2) containing the coordinates of the points
        """

        # Initialize pygame
        pygame.init()
        WIDTH = 800
        HEIGHT = 800
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
                elif event.type == pygame.MOUSEBUTTONUP:
                    running = False

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
        points = points - np.mean(points, axis=0)
        points = points / np.max(np.abs(points), axis=0)

        self.points = points

        return points

    def fit_curve(self, curve_name):
        """
        Fits a curve to the points
        :curve: Curve object
        """
        curve = self.curves[curve_name]()
        curve.fit_curve(self.points)

        return curve


if __name__ == "__main__":
    # pattern = GenericCurve()
    # pattern.create_curve()
    # pattern.save_pattern("patterns/inf0.npy")

    pattern = GenericCurve("patterns/inf0.npy")

    matcher = CurveMatching()
    matcher.create_curve()

    similarity = pattern.frechet_dist(matcher.points)

    print(similarity)

    # Plot pattern and curve
    plt.plot(pattern.points[:, 0], pattern.points[:, 1], label="pattern")
    plt.scatter(
        matcher.points[:, 0],
        matcher.points[:, 1],
        label="curve",
        c="g" if similarity < 0.7 else "r",
    )
    plt.gca().set_aspect("equal", adjustable="box")

    # Limit the plot to the area around the curve
    plt.xlim(np.min(matcher.points[:, 0]) - 0.1, np.max(matcher.points[:, 0]) + 0.1)
    plt.ylim(np.min(matcher.points[:, 1]) - 0.1, np.max(matcher.points[:, 1]) + 0.1)
    plt.legend()
    plt.show()

    # curve = matcher.fit_curve("heart")

    # print(curve.params)
    # print(curve.mse)

    # plt.scatter(matcher.points[:, 0], matcher.points[:, 1])
    # xs, ys = curve.func(100)
    # # plt.plot(xs, ys, label=f"mse: {curve.mse}")
    # plt.gca().set_aspect("equal", adjustable="box")
    # # Limit the plot to the area around the curve
    # plt.xlim(np.min(matcher.points[:, 0]) - 0.1, np.max(matcher.points[:, 0]) + 0.1)
    # plt.ylim(np.min(matcher.points[:, 1]) - 0.1, np.max(matcher.points[:, 1]) + 0.1)
    # plt.legend()
    # plt.show()
