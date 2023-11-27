import numpy as np


class Delaunay2D:
    def __init__(self, points):
        """
        Initialize the Delaunay2D triangulation with a list of points.

        Parameters:
        - points: List of 2D points (tuples or lists).
        """
        self.points = points

        # Dictionary of triangles => triangle:
        # [(adj to vertices 0 and 1), (adj to vertices 1 and 2), (adj to vertices 2 and 0)]
        self.triangles = {}

        # Index of the current point to be added
        self.index = 0

    def triangulate(self):
        """
        Triangulate the input points using the Bowyer-Watson algorithm.
        """
        # Create a super triangle that encompasses all input points
        self.super_triangle = self.makeSuperTriangle()
        self.triangles[self.super_triangle] = [None, None, None]

        # Add each input point to the triangulation
        for point in self.points:
            self.addPoint(point)
            self.index += 1

        # Remove the super triangle from the final triangulation
        self.removeSuperTriangle()

    def makeSuperTriangle(self):
        """
        Create a super triangle that encompasses all input points.

        Returns:
        - Super triangle vertices as a tuple of three points.
        """
        array_of_points = np.array([np.array(point) for point in self.points])
        x_min, x_max = (
            np.min(array_of_points[:, 0]) - 0.05,
            np.max(array_of_points[:, 0]) + 0.05,
        )
        y_min, y_max = (
            np.min(array_of_points[:, 1]) - 0.05,
            np.max(array_of_points[:, 1]) + 0.05,
        )
        dx = x_max - x_min
        dy = y_max - y_min

        v1 = x_max + dx * 0.5, y_min
        v2 = x_min - dx * 0.5, y_min
        v3 = x_min + dx * 0.5, y_max + dy

        return (v1, v2, v3)

    def removeSuperTriangle(self):
        """
        Remove triangles containing vertices from the super triangle.
        """
        for triangle in list(self.triangles):
            if self.super_triangle[0] in triangle:
                del self.triangles[triangle]
            elif self.super_triangle[1] in triangle:
                del self.triangles[triangle]
            elif self.super_triangle[2] in triangle:
                del self.triangles[triangle]

    def addPoint(self, p):
        """
        Add a point to the Delaunay triangulation.

        Parameters:
        - p: Point to be added (tuple or list).
        """
        # Find triangles which circumcircle contains the new point
        bad_triangles = []
        for triangle in self.triangles:
            if self.in_circumcircle(triangle, p):
                bad_triangles.append(triangle)

        # Determine the boundary edges of the "bad" triangles
        boundary = self.get_boundary(bad_triangles)

        # Remove the "bad" triangles from the triangulation
        for triangle in bad_triangles:
            del self.triangles[triangle]

        # Create new triangles connecting the point to add to the boundary edges
        new_triangles = []
        for e0, e1, op_tri in boundary:
            triangle = (p, e0, e1)
            self.triangles[triangle] = [None, op_tri, None]

            # Update the adjacency information of the neighboring triangles
            if op_tri is not None:
                self.triangles[op_tri][list(op_tri).index(e1)] = triangle

            new_triangles.append(triangle)

        # Connect the new triangles to each other
        n_new_triangles = len(new_triangles)
        for i, triangle in enumerate(new_triangles):
            self.triangles[triangle][0] = new_triangles[(i - 1) % n_new_triangles]
            self.triangles[triangle][2] = new_triangles[(i + 1) % n_new_triangles]

    def get_boundary(self, triangles):
        """
        Get the boundary edges of a list of triangles.

        Parameters:
        - triangles: List of triangles.

        Returns:
        - List of boundary edges as tuples of three points (two vertices and their adjacent triangle).
        """
        boundary = []
        triangle = triangles[0]
        edge = 0

        while not boundary or boundary[0][0] != boundary[-1][1]:
            # Traverse all the triangles in a cyclic manner until the boundary is closed
            op_tri = self.triangles[triangle][edge]

            if op_tri not in triangles:
                # If the edge is part of the boundary, add it to the list and move to the next edge
                boundary.append((triangle[edge], triangle[(edge + 1) % 3], op_tri))
                edge = (edge + 1) % 3

            else:
                # If the edge is not part of the boundary, move to the next triangle
                edge = (self.triangles[op_tri].index(triangle) + 1) % 3
                triangle = op_tri

        return boundary

    def in_circumcircle(self, triangle, point):
        """
        Check if a point lies inside the circumcircle of a triangle.

        Parameters:
        - triangle: Tuple representing a triangle.
        - point: Point to be checked.

        Returns:
        - True if the point is inside the circumcircle, False otherwise.
        """

        project = np.array(
            [
                [1, 1, 1, 1],
                [triangle[0][0], triangle[1][0], triangle[2][0], point[0]],
                [triangle[0][1], triangle[1][1], triangle[2][1], point[1]],
                [
                    triangle[0][0] ** 2 + triangle[0][1] ** 2,
                    triangle[1][0] ** 2 + triangle[1][1] ** 2,
                    triangle[2][0] ** 2 + triangle[2][1] ** 2,
                    point[0] ** 2 + point[1] ** 2,
                ],
            ]
        )

        return np.linalg.det(project) > 0

    def convex_hull_graham(self, points):
        """
        Compute the convex hull of a set of points using the Graham scan algorithm.
        """
        hull = []
        for p in points + [points[0]]:
            # if len(hull) >= 2:
            # self.plot(boundary=((hull[-2], hull[-1]), (hull[-1], p), (p, hull[-2])))
            if len(hull) >= 2 and self.signed_area((hull[-2], hull[-1], p)) >= 0:
                self.triangles[(hull[-2], hull[-1], p)] = [None, None, None]
                hull.pop()
            hull.append(p)

        return hull

    def signed_area(self, triangle):
        """
        Compute the signed area of a triangle.

        Parameters:
        - triangle: Tuple representing a triangle.

        Returns:
        - Signed area of the triangle.
        """
        return 0.5 * np.linalg.det(
            np.array(
                [
                    [1, 1, 1],
                    [triangle[0][0], triangle[1][0], triangle[2][0]],
                    [triangle[0][1], triangle[1][1], triangle[2][1]],
                ]
            )
        )


import random

# generate a random set of points
points = [(random.random(), random.random()) for _ in range(100)]
# compute the Delaunay triangulation
d = Delaunay2D(points)
d.triangulate()

import matplotlib.pyplot as plt

# plot the triangulation

for triangle in d.triangles:
    plt.plot(
        [triangle[0][0], triangle[1][0], triangle[2][0], triangle[0][0]],
        [triangle[0][1], triangle[1][1], triangle[2][1], triangle[0][1]],
        "k-",
    )
plt.show()
