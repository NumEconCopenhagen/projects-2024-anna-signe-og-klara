import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy import optimize
import itertools

class BarycentricInterpolation:

    # Calculate the barycentric coordinates, r1, r2, r3
    def r1(self, args, A, B, C):
        # Assigning the two arguments to y1 and y2
        y1 = args[0]
        y2 = args[1]
        # Splitting the formula for r1 in nominator and denominator 
        nom = (B[1] - C[1]) * (y1 - C[0]) + (C[0] - B[0]) * (y2 - C[1])
        denom = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        r = nom / denom
        return r   
    
    def r2(self, args, A, B, C):
        y1 = args[0]
        y2 = args[1]
        nom = (C[1] - A[1]) * (y1 - C[0]) + (A[0] - C[0]) * (y2 - C[1])
        denom = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        r = nom / denom
        return r
    
    def r3(self, r1, r2):
        r = 1 - r1 - r2
        return r
    
    # Function to calculate the distance between points in X and the point y
    def distance(self, X, y):
        return np.sqrt((X[:, 0] - y[0])**2 + (X[:, 1] - y[1])**2)
    
    # Calculating the closest point in different quadrants around y, for A, B, C, D
    def closest_points_in_quadrants(self, X, y):
        # Define conditions for each quadrant
        conditions = [
            (X[:, 0] > y[0]) & (X[:, 1] > y[1]),  # Quadrant A: top-right
            (X[:, 0] > y[0]) & (X[:, 1] < y[1]),  # Quadrant B: bottom-right
            (X[:, 0] < y[0]) & (X[:, 1] < y[1]),  # Quadrant C: bottom-left
            (X[:, 0] < y[0]) & (X[:, 1] > y[1])   # Quadrant D: top-left
        ]
        # Based on the conditions, find the closest point in each quadrant
        closest_points = []
        # loop through the conditions
        for condition in conditions:
            # extract the points that meet the condition
            filtered_points = X[condition]
            # if there are no points that meet the condition, append NaN
            if len(filtered_points) == 0:
                closest_points.append(np.NaN)
            # else find the closest point to y
            else:
                distances = self.distance(filtered_points, y)
                closest_point = filtered_points[np.argmin(distances)]
                closest_points.append(closest_point)

        # Return the four points
        A, B, C, D = closest_points
        return A, B, C, D

    # Function to plot the points in X, the point y, and the points A, B, C, D
    def plot_points_and_triangles(self, X, y, A, B, C, D):
        plt.scatter(X[:, 0], X[:, 1], label='Points in X')
        plt.scatter(*y, color='r', label='Point y')
        plt.scatter(*A, color='g', label='Point A')
        plt.scatter(*B, color='b', label='Point B')
        plt.scatter(*C, color='m', label='Point C')
        plt.scatter(*D, color='y', label='Point D')

        # Draw triangles
        triangle_ABC = plt.Polygon([A, B, C], fill=None, edgecolor='k', linestyle='--')
        triangle_CDA = plt.Polygon([C, D, A], fill=None, edgecolor='c', linestyle='--')
        plt.gca().add_patch(triangle_ABC)
        plt.gca().add_patch(triangle_CDA)

        plt.legend()
        plt.show()

    # Checking if the restrictions on the baycentric coordinates are met for the point y to be inside the triangle
    def isin_triangle(self, args, A, B, C):
        r1val = self.r1(args, A, B, C)
        r2val = self.r2(args, A, B, C)
        r3val = self.r3(r1val, r2val)
        inside = 0 <= r1val <= 1 and 0 <= r2val <= 1 and 0 <= r3val <= 1
        return inside
    
    # Function to calculate the value of y based on the barycentric coordinates
    def y_function(self, args, A, B, C):    
        r1val = self.r1(args, A, B, C)
        r2val = self.r2(args, A, B, C)
        r3val = self.r3(r1val, r2val)
        y = r1val * A + r2val * B + r3val * C
        return y
    
    def compute_barycentric_and_check(self, y, A, B, C, D):
        # Compute barycentric coordinates with respect to ABC
        r1_ABC = self.r1(y, A, B, C)
        r2_ABC = self.r2(y, A, B, C)
        r3_ABC = self.r3(r1_ABC, r2_ABC)

        # Compute barycentric coordinates with respect to CDA
        r1_CDA = self.r1(y, C, D, A)
        r2_CDA = self.r2(y, C, D, A)
        r3_CDA = self.r3(r1_CDA, r2_CDA)

        # Check which triangle y is inside
        inside_ABC = self.isin_triangle(y, A, B, C)
        inside_CDA = self.isin_triangle(y, C, D, A)

        results = {
            "barycentric_ABC": (r1_ABC, r2_ABC, r3_ABC),
            "barycentric_CDA": (r1_CDA, r2_CDA, r3_CDA),
            "inside_ABC": inside_ABC,
            "inside_CDA": inside_CDA
        }
        return results


    # Approximate the value of F based on the barycentric coordinates
    def approximate_f(self, y, X, F):
        A, B, C, D = self.closest_points_in_quadrants(X, y)
        
        print(f"Point {y}: Closest points are A={A}, B={B}, C={C}, D={D}")  # Debugging statement

        if np.isnan(A).any() or np.isnan(B).any() or np.isnan(C).any() or np.isnan(D).any():
            return np.nan
        
        # Find the indices of the points in X since A, B, C, D are not valid indices for array F
        A_idx = np.where((X == A).all(axis=1))[0][0]
        B_idx = np.where((X == B).all(axis=1))[0][0]
        C_idx = np.where((X == C).all(axis=1))[0][0]
        D_idx = np.where((X == D).all(axis=1))[0][0]

        # Checking if y is inside ABC
        if self.isin_triangle(y, A, B, C):
            r1val = self.r1(y, A, B, C)
            r2val = self.r2(y, A, B, C)
            r3val = self.r3(r1val, r2val)
            return r1val * F[A_idx] + r2val * F[B_idx] + r3val * F[C_idx]

        # Checking if y is inside CDA
        if self.isin_triangle(y, C, D, A):
            r1val = self.r1(y, C, D, A)
            r2val = self.r2(y, C, D, A)
            r3val = self.r3(r1val, r2val)
            return r1val * F[C_idx] + r2val * F[D_idx] + r3val * F[A_idx]

        return np.nan
    
    # Function to compute and compare the approximated value of f(y) and the true value of f(y)
    def compute_and_compare(self, y, X):
        # Define the function f(x1, x2)
        f = lambda x: x[0] * x[1]

        # Compute F
        F = np.array([f(x) for x in X])

        # Approximate f(y)
        approx_f_y = self.approximate_f(y, X, F)

        # Compute the true value of f(y)
        true_f_y = f(y)

        results = {
            "approximated_f_y": approx_f_y,
            "true_f_y": true_f_y
        }
        return results

    # Do the same as above, but for multiple points in Y
    def compute_and_compare_multiple_Y(self, X):
        # Define the set Y
        Y = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.8, 0.2), (0.5, 0.5)]
        
        results = []
        differences = []
        for point in Y:
            print(f"Computing for point: {point}")  # Debugging statement
            result = self.compute_and_compare(point, X)
            approximated_f_y = result["approximated_f_y"]
            true_f_y = result["true_f_y"]
            print(f"Approximated value: {approximated_f_y}, True value: {true_f_y}")  # Debugging statement
            difference = np.abs(approximated_f_y - true_f_y)
            differences.append(difference)
            results.append({"point": point, "approximated_f_y": approximated_f_y, "true_f_y": true_f_y})
        