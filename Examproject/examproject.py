import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy import optimize
import itertools
import numpy as np
from scipy import optimize
from types import SimpleNamespace

class ProductionEconomyClass:

    def __init__(self):
        # Define the parameters
        par = self.par = SimpleNamespace()
        # firms
        par.A = 1.0
        par.gamma = 0.5
        # households
        par.alpha = 0.3
        par.nu = 1.0
        par.epsilon = 2.0
        # government
        par.tau = 0.0
        par.T = 0.0
        # welfare
        par.kappa = 0.1
        
        # Placeholder for equilibrium prices
        self.p1 = None
        self.p2 = None
        self.w = 1  # numeraire

    def utility(self, c1, c2, l):
        return np.log(c1 ** self.par.alpha * c2 ** (1 - self.par.alpha)) - self.par.nu * l ** (1 + self.par.epsilon) / (1 + self.par.epsilon)

    def firm_behavior(self, w, p):
        l_star = (p * self.par.A * self.par.gamma / w) ** (1 / (1 - self.par.gamma))
        y_star = self.par.A * l_star ** self.par.gamma
        profit_star = p * y_star - w * l_star
        return l_star, y_star, profit_star

    def consumer_behavior(self, w, p1, p2, T):
        income = w + T
        c1_star = self.par.alpha * income / p1
        c2_star = (1 - self.par.alpha) * income / (p2 + self.par.tau)
        l_star = income / w
        return c1_star, c2_star, l_star

    def check_market_clearing(self):
        p1_values = np.linspace(0.1, 2.0, 10)
        p2_values = np.linspace(0.1, 2.0, 10)
        results = []

        for p1 in p1_values:
            for p2 in p2_values:
                l1_star, y1_star, profit1_star = self.firm_behavior(1, p1)
                l2_star, y2_star, profit2_star = self.firm_behavior(1, p2)
                
                c1_star, c2_star, l_star = self.consumer_behavior(1, p1, p2, self.par.T + profit1_star + profit2_star)
                
                labor_market_clearing = np.isclose(l_star, l1_star + l2_star)
                good1_market_clearing = np.isclose(c1_star, y1_star)
                good2_market_clearing = np.isclose(c2_star, y2_star)
                
                print(f"p1 = {p1:.1f}, p2 = {p2:.1f}")
                print(f"Labor Market Clearing: {labor_market_clearing}")
                print(f"Good 1 Market Clearing: {good1_market_clearing}")
                print(f"Good 2 Market Clearing: {good2_market_clearing}")
                print("-" * 30)
                
                results.append((round(p1, 2), round(p2, 2), labor_market_clearing and good1_market_clearing and good2_market_clearing))
        return #results 

    def find_equilibrium_prices(self):
        def excess_demand(p):
            l1_star, y1_star, profit1_star = self.firm_behavior(1, p[0])
            l2_star, y2_star, profit2_star = self.firm_behavior(1, p[1])
            c1_star, c2_star, l_star = self.consumer_behavior(1, p[0], p[1], self.par.T + profit1_star + profit2_star)
            excess_demand1 = y1_star - c1_star
            excess_demand2 = y2_star - c2_star
            return [excess_demand1, excess_demand2]

        result = optimize.root(excess_demand, [1, 1])
        if result.success:
            self.p1, self.p2 = result.x
            print(f"Equilibrium prices: p1 = {self.p1:.2f}, p2 = {self.p2:.2f}")
            return self.p1, self.p2
        return None, None

    def c1(self, l):
        return self.par.alpha * (self.w * l + self.par.T + self.firm_behavior(self.w, self.p1)[2] + self.firm_behavior(self.w, self.p2)[2]) / self.p1
    
    def c2(self, l):
        return (1 - self.par.alpha) * (self.w * l + self.par.T + self.firm_behavior(self.w, self.p1)[2] + self.firm_behavior(self.w, self.p2)[2]) / (self.p2 + self.par.tau)
    
    def optimal_behavior(self):
        objective = lambda l: -(self.par.alpha * np.log(self.c1(l)) + (1 - self.par.alpha) * np.log(self.c2(l)) - self.par.nu * (l ** (1 + self.par.epsilon)) / (1 + self.par.epsilon))
        result = optimize.minimize_scalar(objective, bounds=(0, 10), method='bounded')
        if result.success:
            return result.x
        else:
            raise ValueError("Optimization failed")
        
    def consumption_without_tax(self):
        self.par.tau = 0.0
        self.par.T = 0.0
        
        l_star = self.optimal_behavior()
        
        c1_no_tax = self.c1(l_star)
        c2_no_tax = self.c2(l_star)
        
        print(f"Consumption of c1 without tax: {c1_no_tax:.2f}")
        print(f"Consumption of c2 without tax: {c2_no_tax:.2f}")
        return # c1_no_tax, c2_no_tax

#Question 3
    def social_welfare(self):
        l_star = self.optimal_behavior()
        c1_star = self.c1(l_star)
        c2_star = self.c2(l_star)
        utility = self.utility(c1_star, c2_star, l_star)
        swf = utility - self.par.kappa * c2_star
        return swf

    def swf_with_tax(self, tau):
        self.par.tau = tau
        self.par.T = self.par.tau * self.c2(self.optimal_behavior())
        return -self.social_welfare()

    def find_optimal_tax(self):
        result = optimize.minimize_scalar(self.swf_with_tax, bounds=(0, 2), method='bounded')
        if result.success:
            self.par.tau = result.x
            self.par.T = self.par.tau * self.c2(self.optimal_behavior())
        else:
            raise ValueError("Optimization failed")
        
        l_star = self.optimal_behavior()
        c1_star_tax = self.c1(l_star)
        c2_star_tax = self.c2(l_star)
        

        print(f"Consumption of c1 with tax:   {c1_star_tax:.2f}")
        print(f"Consumption of c2 with tax:   {c2_star_tax:.2f}")
        print(f"Optimal tax rate:             {self.par.tau:.2f}")
        print(f"Optimal lumpsum transfer:     {self.par.T:.2f}")
        print(f"Social welfare:              {-result.fun:.2f}")
       # return c1_star_tax, c2_star_tax, self.par.tau, self.par.T, -result.fun



class BarycentricInterpolation:

    # Calculate the barycentric coordinates, r1, r2, r3
    def r1(self, args, A, B, C):
        y1 = args[0]
        y2 = args[1]
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
        for condition in conditions:
            filtered_points = X[condition]
            if len(filtered_points) == 0:
                closest_points.append(np.NaN)
            else:
                distances = self.distance(filtered_points, y)
                closest_point = filtered_points[np.argmin(distances)]
                closest_points.append(closest_point)

        A, B, C, D = closest_points
        return A, B, C, D

    # Function to plot the points in X, the point y, and the points A, B, C, D
    def plot_points_and_triangles(self, X, y, A, B, C, D):
        plt.scatter(X[:, 0], X[:, 1], label='Points in X')
        plt.scatter(*y, color='r', label='Point y')

        if not np.isnan(A).any():
            plt.scatter(*A, color='g', label='Point A')
        if not np.isnan(B).any():
            plt.scatter(*B, color='b', label='Point B')
        if not np.isnan(C).any():
            plt.scatter(*C, color='m', label='Point C')
        if not np.isnan(D).any():
            plt.scatter(*D, color='y', label='Point D')

        # Draw triangles
        if not (np.isnan(A).any() or np.isnan(B).any() or np.isnan(C).any()):
            triangle_ABC = plt.Polygon([A, B, C], fill=None, edgecolor='k', linestyle='--')
            plt.gca().add_patch(triangle_ABC)
        if not (np.isnan(C).any() or np.isnan(D).any() or np.isnan(A).any()):
            triangle_CDA = plt.Polygon([C, D, A], fill=None, edgecolor='c', linestyle='--')
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
        for point in Y:
            result = self.compute_and_compare(point, X)
            approximated_f_y = result["approximated_f_y"]
            true_f_y = result["true_f_y"]
            results.append({"point": point, "approximated_f_y": approximated_f_y, "true_f_y": true_f_y})
        return results
