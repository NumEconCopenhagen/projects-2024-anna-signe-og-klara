import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy import optimize
import itertools
import pandas as pd

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

        #set seed
        np.random.seed(1234)

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

class CareerChoiceSimulation:
    def __init__(self, N=10):
        # Set parameters
        self.par = SimpleNamespace()
        self.par.J = 3       # Number of career tracks
        self.par.N = N       # Total number of graduates
        self.par.K = 10000   # Number of draws for each graduate
        self.par.F = np.arange(1, N + 1)  # Fi for each graduate, incrementally from 1 to N
        self.par.sigma = 2
        self.par.v = np.array([1, 2, 3])  # Known values of v_j
        self.par.c = 1
        np.random.seed(1000)  # Set seed for random number generation

    def simulate_utility(self):
        v_known = self.par.v  # Known initial values of v_j
        # Initialize arrays to store results
        expected_utilities = np.zeros((self.par.J,))
        average_realized_utilities = np.zeros((self.par.J,))
        
        for j in range(self.par.J):
            # Generate epsilon values for each career track j
            epsilons = np.random.normal(loc=0, scale=self.par.sigma, size=(self.par.N, self.par.K))
            
            # Calculate utility u^k_(i,j) for each career track j
            utilities = v_known[j] + epsilons
            
            # Calculate expected utility E[u^k_(i,j)|v_j] for career track j
            expected_utilities[j] = v_known[j]  # v_j is the expected value since E[epsilon] = 0
            
            # Calculate average realized utility for career track j
            average_realized_utilities[j] = np.mean(utilities)
        
        return expected_utilities, average_realized_utilities
    
    def simulate_career_choice(self):
        # Simulate career choice and calculate prior expected utility and realized utility
        chosen_careers = np.zeros(self.par.N, dtype=int)
        prior_exp_utilities = np.zeros((self.par.N, self.par.J))
        realized_utilities = np.zeros(self.par.N)
        
        for i in range(self.par.N):  # Loop over graduates i = 1, ..., N
            Fi = self.par.F[i]  # Number of friends for graduate i (Fi)
            epsilons_friends = np.random.normal(loc=0, scale=self.par.sigma, size=(self.par.J * Fi, self.par.K))
            epsilons_self = np.random.normal(loc=0, scale=self.par.sigma, size=(self.par.J, self.par.K))
            
            # Calculate estimated utility for each career track j using friends' information and own information
            estimated_v_j = np.mean(epsilons_friends.reshape(Fi, self.par.J, self.par.K), axis=0).mean(axis=1) + self.par.v
            prior_exp_utilities[i] = estimated_v_j  # Store prior expected utility for graduate i
            
            # Calculate probabilities using softmax of estimated utilities
            probabilities = np.exp(estimated_v_j - np.max(estimated_v_j))
            probabilities /= probabilities.sum()
            chosen_careers[i] = np.random.choice(self.par.J, p=probabilities)
            
            # Calculate realized utility for the chosen career track
            realized_utilities[i] = self.par.v[chosen_careers[i]] + np.mean(epsilons_self[chosen_careers[i]])

        return chosen_careers, prior_exp_utilities, realized_utilities

    def plot_career_choices(self, chosen_careers):
        # Calculate the share of graduates choosing each career
        shares = np.zeros((self.par.N, self.par.J))
        for i in range(self.par.N):
            for j in range(self.par.J):
                shares[i, j] = np.sum(chosen_careers[:i + 1] == j) / (i + 1)
        
        # Plot the results
        plt.figure(figsize=(10, 5))
        for j in range(self.par.J):
            plt.plot(self.par.F, shares[:, j], label=f'Career {j + 1}')
        plt.xlabel('Number of Friends (Fi)')
        plt.ylabel('Share of Graduates Choosing Career')
        plt.title('Share of Graduates Choosing Each Career Based on Number of Friends')
        plt.legend()
        plt.grid(True)
        plt.savefig('career_choices.png')
        plt.show()

    def plot_expected_utility(self, prior_exp_utilities):
        # Calculate the average expected utility for each career
        avg_expected_utilities = np.mean(prior_exp_utilities, axis=1)
        
        # Plot the results
        plt.figure(figsize=(10, 5))
        for j in range(self.par.J):
            plt.plot(self.par.F, avg_expected_utilities[:, j], label=f'Career {j + 1}')
        plt.xlabel('Number of Friends (Fi)')
        plt.ylabel('Average Expected Utility')
        plt.title('Average Expected Utility for Each Career Based on Number of Friends')
        plt.legend()
        plt.grid(True)
        plt.savefig('expected_utility.png')
        plt.show()

    def plot_realized_utility(self, realized_utilities, chosen_careers):
        # Calculate the average realized utility for each career
        avg_realized_utilities = np.zeros((self.par.N, self.par.J))
        for i in range(self.par.N):
            for j in range(self.par.J):
                indices = (chosen_careers[:i + 1] == j)
                if np.sum(indices) > 0:
                    avg_realized_utilities[i, j] = np.mean(realized_utilities[:i + 1][indices])
        
        # Plot the results
        plt.figure(figsize=(10, 5))
        for j in range(self.par.J):
            plt.plot(self.par.F, avg_realized_utilities[:, j], label=f'Career {j + 1}')
        plt.xlabel('Number of Friends (Fi)')
        plt.ylabel('Average Realized Utility')
        plt.title('Average Realized Utility for Each Career Based on Number of Friends')
        plt.legend()
        plt.grid(True)
        plt.savefig('realized_utility.png')
        plt.show()

    def simulate_career_choices_prior(self):
        np.random.seed(7)
        results = []
        
        for k in range(self.par.K):
            epsilon_friends = {i: np.random.normal(0, self.par.sigma, (i, self.par.J)) for i in range(1, self.par.N + 1)}
            epsilon_personal = {i: np.random.normal(0, self.par.sigma, self.par.J) for i in range(1, self.par.N + 1)}
            
            for i in range(1, self.par.N + 1):
                # Calculate prior expected utilities
                prior_exp_utilities = self.par.v + np.mean(epsilon_friends[i], axis=0)
                
                # Determine the career choice with the highest expected utility
                chosen_career = np.argmax(prior_exp_utilities) + 1
                
                # Calculate the realized utility
                realized_utility = self.par.v[chosen_career - 1] + epsilon_personal[i][chosen_career - 1]
                
                results.append({
                    'Graduate': i,
                    'Chosen Career': chosen_career,
                    'Prior Expected Utility': prior_exp_utilities[chosen_career - 1],
                    'Realized Utility': realized_utility
                })
        
        results_df = pd.DataFrame(results)

        # Calculate the required statistics
        share_grad_choosing_career = results_df.groupby('Graduate')['Chosen Career'].value_counts(normalize=True).unstack(fill_value=0)
        ave_sub_exp_utility = results_df.groupby('Graduate')['Prior Expected Utility'].mean()
        ave_ex_post_real_utility = results_df.groupby('Graduate')['Realized Utility'].mean()

        return share_grad_choosing_career, ave_sub_exp_utility, ave_ex_post_real_utility

    def simulate_career_choices_switching(self):
        np.random.seed(7)
        results_with_switching = []

        for k in range(self.par.K):
            epsilon_friends = {i: np.random.normal(0, self.par.sigma, (i, self.par.J)) for i in range(1, self.par.N + 1)}
            epsilon_personal = {i: np.random.normal(0, self.par.sigma, self.par.J) for i in range(1, self.par.N + 1)}
            
            for i in range(1, self.par.N + 1):
                # Calculate prior expected utilities
                prior_exp_utilities = self.par.v + np.mean(epsilon_friends[i], axis=0)
                
                # Determine the career choice with the highest expected utility
                chosen_career = np.argmax(prior_exp_utilities) + 1
                
                # Calculate the realized utility
                realized_utility = self.par.v[chosen_career - 1] + epsilon_personal[i][chosen_career - 1]
                
                # After learning the realized utility, reconsider career choice
                new_prior_exp_utilities = prior_exp_utilities - self.par.c
                new_prior_exp_utilities[chosen_career - 1] = realized_utility
                
                new_chosen_career = np.argmax(new_prior_exp_utilities) + 1
                new_realized_utility = self.par.v[new_chosen_career - 1] + epsilon_personal[i][new_chosen_career - 1]
                
                results_with_switching.append({
                    'Graduate': i,
                    'Initial Chosen Career': chosen_career,
                    'Initial Realized Utility': realized_utility,
                    'New Chosen Career': new_chosen_career,
                    'New Realized Utility': new_realized_utility,
                    'Prior Expected Utility': prior_exp_utilities[chosen_career - 1],
                    'Switched': chosen_career != new_chosen_career
                })
        
        results_with_switching_df = pd.DataFrame(results_with_switching)

        # Calculate the required statistics
        share_grad_career_switching = results_with_switching_df.groupby('Graduate')['New Chosen Career'].value_counts(normalize=True).unstack(fill_value=0)
        ave_sub_exp_utility_switching = results_with_switching_df.groupby('Graduate')['Prior Expected Utility'].mean()
        ave_ex_post_real_utility_switching = results_with_switching_df.groupby('Graduate')['New Realized Utility'].mean()

        return share_grad_career_switching, ave_sub_exp_utility_switching, ave_ex_post_real_utility_switching

    def simulate_career_choices_switching_stats(self):
        share_grad_career_switching, ave_sub_exp_utility_switching, ave_ex_post_real_utility_switching = self.simulate_career_choices_switching()

        # Flatten the results correctly to match the shape
        graduates = np.repeat(range(1, self.par.N + 1), self.par.K)
        switched = share_grad_career_switching.values.flatten()

        # Ensure both arrays are of the same length
        assert len(graduates) == len(switched), "Length of graduates and switched arrays must be the same"

        results_with_switching_df = pd.DataFrame({
            'Graduate': graduates,
            'Switched': switched
        })

        # Calculate the share of graduates that switch careers conditional on their initial career choice
        switching_stats = results_with_switching_df.groupby('Graduate')['Switched'].mean()

        # Calculate the share of graduates switching from each initial career choice
        switching_shares = share_grad_career_switching

        return switching_stats, switching_shares

    def plot_switching_shares(self, switching_shares):
        # Visualize the results
        plt.figure(figsize=(14, 8))

        # Plot the share of graduates switching from each initial career
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        markers = ['o', 's', '^']
        for idx, career in enumerate(range(1, self.par.J + 1)):
            plt.plot(switching_shares.index, switching_shares[career], label=f'Switched from Career {career}', color=colors[idx], marker=markers[idx], linestyle='-')

        plt.title('Share of Graduates Switching Careers (Second Year)', fontsize=14)
        plt.xlabel('Number of Friends (F_i)', fontsize=12)
        plt.ylabel('Share of Graduates Switching Careers', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('switching_shares.png')
        plt.show()
        return switching_shares.index, switching_shares[1], switching_shares[2], switching_shares[3]

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
            "barycentric_ABC": (round(r1_ABC, 2), round(r2_ABC, 2), round(r3_ABC, 2)),
            "barycentric_CDA": (round(r1_CDA, 2), round(r2_CDA, 2), round(r3_CDA, 2)),
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
            "approximated_f_y": round(approx_f_y, 2),
            "true_f_y": round(true_f_y, 2)
        }
        return results

    # Do the same as above, but for multiple points in Y
    def compute_and_compare_multiple_Y(self, X):
        # Define the set Y
        Y = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.8, 0.2), (0.5, 0.5)]
        
        results = []
        for point in Y:
            result = self.compute_and_compare(point, X)
            approximated_f_y = round(result["approximated_f_y"], 2)
            true_f_y = round(result["true_f_y"], 2)
            results.append({"point": point, "approximated_f_y": approximated_f_y, "true_f_y": true_f_y})
        return results
    
    def plot_y_specific(self, X, y, A, B, C, D):
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

        if not (np.isnan(A).any() or np.isnan(B).any() or np.isnan(C).any()):
            triangle_ABC = plt.Polygon([A, B, C], fill=None, edgecolor='k', linestyle='--')
            plt.gca().add_patch(triangle_ABC)
        if not (np.isnan(C).any() or np.isnan(D).any() or np.isnan(A).any()):
            triangle_CDA = plt.Polygon([C, D, A], fill=None, edgecolor='c', linestyle='--')
            plt.gca().add_patch(triangle_CDA)

        plt.legend()
        plt.show()