import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy import optimize
import itertools

class ProductionEconomyClass:

    def __init__(self):
        #Define the parameters
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
        
    def optimal_labor(self, w, p): # optimal labor
        return ((self.par.A * self.par.gamma * p) / w) ** (1 / (1 - self.par.gamma))
    
    def optimal_production(self, w, p): # optimal production
        return self.par.A * (self.optimal_labor(w, p) ** self.par.gamma)
    
    def profit(self, w, p): # profit
        return ((1 - self.par.gamma) / self.par.gamma) * w * ((p * self.par.A * self.par.gamma / w) ** (1 / (1 - self.par.gamma)))
    
    def consumer_utility(self, c1, c2, l): # consumer utility
        return np.log(c1 ** self.par.alpha * c2 ** (1 - self.par.alpha)) - self.par.nu * (l ** (1 + self.par.epsilon)) / (1 + self.par.epsilon)
    
    def check_market_clearing(self): # check market clearing
        results = []
        self.p1_range = np.linspace(0.1, 2.0, 10) # range of p1
        self.p2_range = np.linspace(0.1, 2.0, 10) # range of p2
        for p1 in self.p1_range: # loop over p1
            for p2 in self.p2_range: # loop over p2
                l_star = self.optimal_labor(self.w, p1) + self.optimal_labor(self.w, p2)
                c1_star = self.optimal_production(self.w, p1)
                c2_star = self.optimal_production(self.w, p2)
                
                # Check labor and goods markets
                labor_market_cleared = np.isclose(l_star, self.optimal_labor(self.w, p1) + self.optimal_labor(self.w, p2))
                good_market_1_cleared = np.isclose(c1_star, self.optimal_production(self.w, p1))
                good_market_2_cleared = np.isclose(c2_star, self.optimal_production(self.w, p2))
                
                if labor_market_cleared and good_market_1_cleared and good_market_2_cleared: # check if all markets are cleared
                    results.append((round(p1, 2), round(p2, 2), True)) # append the results
                else:
                    results.append((round(p1, 2), round(p2, 2), False))
        return results 

    def find_equilibrium_prices(self): # find equilibrium prices from the check_market_clearing function results
        results = self.check_market_clearing()
        for result in results:
            if result[2]:
                self.p1, self.p2, _, = result
                print(f"Equilibrium prices: p1 = {self.p1}, p2 = {self.p2}")
                return self.p1, self.p2
        return None, None
    
    def c1(self, l): # consumption of c1 in order to calculate consumption
        return self.par.alpha * (self.w * l + self.par.T + self.profit(self.w, self.p1) + self.profit(self.w, self.p2)) / self.p1
    
    def c2(self, l): # consumption of c2 in order to calculate consumption
        return (1 - self.par.alpha) * (self.w * l + self.par.T + self.profit(self.w, self.p1) + self.profit(self.w, self.p2)) / (self.p2 + self.par.tau)
    
    def optimal_behavior(self): # optimal behavior of the consumer in order to calculate consumption by maximizing the objective function
        objective = lambda l: -(self.par.alpha * np.log(self.c1(l)) + (1 - self.par.alpha) * np.log(self.c2(l)) - self.par.nu * (l ** (1 + self.par.epsilon)) / (1 + self.par.epsilon))
        result = optimize.minimize_scalar(objective, bounds=(0, 10), method='bounded')
        if result.success:
            return result.x
        else:
            raise ValueError("Optimization failed")
        
    def consumption_without_tax(self):
        # Set tau and T to zero explicitly as there is no tax
        self.par.tau = 0.0
        self.par.T = 0.0
        
        # Find the optimal labor for calculating consumption without tax
        l_star = self.optimal_behavior()
        
        # Calculate consumption of c1 and c2 without the tax
        c1_no_tax = self.c1(l_star)
        c2_no_tax = self.c2(l_star)
        
        # Print the results
        print(f"Consumption of c1 without tax: {c1_no_tax}")
        print(f"Consumption of c2 without tax: {c2_no_tax}")
        return c1_no_tax, c2_no_tax
      
        
    def social_welfare(self): #defining and calculating the social welfare function 
        l_star = self.optimal_behavior()
        c1_star = self.c1(l_star)
        c2_star = self.c2(l_star)
        utility = self.consumer_utility(c1_star, c2_star, l_star)
        swf = utility - self.par.kappa * c2_star
        return swf

    def swf_with_tax(self, tau): #defining the tax tau and T
        self.par.tau = tau
        self.par.T = self.par.tau * self.c2(self.optimal_behavior())
        return -self.social_welfare()  # Minimize negative SWF to maximize SWF

    def find_optimal_tax(self): # find the optimal tax rate for the government
        result = optimize.minimize_scalar(self.swf_with_tax, bounds=(0, 2), method='bounded')
        if result.success:
            self.par.tau = result.x
            self.par.T = self.par.tau * self.c2(self.optimal_behavior())
        else:
            raise ValueError("Optimization failed")
    
        #Calculate the consumption for c1 and c2 with the tax

        # Find the optimal labor for calculating consumption without tax
        l_star = self.optimal_behavior()
        
        # Calculate the consumption values 
        c1_star_tax = self.c1(l_star)
        c2_star_tax = self.c2(l_star)

        # Print the results
        print(f"Consumption of c1 with tax: {c1_star_tax}")
        print(f"Consumption of c2 with tax: {c2_star_tax}")
        print(f"Optimal tax rate: {self.par.tau}")
        print(f"Social welfare: {-result.fun}")
        return c1_star_tax, c2_star_tax, self.par.tau, -result.fun
    
class CareerChoiceSimulation: #defining the class for the career choice simulation
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

    def simulate_utility(self): #simulate the utility for the career choice
        v_known = np.array([1, 2, 3])  # Known initial values of v_j
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
        prior_expected_utilities = np.zeros((self.par.N, self.par.J))
        realized_utilities = np.zeros(self.par.N)
        
        for i in range(self.par.N): # Loop over graduates i = 1, ..., N 
            Fi = self.par.F[i] # Number of friends for graduate i (Fi) 
            epsilons_friends = np.random.normal(loc=0, scale=self.par.sigma, size=(self.par.J * Fi, self.par.K)) # Draw epsilon values for friends
            epsilons_self = np.random.normal(loc=0, scale=self.par.sigma, size=(self.par.J, self.par.K)) # Draw epsilon values for graduate i
            
            # Calculate estimated utility for each career track j using friends' information and own information
            estimated_v_j = np.mean(epsilons_friends.reshape(Fi, self.par.J, self.par.K), axis=0).mean(axis=1) + self.par.v
            prior_expected_utilities[i] = estimated_v_j # Store prior expected utility for graduate i
            
            # Calculate probabilities using softmax of estimated utilities
            probabilities = np.exp(estimated_v_j - np.max(estimated_v_j))
            probabilities /= probabilities.sum()
            chosen_careers[i] = np.random.choice(self.par.J, p=probabilities)
            
            # Calculate realized utility for the chosen career track
            realized_utilities[i] = self.par.v[chosen_careers[i]] + np.mean(epsilons_self[chosen_careers[i]])

        return chosen_careers, prior_expected_utilities, realized_utilities

    def analyze_results(self, chosen_careers, prior_expected_utilities, realized_utilities): #analyze the results of the simulation in order to compare the prior and realized utilities
        career_counts = np.zeros(self.par.J, dtype=int) # Initialize array to store counts of chosen careers
        for career in chosen_careers: # Loop over chosen careers
            career_counts[career] += 1 # Increment count for chosen career
        
        career_shares = career_counts / self.par.N # Calculate career shares
        average_prior_utilities = np.mean(prior_expected_utilities, axis=0) # Calculate average prior expected utilities
        average_realized_utilities = np.zeros(self.par.J) # Initialize array to store average realized utilities
        for j in range(self.par.J): # Loop over career tracks
            indices = (chosen_careers == j)
            average_realized_utilities[j] = np.mean(realized_utilities[indices])

        utility_differences = average_realized_utilities - average_prior_utilities # Calculate differences between realized and prior utilities
        
        return career_shares, average_prior_utilities, average_realized_utilities, utility_differences
 
    def simulate_initial_choice(self): #simulate the initial career choice
        chosen_careers = np.zeros(self.par.N, dtype=int) # Initialize array to store chosen careers
        prior_expected_utilities = np.zeros((self.par.N, self.par.J)) # Initialize array to store prior expected utilities
        realized_utilities = np.zeros(self.par.N) # Initialize array to store realized utilities
        
        for i in range(self.par.N): # Loop over graduates i = 1, ..., N 
            Fi = self.par.F[i] # Number of friends for graduate i (Fi) 
            epsilons_friends = np.random.normal(loc=0, scale=self.par.sigma, size=(self.par.J * Fi, self.par.K)) 
            epsilons_self = np.random.normal(loc=0, scale=self.par.sigma, size=(self.par.J, self.par.K))
            
            estimated_v_j = np.mean(epsilons_friends.reshape(Fi, self.par.J, self.par.K), axis=0).mean(axis=1) + self.par.v
            prior_expected_utilities[i] = estimated_v_j
            
            # Calculate probabilities using softmax of estimated utilities
            probabilities = np.exp(estimated_v_j - np.max(estimated_v_j))
            probabilities /= probabilities.sum()
            chosen_careers[i] = np.random.choice(self.par.J, p=probabilities)
            
            # Calculate realized utility for the chosen career track
            realized_utilities[i] = self.par.v[chosen_careers[i]] + np.mean(epsilons_self[chosen_careers[i]])

        return chosen_careers, prior_expected_utilities, realized_utilities


    def simulate_career_switch(self, chosen_careers, realized_utilities):
            new_chosen_careers = np.zeros(self.par.N, dtype=int)
#            new_prior_expected_utilities = np.zeros((self.par.N, self.par.J))
            new_realized_utilities = np.zeros(self.par.N)
            switched_careers = np.zeros(self.par.N, dtype=bool)
            
            for i in range(self.par.N):
                Fi = self.par.F[i]
                epsilons_friends = np.random.normal(loc=0, scale=self.par.sigma, size=(self.par.J * Fi, self.par.K))
                epsilons_self = np.random.normal(loc=0, scale=self.par.sigma, size=(self.par.J, self.par.K))
                
                estimated_v_j = np.mean(epsilons_friends.reshape(Fi, self.par.J, self.par.K), axis=0).mean(axis=1) + self.par.v
                prior_expected_utilities = estimated_v_j.copy()
                prior_expected_utilities[chosen_careers[i]] = self.par.v[chosen_careers[i]]
                prior_expected_utilities[prior_expected_utilities != self.par.v[chosen_careers[i]]] -= self.par.c

                # Calculate probabilities using softmax of estimated utilities
                probabilities = np.exp(prior_expected_utilities - np.max(prior_expected_utilities))
                probabilities /= probabilities.sum()
                new_chosen_careers[i] = np.random.choice(self.par.J, p=probabilities)
                
                # Check if they switched careers
                switched_careers[i] = new_chosen_careers[i] != chosen_careers[i]
                
                # Calculate realized utility for the new chosen career track
                new_realized_utilities[i] = self.par.v[new_chosen_careers[i]] + np.mean(epsilons_self[new_chosen_careers[i]])

            return new_chosen_careers, prior_expected_utilities, new_realized_utilities, switched_careers


class BarycentricInterpolation:
    
    def __init__(self):
        self.rng = np.random.default_rng(2024)
        self.X = self.rng.uniform(size=(50, 2))
        self.y = self.rng.uniform(size=(2,))
        self.F = np.array([self.func(x) for x in self.X])
    
    @staticmethod
    def func(x):
        return x[0] * x[1]
    
    def compute_points(self):
        y = self.y
        distances = np.linalg.norm(self.X - y, axis=1)
        A = self.X[np.argmin(distances + (self.X[:, 0] <= y[0]) + (self.X[:, 1] <= y[1]))]
        B = self.X[np.argmin(distances + (self.X[:, 0] <= y[0]) + (self.X[:, 1] >= y[1]))]
        C = self.X[np.argmin(distances + (self.X[:, 0] >= y[0]) + (self.X[:, 1] >= y[1]))]
        D = self.X[np.argmin(distances + (self.X[:, 0] >= y[0]) + (self.X[:, 1] <= y[1]))]
        return A, B, C, D
    
    @staticmethod
    def barycentric_coordinates(A, B, C, y):
        denom = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        r1 = ((B[1] - C[1]) * (y[0] - C[0]) + (C[0] - B[0]) * (y[1] - C[1])) / denom
        r2 = ((C[1] - A[1]) * (y[0] - C[0]) + (A[0] - C[0]) * (y[1] - C[1])) / denom
        r3 = 1 - r1 - r2
        return r1, r2, r3
    
    @staticmethod
    def is_inside_triangle(r1, r2, r3):
        return (0 <= r1 <= 1) and (0 <= r2 <= 1) and (0 <= r3 <= 1)
    
    def interpolate(self):
        A, B, C, D = self.compute_points()
        r1, r2, r3 = self.barycentric_coordinates(A, B, C, self.y)
        
        if self.is_inside_triangle(r1, r2, r3):
            fA = self.F[np.where((self.X == A).all(axis=1))[0][0]]
            fB = self.F[np.where((self.X == B).all(axis=1))[0][0]]
            fC = self.F[np.where((self.X == C).all(axis=1))[0][0]]
            return "ABC", r1 * fA + r2 * fB + r3 * fC
        else:
            r1, r2, r3 = self.barycentric_coordinates(C, D, A, self.y)
            if self.is_inside_triangle(r1, r2, r3):
                fC = self.F[np.where((self.X == C).all(axis=1))[0][0]]
                fD = self.F[np.where((self.X == D).all(axis=1))[0][0]]
                fA = self.F[np.where((self.X == A).all(axis=1))[0][0]]
                return "CDA", r1 * fC + r2 * fD + r3 * fA
            else:
                return "Outside triangles", np.nan

    def visualize(self):
        A, B, C, D = self.compute_points()
        
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.F, cmap='viridis', label='Points')
        plt.scatter(self.y[0], self.y[1], c='red', label='Point y')
        plt.scatter([A[0], B[0], C[0], D[0]], [A[1], B[1], C[1], D[1]], c='blue', label='Points A, B, C, D')
        
        plt.plot([A[0], B[0]], [A[1], B[1]], 'b-')
        plt.plot([B[0], C[0]], [B[1], C[1]], 'b-')
        plt.plot([C[0], A[0]], [C[1], A[1]], 'b-')
        
        plt.plot([C[0], D[0]], [C[1], D[1]], 'g-')
        plt.plot([D[0], A[0]], [D[1], A[1]], 'g-')
        
        plt.colorbar(label='Function values')
        plt.legend()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Barycentric Interpolation: Points and Triangles ABC and CDA')
        plt.show()

    def interpolate_set(self, Y):
        results = []
        for y in Y:
            self.y = np.array(y)
            triangle, interp_value = self.interpolate()
            true_value = self.func(self.y)
            results.append((y, triangle, interp_value, true_value))
            print(f"Point y: {y}, Triangle: {triangle}, Interpolated value: {interp_value}, True value: {true_value}")
        return results







