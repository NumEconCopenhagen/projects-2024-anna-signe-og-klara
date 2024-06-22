import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy import optimize
import itertools

class ProductionEconomyClass:

    def __init__(self):
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
        
    def optimal_labor(self, w, p):
        return ((self.par.A * self.par.gamma * p) / w) ** (1 / (1 - self.par.gamma))
    
    def optimal_production(self, w, p):
        return self.par.A * (self.optimal_labor(w, p) ** self.par.gamma)
    
    def profit(self, w, p):
        return ((1 - self.par.gamma) / self.par.gamma) * w * ((p * self.par.A * self.par.gamma / w) ** (1 / (1 - self.par.gamma)))
    
    def consumer_utility(self, c1, c2, l):
        return np.log(c1 ** self.par.alpha * c2 ** (1 - self.par.alpha)) - self.par.nu * (l ** (1 + self.par.epsilon)) / (1 + self.par.epsilon)
    
    def check_market_clearing(self):
        results = []
        self.p1_range = np.linspace(0.1, 2.0, 10)
        self.p2_range = np.linspace(0.1, 2.0, 10)
        for p1 in self.p1_range:
            for p2 in self.p2_range:
                l_star = self.optimal_labor(self.w, p1) + self.optimal_labor(self.w, p2)
                c1_star = self.optimal_production(self.w, p1)
                c2_star = self.optimal_production(self.w, p2)
                
                # Check labor and goods markets
                labor_market_cleared = np.isclose(l_star, self.optimal_labor(self.w, p1) + self.optimal_labor(self.w, p2))
                good_market_1_cleared = np.isclose(c1_star, self.optimal_production(self.w, p1))
                good_market_2_cleared = np.isclose(c2_star, self.optimal_production(self.w, p2))
                
                if labor_market_cleared and good_market_1_cleared and good_market_2_cleared:
                    results.append((round(p1, 2), round(p2, 2), True))
                else:
                    results.append((round(p1, 2), round(p2, 2), False))
        return results

    def find_equilibrium_prices(self):
        results = self.check_market_clearing()
        for result in results:
            if result[2]:
                self.p1, self.p2, _, = result
                print(f"Equilibrium prices: p1 = {self.p1}, p2 = {self.p2}")
                return self.p1, self.p2
        return None, None
    
    def c1(self, l):
        return self.par.alpha * (self.w * l + self.par.T + self.profit(self.w, self.p1) + self.profit(self.w, self.p2)) / self.p1
    
    def c2(self, l):
        return (1 - self.par.alpha) * (self.w * l + self.par.T + self.profit(self.w, self.p1) + self.profit(self.w, self.p2)) / (self.p2 + self.par.tau)
    
    def optimal_behavior(self):
        objective = lambda l: -(self.par.alpha * np.log(self.c1(l)) + (1 - self.par.alpha) * np.log(self.c2(l)) - self.par.nu * (l ** (1 + self.par.epsilon)) / (1 + self.par.epsilon))
        result = optimize.minimize_scalar(objective, bounds=(0, 10), method='bounded')
        if result.success:
            return result.x
        else:
            raise ValueError("Optimization failed")
        
    def consumption_without_tax(self):
        # Set tau and T to zero explicitly for this calculation
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
      
        
    def social_welfare(self):
        l_star = self.optimal_behavior()
        c1_star = self.c1(l_star)
        c2_star = self.c2(l_star)
        utility = self.consumer_utility(c1_star, c2_star, l_star)
        swf = utility - self.par.kappa * c2_star
        return swf

    def swf_with_tax(self, tau):
        self.par.tau = tau
        self.par.T = self.par.tau * self.c2(self.optimal_behavior())
        return -self.social_welfare()  # Minimize negative SWF to maximize SWF

    def find_optimal_tax(self):
        result = optimize.minimize_scalar(self.swf_with_tax, bounds=(0, 2), method='bounded')
        if result.success:
            self.par.tau = result.x
            self.par.T = self.par.tau * self.c2(self.optimal_behavior())
        else:
            raise ValueError("Optimization failed")
    
        # Find the optimal labor for calculating consumption without tax
        l_star = self.optimal_behavior()
        

        # Calculate the consumption values 
        c1_star_tax = self.c1(l_star)
        c2_star_tax = self.c2(l_star)

        # Calculate consumption of c1 and c2 without the tax
          
        # Print the results
        print(f"Consumption of c1 with tax: {c1_star_tax}")
        print(f"Consumption of c2 with tax: {c2_star_tax}")
        print(f"Optimal tax rate: {self.par.tau}")
        print(f"Social welfare: {-result.fun}")
        return c1_star_tax, c2_star_tax, self.par.tau, -result.fun
    
import numpy as np
from types import SimpleNamespace

class CareerChoiceSimulation:
    def __init__(self, N=10):
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
            expected_utilities[j] = v_known[j] + (1 / self.par.K) * np.sum(epsilons)
            
            # Calculate average realized utility for career track j
            average_realized_utilities[j] = np.mean(utilities)
        
        return expected_utilities, average_realized_utilities

    def simulate_career_choice(self):
        # Simulate career choice and calculate prior expected utility and realized utility
        chosen_careers = np.zeros(self.par.N, dtype=int)
        prior_expected_utilities = np.zeros((self.par.N, self.par.J))
        realized_utilities = np.zeros(self.par.N)
        
        for i in range(self.par.N):
            Fi = self.par.F[i]
            epsilons_friends = np.random.normal(loc=0, scale=self.par.sigma, size=(self.par.J * Fi, self.par.K))
            epsilons_self = np.random.normal(loc=0, scale=self.par.sigma, size=(self.par.J, self.par.K))
            
            estimated_v_j = np.mean(epsilons_friends.reshape(Fi, self.par.J, self.par.K), axis=0).mean(axis=1)
            prior_expected_utilities[i] = estimated_v_j
            
            # Calculate probabilities using softmax of estimated utilities
            probabilities = np.exp(estimated_v_j - np.max(estimated_v_j))
            probabilities /= probabilities.sum()
            chosen_careers[i] = np.random.choice(self.par.J, p=probabilities)
            
            # Calculate realized utility for the chosen career track
            realized_utilities[i] = estimated_v_j[chosen_careers[i]] + np.mean(epsilons_self[chosen_careers[i]])

        return chosen_careers, prior_expected_utilities, realized_utilities
