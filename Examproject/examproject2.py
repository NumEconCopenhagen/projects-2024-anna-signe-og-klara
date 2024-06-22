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
                    results.append((p1, p2, True))
                else:
                    results.append((p1, p2, False))
        return results
    
    def find_equilibrium_prices(self):
        results = self.check_market_clearing()
        for result in results:
            if result[2]:
                self.p1, self.p2 = result[0], result[1]
                break
        print(f"Equilibrium prices: p1 = {economy.p1}, p2 = {economy.p2}")
        return self.p1, self.p2
        
    
    def optimize_social_welfare(self):
        self.find_equilibrium_prices()
        if self.p1 is None or self.p2 is None:
            return None
        c1_star = self.optimal_production(self.w, self.p1)
        c2_star = self.optimal_production(self.w, self.p2)
        l_star = self.optimal_labor(self.w, self.p1) + self.optimal_labor(self.w, self.p2)
        utility = self.consumer_utility(c1_star, c2_star, l_star)
        swf = utility - self.par.kappa * c2_star
        print(f"Social welfare function value: {economy.optimize_social_welfare()}")
        return swf

# Example usage
economy = ProductionEconomyClass()
print(f"Equilibrium prices: p1 = {economy.p1}, p2 = {economy.p2}")
print(f"Social welfare function value: {economy.optimize_social_welfare()}")
