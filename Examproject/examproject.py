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
