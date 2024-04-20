#Exchange economy from Project

import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import numpy as np
from scipy import optimize

#Defining the class ExchangeEconomyClass
class ExchangeEconomyClass:
    def __init__(self):

        par = self.par = SimpleNamespace()

#Defining parameters and endowments
    
        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3

        # Total endowments
        par.w1B = 1 - par.w1A
        par.w2B = 1 - par.w2A

        par.p2 = 1

    def utility_A(self, x1A, x2A):
        par = self.par 
        return x1A**(par.alpha) * x2A**(1 - par.alpha)

    def utility_B(self, x1B, x2B):
        par = self.par 
        return x1B**(par.beta) * x2B**(1 - par.beta)

    def demand_A(self, p1):
        par = self.par
        x1A = par.alpha * ((p1 * par.w1A + 1 * par.w2A) / (p1))
        x2A = (1 - par.alpha) * ((p1 * par.w1A + 1 * par.w2A) / (par.p2))
        return x1A, x2A

    def demand_B(self, p1):
        par = self.par
        x1B = par.beta * ((p1 * par.w1B + 1 * par.w2B) / (p1))
        x2B = (1 - par.beta) * ((p1 * par.w1B + 1 * par.w2B) / (par.p2))
        return x1B, x2B

#Defining functions for market clearing
    def check_market_clearing(self, p1):

        par = self.par

# Calculating the demands for both individuals for given prices of p1
        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

# Checking if the market is clearing by comparing the sum of demands and the endowments
        eps1 = x1A - par.w1A + x1B - (1 - par.w1A)
        eps2 = x2A - par.w2A + x2B - (1 - par.w2A)

        return eps1, eps2
    
    def plot_edgeworth_box(self, N):
        combinations_A = []
        combinations_B = []

# Generate N evenly spaced values between 0 and 1 with 75 intervals
        x_values = np.linspace(0, 1, 75)

# Loop over all combinations of x1A and x2A in order to find the utility-maximizing allocation for individual A
        for x1A in x_values:
            for x2A in x_values:
                # Check conditions for allocation (x1A, x2A) for individual A
                if self.utility_A(x1A, x2A) > self.utility_A(self.par.w1A, self.par.w2A):
                    # Calculate corresponding x1B and x2B for individual B
                    x1B = 1 - x1A
                    x2B = 1 - x2A
                    # Check conditions for allocation (x1B, x2B) for individual B
                    if self.utility_B(x1B, x2B) > self.utility_B(self.par.w1B, self.par.w2B):
                        combinations_A.append([x1A, x2A])
                        combinations_B.append([x1B, x2B])

# Convert lists to arrays
        combinations_A = np.array(combinations_A)
        combinations_B = np.array(combinations_B)

# Plot the Edgeworth box
        plt.plot(combinations_A[:, 0], combinations_A[:, 1], 'o', label='Individual A', linewidth = 0)
        plt.plot(self.par.w1A, self.par.w2A, 'ro', label='Endowment A')
    
        plt.xlabel('x_1^a')
        plt.ylabel('x_2^a')
        plt.title('Edgeworth Box')
        plt.legend()
        plt.show()

    def calc_eps(self, N):
        p1 = 0.5
        i = 1
        result = {'p1': [], 'eps1': [], 'eps2': []}  # Initialize an empty dictionary
        while p1 <= 2.5:
            eps1, eps2 = self.check_market_clearing(p1)
            print(f"For p1 = {p1:.2f}: epsilon1 = {eps1:.4f} and epsilon2 = {eps2:.4f}")
            result['p1'].append(p1)  # Append the value of p1 to the list associated with the 'p1' key
            result['eps1'].append(eps1)  # Append the value of eps1 to the list associated with the 'eps1' key
            result['eps2'].append(eps2)  # Append the value of eps2 to the list associated with the 'eps2' key
            i += 1
            p1 = 0.5 + 2 * i / N
        return result  # Return the dictionary

    def plot_eps(self, N):    
        # Get the values of p1, eps1, and eps2
        p1_values = self.calc_eps(N = 75)['p1']
        eps1_values = self.calc_eps(N = 75)['eps1']
        eps2_values = self.calc_eps(N = 75)['eps2']

        # Plot eps1 and eps2 as a function of p1
        plt.plot(p1_values, eps1_values, label = 'eps1')
        plt.plot(p1_values, eps2_values, label = 'eps2')

        # Add labels and legend
        plt.xlabel('p1')
        plt.ylabel('eps')
        plt.legend()

        # Show the plot
        plt.show()

## 4a
    #objective function 
    def find_prices_4a(self, p1):
        par = self.par
        x1A, x2A = self.demand_A(p1)
        x1B, x2B = self.demand_B(p1)
        return -self.utility_A(1 - x1B, 1 - x2B)
    
    #call solver
    def solve_4a(self):
        sol_case1 = optimize.minimize_scalar(
            self.find_prices_4a, 
            method='bounded',
            bounds=(0.5, 2.5))

        #unpack solution
        p1_case1 = sol_case1.x
        x1A_case1, x2A_case1 = self.demand_A(p1_case1)
        x1B_case1, x2B_case1 = self.demand_B(p1_case1)
        u = self.utility_A(x1A_case1, x2A_case1)
        print(f"x1A: {x1A_case1:.4f}, x2A: {x2A_case1:.4f}, x1B: {x1B_case1:.4f}, x2B: {x2B_case1:.4f}, p1: {p1_case1:.4f}, utility: {u:.4f}")

        return sol_case1
    
## 4b
    def find_prices_4b(self, p1):
        par = self.par
        x1A, x2A = self.demand_A(p1)
        x1B, x2B = self.demand_B(p1)
        return -self.utility_A(1 - x1B, 1 - x2B)
    
    #call solver
    def solve_4b(self):
        sol_case2 = optimize.minimize_scalar(
            self.find_prices_4b, 
            method='bounded',
            bounds=(0, 10))

        #unpack solution
        p1_case2 = sol_case2.x
        x1A_case2,x2A_case2 = self.demand_A(p1_case2)
        x1B_case2,x2B_case2 = self.demand_B(p1_case2)
        u = self.utility_A(x1A_case2, x2A_case2)
        print(f"x1A: {x1A_case2:.4f}, x2A: {x2A_case2:.4f}, x1B: {x1B_case2:.4f}, x2B: {x2B_case2:.4f}, p1: {p1_case2:.4f}, utility: {u:.4f}")

        return sol_case2
    
## 5a
    def solve_5a(self):
        max_utility = -np.inf
        optimal_price = np.nan
        x1A_allocation = np.nan
        x2A_allocation = np.nan
        x1B_allocation = np.nan
        x2B_allocation = np.nan

        x_values = np.linspace(0, 1, 75)
        for x1 in x_values:
            for x2 in x_values:
                utility_A = self.utility_A(x1, x2)
                utility_B = self.utility_B(1 - x1, 1 - x2)
                price = self.par.alpha * self.par.w2A / (x1 - self.par.alpha * self.par.w1A)

                if utility_A > max_utility and utility_B >= self.utility_B(self.par.w1B, self.par.w2B):
                    max_utility = utility_A
                    optimal_price = price
                    x1A_allocation = x1
                    x2A_allocation = x2
                    x1B_allocation = 1 - x1
                    x2B_allocation = 1 - x2

        iteration_5a_results = {
            "Optimal Price for Consumer A": f"{optimal_price:.3f}",
            "Maximum Utility of Consumer A": f"{max_utility:.3f}",
            "Allocation of x1A": f"{x1A_allocation:.3f}",
            "Allocation of x2A": f"{x2A_allocation:.3f}",
            "Allocation of x1B": f"{x1B_allocation:.3f}",
            "Allocation of x2B": f"{x2B_allocation:.3f}"
        }
        return iteration_5a_results