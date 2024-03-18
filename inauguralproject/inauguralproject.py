#Exchange economy from Project

import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

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

#Defining utility functions and demand functions
    def utility_A(self,x1A,x2A):
        par = self.par 
        return x1A**par.alpha*x2A**(1-par.alpha)

    def utility_B(self,x1B,x2B):
        par = self.par 
        return x1B**par.beta*x2B**(1-par.beta)

    def demand_A(self,p1):
        par = self.par
        x1A = par.alpha * ((p1 * par.w1A + 1 * par.w2A) / (p1))
        x2A = (1 - par.alpha) * ((p1 * par.w1A + 1 * par.w2A) / (p1))
        return x1A, x2A

    def demand_B(self,p1):
        par = self.par
        x1B = par.beta * ((p1 * par.w1B + 1 * par.w2B) / (p1))
        x2B = (1 - par.beta) * ((p1 * par.w1B + 1 * par.w2B) / (p1))
        return x1B, x2B

#Defining functions for market clearing
    def check_market_clearing(self,p1):

        par = self.par

#Calculating the demands for both individuals for given prices of p1
        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

#Checking if the market is clearing by comparing the sum of demands and the endowments
        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2
    
#Defining two empty lists for the combination of consumption bundles    
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

# Add labels and title
        plt.xlabel('$x_1^A$')
        plt.ylabel('$x_2^A$')
        plt.title('Edgeworth Box')
        plt.legend()
        plt.show()







