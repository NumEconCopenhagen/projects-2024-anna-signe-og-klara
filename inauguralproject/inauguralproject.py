#Exchange economy from Project

import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import numpy as np

class ExchangeEconomyClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

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

    def utility_A(self,x1A,x2A):
        par = self.par 
        return x1A**(par.alpha)*x2A**(1-par.alpha)

    def utility_B(self,x1B,x2B):
        par = self.par 
        return x1B**(par.beta)*x2B**(1-par.beta)

    def demand_A(self,p1):
        par = self.par
        x1A = par.alpha * ((p1 * par.w1A + 1 * par.w2A) / (p1))
        x2A = (1 - par.alpha) * ((p1 * par.w1A + 1 * par.w2A) / (par.p2))
        return x1A, x2A

    def demand_B(self,p1):
        par = self.par
        x1B = par.beta * ((p1 * par.w1B + 1 * par.w2B) / (p1))
        x2B = (1 - par.beta) * ((p1 * par.w1B + 1 * par.w2B) / (par.p2))
        return x1B, x2B

    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2

    def plot_edgeworth_box(self, N):
        combinations_A = []
        combinations_B = []

        # Generate N evenly spaced values between 0 and 1
        x_values = np.linspace(0, 1, 75)

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

        combinations_A = np.array(combinations_A)
        combinations_B = np.array(combinations_B)

        plt.plot(combinations_A[:, 0], combinations_A[:, 1], 'o', label='Individual A', linewidth = 0)
        plt.plot(self.par.w1A, self.par.w2A, 'ro', label='Endowment A')
    
        plt.xlabel('x_1^a')
        plt.ylabel('x_2^a')
        plt.title('Edgeworth Box')
        plt.legend()
        plt.show()

    def test_eps(self, N):
        P1 = np.linspace(0.5, 2.5, N+1)

        eps1 = []
        eps2 = []
        p_1 = []

        for p1 in P1:
            demandA1, demandA2 = self.demand_A(p1)
            demandB1, demandB2 = self.demand_B(p1)

            eps1.append(demandA1 + demandB1 - self.par.w1A - self.par.w1B)
            eps2.append(demandB1 + demandB2 - self.par.w2A - self.par.w2B)
            p_1.append(p1)   

        for price, error1, error2 in zip(p_1, eps1, eps2):
            sign1 = '+' if error1 >= 0 else '-'
            sign2 = '+' if error2 >= 0 else '-'

            print(f'For p1 = {price:.2f} epsilon1 = {sign1}{abs(error1):.4f} and epsilon2 = {sign2}{abs(error2):.4f}')

    
    
    def calc_eps(self, N):
        p1 = 0.5
        i = 1
        result = {'p1': [], 'eps1': [], 'eps2': []}  # Initialize an empty dictionary
        while p1 <= 2.5:
            eps1, eps2 = self.check_market_clearing(p1)
            print(f"For p1 = {p1}, eps1 = {eps1}, eps2 = {eps2}")
            result['p1'].append(p1)  # Append the value of p1 to the list associated with the 'p1' key
            result['eps1'].append(eps1)  # Append the value of eps1 to the list associated with the 'eps1' key
            result['eps2'].append(eps2)  # Append the value of eps2 to the list associated with the 'eps2' key
            i += 1
            p1 = 0.5 + 2 * i / N
        return result  # Return the dictionary

    def plot_eps(self, N):    
        # Get the values of p1, eps1, and eps2
        p1_values = self.calc_eps(N=75)['p1']
        eps1_values = self.calc_eps(N=75)['eps1']
        eps2_values = self.calc_eps(N=75)['eps2']

        # Plot eps1 and eps2 as a function of p1
        plt.plot(p1_values, eps1_values, label='eps1')
        plt.plot(p1_values, eps2_values, label='eps2')

        # Add labels and legend
        plt.xlabel('p1')
        plt.ylabel('eps')
        plt.legend()

        # Show the plot
        plt.show()

    

