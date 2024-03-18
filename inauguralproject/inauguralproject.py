def square(x):
    """ square numpy array
    
    Args:
    
        x (ndarray): input array
        
    Returns:
    
        y (ndarray): output array
    
    """
    
    y = x**2
    return y


#Exchange economy from Project

import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

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
        #plt.plot(combinations_B[:, 0], combinations_B[:, 1], 'o', label='Individual B', linewidth = 0)
        plt.plot(self.par.w1A, self.par.w2A, 'ro', label='Endowment A')
        #plt.plot(self.par.w1B, self.par.w2B, 'bo', label='Endowment B')

        plt.xlabel('x_1^a')
        plt.ylabel('x_2^a')
        plt.title('Edgeworth Box')
        plt.legend()
        plt.show()







