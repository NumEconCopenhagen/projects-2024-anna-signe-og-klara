#Exchange economy from Project

import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import numpy as np
from scipy import optimize

#Defining the class ExchangeEconomyClass
class ExchangeEconomyClass:
    results = []

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
        par = self.par
        max_utility = -np.inf # initializing the utility to minus infinity, to ensure that finite values found are larger
        optimal_price = np.nan # initializing the rest of the variables to 'Not a Number' 
        x1A_allocation = np.nan
        x2A_allocation = np.nan
        x1B_allocation = np.nan
        x2B_allocation = np.nan

        x_values = np.linspace(0, 1, 75) # defining the range of x-values from 0 to 1 with 75 intervals
        for x1 in x_values: # looping over all combinations of x1 and x2 within the defined range
            for x2 in x_values:
                utility_A = self.utility_A(x1, x2)
                utility_B = self.utility_B(1 - x1, 1 - x2) # restricting the x-values for consumer B according to choice set C
                price = par.alpha * par.w2A / (x1 - par.alpha * par.w1A) # calculating the price

                if utility_A > max_utility and utility_B >= self.utility_B(par.w1B, par.w2B): # checking if the utility is larger than the maximum utility found so far and if the utility of consumer B is larger than the utility of the initial endowment
                    max_utility = utility_A # when conditions are met, the utility is updated
                    optimal_price = price # the price and allocations are updated
                    x1A_allocation = x1 
                    x2A_allocation = x2
                    x1B_allocation = 1 - x1
                    x2B_allocation = 1 - x2

         # Save results for later use
        results_5a = {
            "Optimal Price for Consumer A": price,
            "Maximum Utility of Consumer A": max_utility,
            "Allocation of x1A": x1A_allocation,
            "Allocation of x2A": x2A_allocation,
            "Allocation of x1B": x1B_allocation,
            "Allocation of x2B": x2B_allocation
        }
        self.results.append(results_5a)            

        iteration_5a_results = {
            "Optimal Price for Consumer A": f"{optimal_price:.3f}",
            "Maximum Utility of Consumer A": f"{max_utility:.3f}",
            "Allocation of x1A": f"{x1A_allocation:.3f}",
            "Allocation of x2A": f"{x2A_allocation:.3f}",
            "Allocation of x1B": f"{x1B_allocation:.3f}",
            "Allocation of x2B": f"{x2B_allocation:.3f}"
        }
        return iteration_5a_results
    
    def solve_5b(self):
        par = self.par
        u_B_initial = np.nan  # Define the missing variable "u_B_initial"
        optimal_price = np.nan  # Define the missing variable "optimal_price"
        max_utility = np.nan  # Define the missing variable "max_utility"
        results = []  # Define the missing variable "results"

        def util_pareto(x): 
            if 1 < x[0] < 0 or 1 < x[1] < 0: # the function takes a list x of two elements as input and checks if the elements are within the interval [0,1]
                return 0 # if the elements are not within the interval, the function returns 0
            return -self.utility_A(x[0], x[1]) # if the elements are within the interval, the function returns the negative utility of consumer A

        constraint = ({'type': 'ineq', 'fun': lambda x: self.utility_B(1-x[0], 1-x[1]) - u_B_initial}) # defining the constraint, that the utility of consumer B must be larger than the endownment
        bounds = ( (0,1) , (0,1) ) # defining the bounds for the x-values as two tuples

        # Initialize variables to store maximum utility and corresponding price
        optimal = optimize.minimize(util_pareto, constraints=constraint, method='SLSQP',x0=[0.5,0.5],bounds=bounds) # maximizing the util_pareto function with the initial guess of x-values [0.5,0.5] 
        
        x1A_allocation = optimal.x[0] # the optimal values calculated are stored
        x2A_allocation = optimal.x[1]
        x1B_allocation = 1 - optimal.x[0]
        x2B_allocation = 1 - optimal.x[1]

        # Calculate price that goes along with the allocation
        price = par.alpha*par.w2A/(optimal.x[0]-par.alpha*par.w1A)

      
        iteration_5b_results = {

            # Print the optimal price and maximum utility
            "Optimal Price:": price,
            "Maximum Utility of Consumer A:": self.utility_A(optimal.x[0], optimal.x[1]),
            # Print the allocations
            "Allocation of goods for Consumer A:"
            "Good 1:": x1A_allocation,
            "Good 2:": x2A_allocation,

            "Allocation of goods for Consumer B:"
            "Good 1:": x1B_allocation,
            "Good 2:": x2B_allocation,
        }
        return iteration_5b_results

 # 6a. Allocation if consumption levels are chosen by a utilitarian social planner        
    def solve_6a(self):
        par = self.par
        u_B_initial = np.nan  # define the missing variable "u_B_initial"
        def util_planner(x): # nested function of social planner's max problem

            if 1 < x[0] < 0 or 1 < x[1] < 0: # check if x-values are within the interval [0,1]
                return 0 # if not, return 0

            return - ( self.utility_A(x[0], x[1]) + self.utility_B(1-x[0], 1-x[1])) # if they are, maximize the sum of utilities of consumer A and B

        constraint = ({'type': 'ineq', 'fun': lambda x: self.utility_B(1-x[0], 1-x[1]) - u_B_initial}) # constraint that the utility of consumer B must be larger than the endowment
        bounds = ( (0,1) , (0,1) ) # bounds for the x-values

        # Initialize variables to store maximum utility and corresponding price
        optimal = optimize.minimize(util_planner, constraints=constraint, method='SLSQP',x0=[0.5,0.5],bounds=bounds) # maximizing the util_planner function with the initial guess of x-values [0.5,0.5]

        x1A_allocation = optimal.x[0] # store the optimal values
        x2A_allocation = optimal.x[1]
        x1B_allocation = 1 - optimal.x[0]
        x2B_allocation = 1 - optimal.x[1]

        max_utility = self.utility_A(optimal.x[0], optimal.x[1]) # calculate the maximum utility of consumer A

        # Calculate price corresponding to allcoation
        price = par.alpha*par.w2A/(optimal.x[0]-par.alpha*par.w1A)

         # Save results for later use
        results_6b = {
            "Optimal Price for Consumer A": price,
            "Maximum Utility of Consumer A": max_utility,
            "Allocation of x1A": x1A_allocation,
            "Allocation of x2A": x2A_allocation,
            "Allocation of x1B": x1B_allocation,
            "Allocation of x2B": x2B_allocation
        }
        self.results.append(results_6b)

        # Print the results
        iteration_6a_results = {
            "Optimal Price for Consumer A": price,
            "Maximum Utility of Consumer A": max_utility,
            "Allocation of x1A": x1A_allocation,
            "Allocation of x2A": x2A_allocation,
            "Allocation of x1B": x1B_allocation,
            "Allocation of x2B": x2B_allocation
        }
        return iteration_6a_results
    
 # 6b. Illustrate and compare with other results       
    def solve_6b(self):
        import pandas as pd

        # Create an empty DataFrame
        df = pd.DataFrame(self.results)

        # Your existing code
        index_labels = ['3', '4a', '4b', '5a', '5b', '6a'] 
        df.index = index_labels
        df_rounded = df.round(3)
        # Print the DataFrame

        iteration_6a_results = {
        print(df_rounded)
        }
        return iteration_6a_results

    # 7. Draw a set W of 50 elements
    def solve_7(self):
        par = self.par
        W = np.random.uniform(0, 1, size=(50, 2)) # draw a set W of 50 elements 
        num_elements = W.shape[0] # define the number of elements in W
        for i in range(num_elements): # loop over the elements in W
            w1A = W[i, 0] # define w1A as the first element in the i-th row of W
            w2A = W[i, 1] # define w2A as the second element in the i-th row of W
        # checking the number of elements in W
        print(f"The number of elements in W is: {num_elements}")

# 8. Find market equilibrium allocation for each wA in W
