#Exchange economy from Project

import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import numpy as np
from scipy import optimize
import itertools

#Defining the class ExchangeEconomyClass
class ExchangeEconomyClass:

    def __init__(self):

        par = self.par = SimpleNamespace()
        self.results = []  # Initialize results as an instance variable
        
##1
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

        #setting the price of good to as numeraire, hence p2 = 1.
        par.p2 = 1

    #Defining utility functions and demands
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


##2
# Method to calculate epsilon values (error term) for different prices of p1 in P1
#Firstly we define calc_eps, a class method that takes the number of iterations N as input
    def calc_eps(self, N):
        p1 = 0.5
        i = 1
        result = {'p1': [], 'eps1': [], 'eps2': []}  # Initialize an empty dictionary
        while p1 <= 2.5: # Loop over different values of p1 in the range [0.5, 2.5] with N intervals
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

## 3. Calculating the market clearing price
    def solve_3(self, p1_guess=1.0, tolerance=1e-3): # Set default values for p1_guess and tolerance
        clearing_prices = []
        min_combined_error = float('inf')
        price = None
        min_eps1 = None
        min_eps2 = None

        while p1_guess >= 0: # Loop over different values of p1 in the range [0, 1] with 75 intervals
            eps1, eps2 = self.check_market_clearing(p1_guess)
            if abs(eps1) < tolerance and abs(eps2) < tolerance: # Check if the market clears
                clearing_prices.append(p1_guess)
                combined_error = eps1**2 + eps2**2
                if combined_error < min_combined_error: # Check if the combined error is less than the minimum combined error found so far
                    min_combined_error = combined_error
                    price = p1_guess
                    min_eps1 = eps1
                    min_eps2 = eps2
            p1_guess -= 0.001 # Update the value of p1_guess

        # Print the results
        if clearing_prices:
            print(f"Minimum combined error: {min_combined_error} at price: {price:.5f}")
            print(f"Epsilon1: {min_eps1}, Epsilon2: {min_eps2}")
        else:
            print("No price found where the market clears.")

        # Saves the results in a dictionary, appending 
        iteration_3_results = {
            "Optimal Price for Consumer A": f"{price:.3f}"
        }
   
        # Append the results to the list "results" which we defined in the very start
        self.results.append(iteration_3_results)

        # Return the results dictionary in order to call the class in the jupiter notebook
        return iteration_3_results

## 4a. Find the allocation if only prices in P1
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

        #Unpack solution
        p1_case1 = sol_case1.x
        x1A_case1, x2A_case1 = self.demand_A(p1_case1)
        x1B_case1, x2B_case1 = self.demand_B(p1_case1)
        u = self.utility_A(x1A_case1, x2A_case1)
        print(f"x1A: {x1A_case1:.4f}, x2A: {x2A_case1:.4f}, x1B: {x1B_case1:.4f}, x2B: {x2B_case1:.4f}, p1: {p1_case1:.4f}, utility: {u:.4f}")

        return sol_case1
    
## 4b. Find the allocation if any positive price can be chosen
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
    
## 5a. Find the allocation if the choice set is restricted to C
    def solve_5a(self):
        par = self.par
        max_utility = -np.inf # initializing the utility to minus infinity, to ensure that finite values found are larger
        x1A_allocation = np.nan # initializing the allocations to NaN for the allocation of the goods
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


        iteration_5a_results = { # saving the results in a dictionary
            "Optimal Price for Consumer A": f"{price:.3f}",
            "Maximum Utility of Consumer A": f"{max_utility:.3f}",
            "Allocation of x1A": f"{x1A_allocation:.3f}",
            "Allocation of x2A": f"{x2A_allocation:.3f}",
            "Allocation of x1B": f"{x1B_allocation:.3f}",
            "Allocation of x2B": f"{x2B_allocation:.3f}"
        }
   
        # Append the results to the same list at earlier
        self.results.append(iteration_5a_results)

        # Returning the results in a dictionary for the jupyter notebook
        return iteration_5a_results
#5b. Find the allocation if no further restrictions are imposed
    def solve_5b(self): 
        par = self.par
        u_B_initial = np.nan  # Define the missing variable "u_B_initial"
        price = np.nan  # Define the missing variable "optimal_price"
        max_utility = -np.inf # initializing the utility to minus infinity, to ensure that finite values found are larger

        def util_pareto(x): # nested function of the pareto optimal problem
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

        # Calculate price that goes along with the allocation and the max utility
        max_utility = self.utility_A(optimal.x[0], optimal.x[1]) # calculate the maximum utility of consumer A

        price = par.alpha*par.w2A/(optimal.x[0]-par.alpha*par.w1A) # calculate the price

        iteration_5b_results = { # saving the results in a dictionary
            "Optimal Price for Consumer A": f"{price:.3f}",
            "Maximum Utility of Consumer A": f"{max_utility:.3f}",
            "Allocation of x1A": f"{x1A_allocation:.3f}",
            "Allocation of x2A": f"{x2A_allocation:.3f}",
            "Allocation of x1B": f"{x1B_allocation:.3f}",
            "Allocation of x2B": f"{x2B_allocation:.3f}"
        }

        # Appending the results to the list "results" yet again. 
        self.results.append(iteration_5b_results)

        # Returning the results in a dictionary for the jupyter notebook
        return iteration_5b_results
    
 # 6a. Allocation if consumption levels are chosen by a utilitarian social planner        
    def solve_6a(self): 
        par = self.par
        def util_planner(x): # nested function of social planner's max problem

            if 1 < x[0] < 0 or 1 < x[1] < 0: # check if x-values are within the interval [0,1]
                return 0 # if not, return 0

            return - ( self.utility_A(x[0], x[1]) + self.utility_B(1-x[0], 1-x[1])) # if they are, maximize the sum of utilities of consumer A and B

        bounds = ( (0,1) , (0,1) ) # bounds for the x-values

        # Initialize variables to store maximum utility and corresponding price
        optimal = optimize.minimize(util_planner, method='SLSQP',x0=[0.5,0.5],bounds=bounds) # maximizing the util_planner function with the initial guess of x-values [0.5,0.5]

        x1A_allocation = optimal.x[0] # store the optimal values
        x2A_allocation = optimal.x[1]
        x1B_allocation = 1 - optimal.x[0]
        x2B_allocation = 1 - optimal.x[1]

        max_utility = self.utility_A(optimal.x[0], optimal.x[1]) # calculate the maximum utility of consumer A

        # Calculate price corresponding to allcoation
        price = par.alpha*par.w2A/(optimal.x[0]-par.alpha*par.w1A)

        # Print the results
        iteration_6a_results = {
            "Optimal Price for Consumer A": f"{price:.3f}",
            "Maximum Utility of Consumer A": f"{max_utility:.3f}",
            "Allocation of x1A": f"{x1A_allocation:.3f}",
            "Allocation of x2A": f"{x2A_allocation:.3f}",
            "Allocation of x1B": f"{x1B_allocation:.3f}",
            "Allocation of x2B": f"{x2B_allocation:.3f}"
        }
         # Appending the results to the list "results" yet again.
        self.results.append(iteration_6a_results)

        # Returning the results in a dictionary for the jupyter notebook
        return iteration_6a_results
    
 # 6b. Illustrate and compare with other results       
    def solve_6b(self): 
        import pandas as pd

        # Access the results_list variable from the instance
        results_list = self.results

        # Create a DataFrame from results_list
        results_df = pd.DataFrame(results_list, columns=[
            "Optimal Price for Consumer A",
            "Maximum Utility of Consumer A",
            "Allocation of x1A",
            "Allocation of x2A",
            "Allocation of x1B",
            "Allocation of x2B"
        ])

        custom_index_labels = ["3", "5a", "5b", "6a"]  # Customize this list as needed

        results_df.index = custom_index_labels # Set the index of the DataFrame to the custom labels

        # Print the DataFrame
        display(results_df)


## 7. Draw a set W of 50 elements
    def solve_7(self):
        # Generate random endowment sets
        W = np.random.uniform(0, 1, size=(50, 2))  # draw a set W of 50 elements 

        # Print the elements of W
        print("Elements of W:")
        for i, (w1A, w2A) in enumerate(W, 1):
            print(f"Element {i}: w1A = {w1A:.4f}, w2A = {w2A:.4f}")

        # Print the number of elements in W
        print(f"\nThe number of elements in W is: {len(W)}")
        
        return W

## 8. Find market equilibrium allocation for each wA in W
    def solve_8(self, W):
        market_equilibrium_allocations_A_all = []
        market_equilibrium_allocations_B_all = []

        for omega_A in W:
            N = 75
            C = set()

            # Generate all possible combinations of x1 and x2
            for x1, x2 in itertools.product(range(N+1), repeat=2):
                x1 /= N
                x2 /= N

                # Check if the allocation satisfies the given conditions for the specified omega_A
                if (self.utility_A(x1, x2) >= self.utility_A(omega_A[0], omega_A[1]) and
                    self.utility_B(1 - x1, 1 - x2) >= self.utility_B(omega_A[0], omega_A[1])):

                    # Add the allocation to set C
                    C.add((x1, x2, 1 - x1, 1 - x2))

            market_equilibrium_allocations_A = []
            market_equilibrium_allocations_B = []

            for x1, x2, x1B, x2B in C:
                # Calculate the corresponding allocation for both consumers A and B
                x1A, x2A = x1, x2

                # Append the allocations
                market_equilibrium_allocations_A.append((x1A, x2A))
                market_equilibrium_allocations_B.append((x1B, x2B))

            # Append the market equilibrium allocations for the current omega_A to the lists
            market_equilibrium_allocations_A_all.append(market_equilibrium_allocations_A)
            market_equilibrium_allocations_B_all.append(market_equilibrium_allocations_B)

        # Plot the Edgeworth box with market equilibrium allocations for all omega_A values
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line representing the line of equal division
        plt.fill_between([0, 1], 0, 1, color='lightgray', alpha=0.5)  # Fill the box with light gray color
        
        for market_equilibrium_allocations_A, market_equilibrium_allocations_B in zip(market_equilibrium_allocations_A_all, market_equilibrium_allocations_B_all):
            for x1A, x2A in market_equilibrium_allocations_A:
                plt.plot(x1A, x2A, 'ro')  # A's allocations
            for x1B, x2B in market_equilibrium_allocations_B:
                plt.plot(x1B, x2B, 'bo')  # B's allocations

        plt.xlabel('Good 1')
        plt.ylabel('Good 2')
        plt.title('Edgeworth Box with Market Equilibrium Allocations for $\omega_A \in C$')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.show()

        return market_equilibrium_allocations_A_all, market_equilibrium_allocations_B_all
    
   


#8a. Caluclate the price for each element in W and find the allocation
    def solve_8a1(self, W):
        if W is None:
            raise ValueError("Set of prices W must be provided.")

        allocation_A_list = []  # Create empty lists to store allocations for consumers A and B
        allocation_B_list = []
        price_list = []  # Create an empty list to store the prices

        for w in W:  # Iterate over each element in the set W
            x1A, x2A, x1B, x2B = self.calculate_allocation(w)  # Calculate allocation for each price
            allocation_A_list.append((x1A, x2A))  # Append the allocation for consumer A to the list
            allocation_B_list.append((x1B, x2B))  # Append the allocation for consumer B to the list
            price = self.calculate_price(w)
            price_list.append(price)  # Append the price to the list

        # Print the results
        print("Allocations calculated for each price in W:")
        for w, allocation_A, allocation_B, price in zip(W, allocation_A_list, allocation_B_list, price_list):
            print(f"W: {w}, Price: {price}, Allocation A: {allocation_A[0]}, Allocation B: {allocation_B[0]}")

        return allocation_A_list, allocation_B_list

    
    def calculate_allocation(self, w):
        # Implement the calculation of the allocation for a given price w
        # Use the demand functions to calculate the allocation for consumers A and B
        p1 = w  # Assuming price of good 1 is given by the price list
        # Calculate demand for consumer A
        x1A, x2A = self.demand_A(p1)
        # Calculate demand for consumer B
        x1B, x2B = self.demand_B(p1)
        return x1A, x2A, x1B, x2B
    
    def calculate_price(self, w):
    # Implement the calculation of the price for a given w
    # You need to define the logic for calculating the price based on w
    # Replace the following line with your actual calculation
        return w * 0.9  # Placeholder calculation

#scatterplot the allocation
import matplotlib.pyplot as plt

def scatterplot_allocations(allocation_A_list, allocation_B_list, price_list):
    fig, ax = plt.subplots()
    for allocation_A, allocation_B, price in zip(allocation_A_list, allocation_B_list, price_list):
        ax.scatter(allocation_A[0], allocation_B[0], label=f'Price: {price:.2f}')  # Plot the allocation for consumer A and B
    ax.set_xlabel('Allocation of x1A')
    ax.set_ylabel('Allocation of x1B')
    ax.set_title('Scatterplot of Allocations')
    ax.legend()
    plt.show()
    return

