import numpy as np
from scipy.optimize import minimize
import sympy as sp
import matplotlib.pyplot as plt

class HOmodelClass:
    def __init__(self):
        self.alpha_DK_W = 0.7  # Capital elasticity in Denmark for Windmills
        self.beta_DK_W = 0.3   # Labor elasticity in Denmark for Windmills
        self.alpha_CN_T = 0.3  # Capital elasticity in China for Textiles
        self.beta_CN_T = 0.7   # Labor elasticity in China for Textiles
        self.K_DK, self.L_DK = 100, 50  # Denmark's capital and labor
        self.K_CN, self.L_CN = 50, 100  # China's capital and labor
        self.rho = 0.5  # Elasticity of substitution in utility

    def production(self, K, L, alpha, beta): # Production function
        return K**alpha * L**beta # Cobb-Douglas production function

    def utility(self, C1, C2, rho): # Utility function
        return C1**rho + C2**rho # Constant relative risk aversion utility function

    def constraints(self): # Constraints for optimization
        return [
            {'type': 'ineq', 'fun': lambda x: self.K_DK - x[0] - x[2]}, # Capital constraint for Denmark
            {'type': 'ineq', 'fun': lambda x: self.L_DK - x[1] - x[3]}, # Labor constraint for Denmark
            {'type': 'ineq', 'fun': lambda x: self.K_CN - x[4] - x[6]}, # Capital constraint for China
            {'type': 'ineq', 'fun': lambda x: self.L_CN - x[5] - x[7]}, # Labor constraint for China
            {'type': 'ineq', 'fun': lambda x: x}  # Non-negativity
        ]

    def objective(self, x):
        # Update the objective to account for production of both goods in both countries
        prod_DK_W = self.production(x[0], x[1], self.alpha_DK_W, self.beta_DK_W) # Production of Windmills in Denmark
        prod_DK_T = self.production(x[2], x[3], self.alpha_CN_T, self.beta_CN_T) # Production of Textiles in Denmark
        prod_CN_W = self.production(x[4], x[5], self.alpha_DK_W, self.beta_DK_W) # Production of Windmills in China
        prod_CN_T = self.production(x[6], x[7], self.alpha_CN_T, self.beta_CN_T) # Production of Textiles in China
        U_DK = self.utility(prod_DK_W + prod_CN_W, prod_DK_T + prod_CN_T, self.rho) # Utility for Denmark
        U_CN = self.utility(prod_CN_T + prod_CN_W, prod_DK_T + prod_DK_W, self.rho) # Utility for China
        return -(U_DK + U_CN) # Objective is to maximize total utility

    def run_optimization(self): # Run optimization
        initial_guess = [self.K_DK/2, self.L_DK/2, self.K_DK/2, self.L_DK/2, self.K_CN/2, self.L_CN/2, self.K_CN/2, self.L_CN/2] # Initial guess
        result = minimize(self.objective, initial_guess, constraints=self.constraints(), method='SLSQP') # Optimization
        if result.success: # If optimization is successful
            x = result.x
            # Calculate production based on optimized resource allocations
            prod_DK_W = self.production(x[0], x[1], self.alpha_DK_W, self.beta_DK_W)
            prod_DK_T = self.production(x[2], x[3], self.alpha_CN_T, self.beta_CN_T)
            prod_CN_W = self.production(x[4], x[5], self.alpha_DK_W, self.beta_DK_W)
            prod_CN_T = self.production(x[6], x[7], self.alpha_CN_T, self.beta_CN_T)
            
            allocations = { # Resource allocations and production in each country
                'DK': {'K_Windmills': x[0], 'L_Windmills': x[1], 'K_Textiles': x[2], 'L_Textiles': x[3],
                    'Production_Windmills': prod_DK_W, 'Production_Textiles': prod_DK_T},
                'CN': {'K_Windmills': x[4], 'L_Windmills': x[5], 'K_Textiles': x[6], 'L_Textiles': x[7],
                    'Production_Windmills': prod_CN_W, 'Production_Textiles': prod_CN_T}
            }
            U_DK, U_CN = self.separate_utilities(x)
            return allocations, U_DK, U_CN
        else:
            raise Exception("Optimization failed: " + result.message)

    def before_trade_old(self):
        prod_DK_W = self.production(self.K_DK, self.L_DK, self.alpha_DK_W, self.beta_DK_W)
        prod_CN_T = self.production(self.K_CN, self.L_CN, self.alpha_CN_T, self.beta_CN_T)
        U_DK = self.utility(prod_DK_W, 0, self.rho)  # Assuming only Windmills contribute to utility in Denmark
        U_CN = self.utility(0, prod_CN_T, self.rho)  # Assuming only Textiles contribute to utility in China

        return {
            'DK': {'K_Windmills': self.K_DK, 'L_Windmills': self.L_DK, 'K_Textiles': 0, 'L_Textiles': 0,
                'Production_Windmills': prod_DK_W, 'Production_Textiles': 0},
            'CN': {'K_Windmills': 0, 'L_Windmills': 0, 'K_Textiles': self.K_CN, 'L_Textiles': self.L_CN,
                'Production_Windmills': 0, 'Production_Textiles': prod_CN_T}
        }, U_DK, U_CN
    

    # ÆNDRET HER --> for at få parameterværdier til at ændre!!!!
    def before_trade(self):
        # Calculate production quantities
        prod_DK_W = self.production(self.K_DK, self.L_DK, self.alpha_DK_W, self.beta_DK_W)
        prod_CN_T = self.production(self.K_CN, self.L_CN, self.alpha_CN_T, self.beta_CN_T)
        
        # Calculate utility for Denmark
        U_DK = self.utility(prod_DK_W, 0, self.rho)
        
        # Calculate utility for China
        U_CN = self.utility(0, prod_CN_T, self.rho)

        return {
            'DK': {'K_Windmills': self.K_DK, 'L_Windmills': self.L_DK, 'K_Textiles': 0, 'L_Textiles': 0,
                'Production_Windmills': prod_DK_W, 'Production_Textiles': 0},
            'CN': {'K_Windmills': 0, 'L_Windmills': 0, 'K_Textiles': self.K_CN, 'L_Textiles': self.L_CN,
                'Production_Windmills': 0, 'Production_Textiles': prod_CN_T}
        }, U_DK, U_CN
    # !!!!!!!!!!!!!!!


    def separate_utilities(self, x):
        # Re-calculate production based on optimized values
        prod_DK_W = self.production(x[0], x[1], self.alpha_DK_W, self.beta_DK_W)
        prod_DK_T = self.production(x[2], x[3], self.alpha_CN_T, self.beta_CN_T)
        prod_CN_W = self.production(x[4], x[5], self.alpha_DK_W, self.beta_DK_W)
        prod_CN_T = self.production(x[6], x[7], self.alpha_CN_T, self.beta_CN_T)
        # Calculate utility for each country
        U_DK = self.utility(prod_DK_W + prod_CN_W, prod_DK_T + prod_CN_T, self.rho) # Utility for Denmark
        U_CN = self.utility(prod_CN_T + prod_CN_W, prod_DK_T + prod_DK_W, self.rho) # Utility for China
        return U_DK, U_CN
    
    def plot_trade_results(self):
        # Get the results before trade
        before_trade_results, U_DK_before, U_CN_before = self.before_trade()

        # Get the results after trade
        try:
            allocations, U_DK, U_CN = self.run_optimization()
        except Exception as e:
            print(e)
            allocations = None

        if allocations:
            # Data Preparation
            labels = ['Denmark', 'China']
            utility = [U_DK_before, U_DK, U_CN_before, U_CN]
            
            x = np.arange(len(labels))  # Label locations
            width = 0.35  # Width of the bars
            offset = width / 2  # Offset for side-by-side bar placement

            fig, ax = plt.subplots(figsize=(10, 5))

            # Utility Plot
            ax.bar(x - offset, utility[::2], width, label='Utility Before Trade')
            ax.bar(x + offset, utility[1::2], width, label='Utility After Trade', alpha=0.5)

            ax.set_ylabel('Utility')
            ax.set_title('Utility by Country Before and After Trade')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()

            plt.tight_layout()
            plt.show()
    

    def plot_parameter_effect_combined(self, parameter_values, parameter_type, result_type='utility'):
        results_DK = []
        results_CN = []
        for value in parameter_values:
            if parameter_type == 'alpha':
                self.alpha_DK_W = value
                self.alpha_CN_T = value
            elif parameter_type == 'beta':
                self.beta_DK_W = value
                self.beta_CN_T = value
            _, U_DK, U_CN = self.before_trade()
            results_DK.append(U_DK)
            results_CN.append(U_CN)
        
        plt.figure(figsize=(8, 6))
        plt.plot(parameter_values, results_DK, label=f'Denmark {parameter_type}')
        plt.plot(parameter_values, results_CN, label=f'China {parameter_type}')
        plt.xlabel(f'{parameter_type} values')
        plt.ylabel('Utility')
        plt.title(f'Effect of {parameter_type} on {result_type.capitalize()}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_rho_effect_utility(self, rho_values):
        results_DK = []
        results_CN = []

        for rho in rho_values:
            self.rho = rho
            _, U_DK, U_CN = self.before_trade()
            results_DK.append(U_DK)
            results_CN.append(U_CN)

        plt.plot(rho_values, results_DK, label='Denmark')
        plt.plot(rho_values, results_CN, label='China')
        plt.xlabel('rho')
        plt.ylabel('Utility')
        plt.title('Utility as a Result of Change in rho')
        plt.legend()
        plt.show()

    # Generating PPF's for the two countries. 
    def generate_ppf_data(self, K_total, L_total, alpha_W, beta_W, alpha_T, beta_T):
        windmill_production = []
        textile_production = []

        # Varying capital allocation to windmills from 0 to K_total
        for K_windmills in np.linspace(0, K_total, 100):
            K_textiles = K_total - K_windmills
            L_windmills = L_total * (K_windmills / K_total)  # Assuming proportional allocation of labor
            L_textiles = L_total - L_windmills

            # Calculate production of windmills and textiles
            prod_windmills = self.production(K_windmills, L_windmills, alpha_W, beta_W)
            prod_textiles = self.production(K_textiles, L_textiles, alpha_T, beta_T)

            windmill_production.append(prod_windmills)
            textile_production.append(prod_textiles)

        return windmill_production, textile_production

    def plot_ppf(self):
        # Generate PPF data for Denmark
        windmill_prod_DK, textile_prod_DK = self.generate_ppf_data(
            self.K_DK, self.L_DK, self.alpha_DK_W, self.beta_DK_W, self.alpha_CN_T, self.beta_CN_T)
        
        # Generate PPF data for China
        windmill_prod_CN, textile_prod_CN = self.generate_ppf_data(
            self.K_CN, self.L_CN, self.alpha_DK_W, self.beta_DK_W, self.alpha_CN_T, self.beta_CN_T)

        # Plotting the PPFs on the same graph
        plt.figure(figsize=(8, 6))
        plt.plot(windmill_prod_DK, textile_prod_DK, label='PPF of Denmark')
        plt.plot(windmill_prod_CN, textile_prod_CN, label='PPF of China', linestyle='--')
        plt.xlabel('Production of Windmills')
        plt.ylabel('Production of Textiles')
        plt.title('Production Possibility Frontier (PPF) for Denmark and China')
        plt.legend()
        plt.grid(True)
        plt.show()


    # ############################################################################################################### #
    # ##################################### EXTENSION: CES PRODUCTION FUNCTION ###################################### #
    # ############################################################################################################### #
    # From here we do the extension by using another production function: A Constant Elasticity of Substitution (CES) production function.
    def production_ces(self, K, L, alpha, rho):
        # This assumes alpha is the capital share and (1-alpha) is the labor share
        return (alpha * K**rho + (1 - alpha) * L**rho)**(1/rho)
    
    def utility_ces(self, C1, C2, rho): 
        return (C1**rho + C2**rho)**(1/rho) # Utility function for a CES

    def constraints_ces(self): # Constraints for optimization
        return [
            {'type': 'ineq', 'fun': lambda x: self.K_DK - x[0] - x[2]}, # Capital constraint for Denmark
            {'type': 'ineq', 'fun': lambda x: self.L_DK - x[1] - x[3]}, # Labor constraint for Denmark
            {'type': 'ineq', 'fun': lambda x: self.K_CN - x[4] - x[6]}, # Capital constraint for China
            {'type': 'ineq', 'fun': lambda x: self.L_CN - x[5] - x[7]}, # Labor constraint for China
            {'type': 'ineq', 'fun': lambda x: x}  # Non-negativity
        ]

    def objective_ces(self, x):
        # Update the objective to account for production of both goods in both countries
        prod_DK_W = self.production_ces(x[0], x[1], self.alpha_DK_W, self.beta_DK_W) # Production of Windmills in Denmark
        prod_DK_T = self.production_ces(x[2], x[3], self.alpha_CN_T, self.beta_CN_T) # Production of Textiles in Denmark
        prod_CN_W = self.production_ces(x[4], x[5], self.alpha_DK_W, self.beta_DK_W) # Production of Windmills in China
        prod_CN_T = self.production_ces(x[6], x[7], self.alpha_CN_T, self.beta_CN_T) # Production of Textiles in China
        U_DK = self.utility_ces(prod_DK_W + prod_CN_W, prod_DK_T + prod_CN_T, self.rho) # Utility for Denmark
        U_CN = self.utility_ces(prod_CN_T + prod_CN_W, prod_DK_T + prod_DK_W, self.rho) # Utility for China
        return -(U_DK + U_CN) # Objective is to maximize total utility

    def run_optimization_ces(self): # Run optimization
        initial_guess = [self.K_DK/2, self.L_DK/2, self.K_DK/2, self.L_DK/2, self.K_CN/2, self.L_CN/2, self.K_CN/2, self.L_CN/2] # Initial guess
        result = minimize(self.objective_ces, initial_guess, constraints=self.constraints_ces(), method='SLSQP') # Optimization
        if result.success: # If optimization is successful
            x = result.x
            # Calculate production based on optimized resource allocations
            prod_DK_W = self.production_ces(x[0], x[1], self.alpha_DK_W, self.beta_DK_W)
            prod_DK_T = self.production_ces(x[2], x[3], self.alpha_CN_T, self.beta_CN_T)
            prod_CN_W = self.production_ces(x[4], x[5], self.alpha_DK_W, self.beta_DK_W)
            prod_CN_T = self.production_ces(x[6], x[7], self.alpha_CN_T, self.beta_CN_T)
            
            allocations = { # Resource allocations and production in each country
                'DK': {'K_Windmills': x[0], 'L_Windmills': x[1], 'K_Textiles': x[2], 'L_Textiles': x[3],
                    'Production_Windmills': prod_DK_W, 'Production_Textiles': prod_DK_T},
                'CN': {'K_Windmills': x[4], 'L_Windmills': x[5], 'K_Textiles': x[6], 'L_Textiles': x[7],
                    'Production_Windmills': prod_CN_W, 'Production_Textiles': prod_CN_T}
            }
            U_DK, U_CN = self.separate_utilities_ces(x)
            return allocations, U_DK, U_CN
        else:
            raise Exception("Optimization failed: " + result.message)

    def before_trade_ces(self):
        # Calculate production quantities
        prod_DK_W = self.production_ces(self.K_DK, self.L_DK, self.alpha_DK_W, self.beta_DK_W)
        prod_CN_T = self.production_ces(self.K_CN, self.L_CN, self.alpha_CN_T, self.beta_CN_T)
        
        # Calculate utility for Denmark
        U_DK = self.utility(prod_DK_W, 0, self.rho)
        
        # Calculate utility for China
        U_CN = self.utility(0, prod_CN_T, self.rho)

        return {
            'DK': {'K_Windmills': self.K_DK, 'L_Windmills': self.L_DK, 'K_Textiles': 0, 'L_Textiles': 0,
                'Production_Windmills': prod_DK_W, 'Production_Textiles': 0},
            'CN': {'K_Windmills': 0, 'L_Windmills': 0, 'K_Textiles': self.K_CN, 'L_Textiles': self.L_CN,
                'Production_Windmills': 0, 'Production_Textiles': prod_CN_T}
        }, U_DK, U_CN

    def separate_utilities_ces(self, x):
        # Re-calculate production based on optimized values
        prod_DK_W = self.production_ces(x[0], x[1], self.alpha_DK_W, self.beta_DK_W)
        prod_DK_T = self.production_ces(x[2], x[3], self.alpha_CN_T, self.beta_CN_T)
        prod_CN_W = self.production_ces(x[4], x[5], self.alpha_DK_W, self.beta_DK_W)
        prod_CN_T = self.production_ces(x[6], x[7], self.alpha_CN_T, self.beta_CN_T)
        # Calculate utility for each country
        U_DK = self.utility_ces(prod_DK_W + prod_CN_W, prod_DK_T + prod_CN_T, self.rho) # Utility for Denmark
        U_CN = self.utility_ces(prod_CN_T + prod_CN_W, prod_DK_T + prod_DK_W, self.rho) # Utility for China
        return U_DK, U_CN
    
    def plot_trade_results_ces(self):
        # Get the results before trade
        before_trade_results, U_DK_before, U_CN_before = self.before_trade_ces()

        # Get the results after trade
        try:
            allocations, U_DK, U_CN = self.run_optimization_ces()
        except Exception as e:
            print(e)
            allocations = None

        if allocations:
            # Data Preparation
            labels = ['Denmark', 'China']
            utility = [U_DK_before, U_DK, U_CN_before, U_CN]
            
            x = np.arange(len(labels))  # Label locations
            width = 0.35  # Width of the bars
            offset = width / 2  # Offset for side-by-side bar placement

            fig, ax = plt.subplots(figsize=(10, 5))

            # Utility Plot
            ax.bar(x - offset, utility[::2], width, label='Utility Before Trade')
            ax.bar(x + offset, utility[1::2], width, label='Utility After Trade', alpha=0.5)

            ax.set_ylabel('Utility')
            ax.set_title('Utility by Country Before and After Trade')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()

            plt.tight_layout()
            plt.show()
    
    def plot_parameter_effect_combined_ces(self, parameter_values, parameter_type, result_type='utility'):
        results_DK = []
        results_CN = []
        for value in parameter_values:
            if parameter_type == 'alpha':
                self.alpha_DK_W = value
                self.alpha_CN_T = value
            elif parameter_type == 'beta':
                self.beta_DK_W = value
                self.beta_CN_T = value
            
            _, U_DK, U_CN = self.before_trade_ces()
            results_DK.append(U_DK)
            results_CN.append(U_CN)
        
        plt.figure(figsize=(8, 6))
        plt.plot(parameter_values, results_DK, label=f'Denmark {parameter_type}')
        plt.plot(parameter_values, results_CN, label=f'China {parameter_type}')
        plt.xlabel(f'{parameter_type} values')
        plt.ylabel('Utility' if result_type == 'utility' else 'Production')
        plt.title(f'Effect of {parameter_type} on {result_type.capitalize()}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_parameter_effect_ces(self, parameter_values, parameter_name, result_type='utility'):
        results = []
        for value in parameter_values:
            if parameter_name == 'alpha_DK_W':
                self.alpha_DK_W = value
            elif parameter_name == 'beta_DK_W':
                self.beta_DK_W = value
            elif parameter_name == 'alpha_CN_T':
                self.alpha_CN_T = value
            elif parameter_name == 'beta_CN_T':
                self.beta_CN_T = value
            
            if result_type == 'utility':
                _, U_DK, U_CN = self.before_trade_ces()
                results.append((U_DK, U_CN))
            elif result_type == 'production':
                allocations, _, _ = self.run_optimization_ces()
                prod_DK_W = allocations['DK']['Production_Windmills']
                prod_DK_T = allocations['DK']['Production_Textiles']
                prod_CN_W = allocations['CN']['Production_Windmills']
                prod_CN_T = allocations['CN']['Production_Textiles']
                results.append((prod_DK_W + prod_CN_W, prod_DK_T + prod_CN_T))
        
        results = np.array(results)
        plt.plot(parameter_values, results[:, 0], label='Denmark')
        plt.plot(parameter_values, results[:, 1], label='China')
        plt.xlabel(parameter_name)
        if result_type == 'utility':
            plt.ylabel('Utility')
        elif result_type == 'production':
            plt.ylabel('Production')
        plt.title(f'{result_type.capitalize()} as a result of change in {parameter_name}')
        plt.legend()
        plt.show()

    def plot_rho_effect_utility_ces(self, rho_values):
        results_DK = []
        results_CN = []

        for rho in rho_values:
            self.rho = rho
            _, U_DK, U_CN = self.before_trade_ces()
            results_DK.append(U_DK)
            results_CN.append(U_CN)

        plt.plot(rho_values, results_DK, label='Denmark')
        plt.plot(rho_values, results_CN, label='China')
        plt.xlabel('rho')
        plt.ylabel('Utility')
        plt.title('Utility as a Result of Change in rho')
        plt.legend()
        plt.show()

    def generate_ppf_data_ces(self, K_total, L_total, alpha_W, alpha_T, rho):
        windmill_production = []
        textile_production = []

        # Varying capital allocation to windmills from 0 to K_total
        for K_windmills in np.linspace(0, K_total, 100):
            K_textiles = K_total - K_windmills
            L_windmills = L_total * (K_windmills / K_total)  # Assuming proportional allocation of labor
            L_textiles = L_total - L_windmills

            # Calculate production of windmills and textiles
            prod_windmills = self.production_ces(K_windmills, L_windmills, alpha_W, rho)
            prod_textiles = self.production_ces(K_textiles, L_textiles, alpha_T, rho)

            windmill_production.append(prod_windmills)
            textile_production.append(prod_textiles)

        return windmill_production, textile_production

    def plot_ppf_ces(self):
        # Generate PPF data for Denmark
        windmill_prod_DK, textile_prod_DK = self.generate_ppf_data_ces(
            self.K_DK, self.L_DK, self.alpha_DK_W, self.alpha_CN_T, self.rho)
        
        # Generate PPF data for China
        windmill_prod_CN, textile_prod_CN = self.generate_ppf_data_ces(
            self.K_CN, self.L_CN, self.alpha_DK_W, self.alpha_CN_T, self.rho)

        # Plotting the PPFs on the same graph
        plt.figure(figsize=(8, 6))
        plt.plot(windmill_prod_DK, textile_prod_DK, label='PPF of Denmark')
        plt.plot(windmill_prod_CN, textile_prod_CN, label='PPF of China', linestyle='--')
        plt.xlabel('Production of Windmills')
        plt.ylabel('Production of Textiles')
        plt.title('Production Possibility Frontier (PPF) for Denmark and China')
        plt.legend()
        plt.grid(True)
        plt.show()