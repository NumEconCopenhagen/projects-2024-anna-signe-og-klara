import numpy as np
from scipy.optimize import minimize
import sympy as sp

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

    def before_trade(self):
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
    
    def analyze_equilibrium(self):
        # Define symbols
        K_DK, L_DK, K_CN, L_CN, p_W, p_T = sp.symbols('K_DK L_DK K_CN L_CN p_W p_T')
        K_DK, L_DK, K_CN, L_CN = sp.symbols('K_DK L_DK K_CN L_CN', positive=True)
        alpha_DK_W, beta_DK_W = self.alpha_DK_W, self.beta_DK_W
        alpha_CN_T, beta_CN_T = self.alpha_CN_T, self.beta_CN_T

        # Production functions
        prod_DK_W = K_DK**alpha_DK_W * L_DK**beta_DK_W
        prod_CN_T = K_CN**alpha_CN_T * L_CN**beta_CN_T

        # Assume total demand for Windmills and Textiles is some fixed amount
        demand_W, demand_T = sp.symbols('demand_W demand_T')

        # Market-clearing conditions
        market_clearing_W = sp.Eq(prod_DK_W, demand_W)
        market_clearing_T = sp.Eq(prod_CN_T, demand_T)

        # Solve for equilibrium conditions
        sol = sp.solve([K_DK + K_CN - self.K_DK, L_DK + L_CN - self.L_DK,
                        market_clearing_W, market_clearing_T], 
                        (K_DK, L_DK, K_CN, L_CN), dict=True)

        return sol