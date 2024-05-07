from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import numpy as np
from scipy import optimize
import itertools
from scipy.optimize import minimize
from types import SimpleNamespace
from scipy.optimize import fsolve

class HOmodelClass():

    def __init__(self):

        par = self.par = SimpleNamespace()

        #Baseline parameters are defined
        #Defining parametervalues the economy and initial values
        par.alpha = 1/3    # weight on capital for Denmarks (DK) production
        par.beta = 2/3     # weight on capital for Chinas (CN) production
        par.phi = 0.6       # Substitutionelasticity between the two goods for DK(CES-function)
        par.psi = 1       # Substitutionelasticity between the two goods for CN (CES-function)
        par.w = 1           # Wage
        par.r = 0.05        # Interest rate
        par.Aw = 1          # Productivity for DK
        par.Ax = 1          # Productivity for CN

        #Defining the initial values
        self.Lw = 300           # Labor for DK
        self.Lx = 1000           # Labor for CN
        self.Kw = 1000           # Capital for DK
        self.Kx = 300           # Capital for CN

        #Defining the production functions
        self.Yw = lambda Lw, Kw, : par.Aw *(Lw**(1-par.alpha))*(Kw**par.alpha)
        self.Yx = lambda Lx, Kx, : par.Ax *(Lx**(1-par.beta))*(Kx**par.beta)

        #Defining the resource constraints #MANGLER 0'er ???
        self.Yww_max = self.Yw(self.Lw, self.Kw) #Maximum production for DK of wind (Yww = Production of w from w producer)
        self.Yxw_max = self.Yx(self.Lw, self.Kw) #Maximum production for DK of textile (Yxw = Production of x from w producer)
        self.Yxx_max = self.Yx(self.Lx, self.Kx)  #Maximum production for CN of textile (Yxx = Production of x from x producer)
        self.Ywx_max = self.Yw(self.Lx, self.Kx) #Maximum production for CN of wind (Ywx = Production of w from x producer)

        #Defining the utility functions
        self.Uw = lambda Yw, Yx : Yx**(par.phi)*Yw**(1-par.phi) 
        self.Ux = lambda Yw, Yx : Yx**(par.psi)*Yw**(1-par.psi)

        print(f"alpha: {par.alpha:.4f}, beta: {par.beta:.4f}, phi: {par.phi:.4f}, psi: {par.psi:.4f}, w: {par.w:.4f}, r: {par.r:.4f}, Aw: {par.Aw:.4f}, Ax: {par.Ax:.4f}")

        return 
     
    def capital_labor_ratio(self):
        par = self.par

        #Firstly we define the capital-labor ratio for each country
        KLr_w= self.Kw/self.Lw
        KLr_x= self.Kx/self.Lx
        
        #We define wether the countries are capital abundant or labor abundant
        if KLr_w > KLr_x:
            print("Denmark is capital abundant")
            print("China is labor abundant")
        else:
            print("Denmark is labor abundant")
            print("China is capital abundant")
        
        return
    

    def optimize_production_without_trade(self):       
        def production_constraint(x):
            return self.Lw - x[0] - x[1], self.Kw - x[0] - x[1], self.Lx - x[2] - x[3], self.Kx - x[2] - x[3]

        x0 = [0.25 * self.Lw, 0.25 * self.Kw, 0.25 * self.Lx, 0.25 * self.Kx]  # Initial guess for labor and capital allocation

        # Constraints
        constraints = ({'type': 'eq', 'fun': production_constraint})


        result = minimize(lambda x: -(self.Uw(self.Yw(x[0], x[1]), self.Yx(x[2], x[3])) + self.Ux(self.Yw(self.Lx - x[2], self.Kx - x[3]), self.Yx(x[2], x[3]))), x0, constraints=constraints)
        # Optimization
        #result = minimize(lambda x: -(self.Yw(x[0], x[1]) + self.Yx(x[2], x[3])), x0, constraints=constraints)
        #result = -minimize(self.Uw, x0, constraints=constraints)

        Yw_DK = self.Yw(result.x[0], result.x[1])
        Yx_DK = self.Yx(result.x[2], result.x[3])
        utility_DK = self.Uw(Yw_DK, Yx_DK)
        
        Yw_CN = self.Yw(self.Lx - result.x[2], self.Kx - result.x[3])
        Yx_CN = self.Yx(result.x[2], result.x[3])
        utility_CN = self.Ux(Yw_CN, Yx_CN)

    # Print the additional information
        print()  # Add an empty print statement for indentation
        print("Production without trade:")
        print("Labor DK:   ", round(result.x[0], 2))
        print("Capital DK: ", round(result.x[1], 2))
        print("Labor CN:   ", round(result.x[2], 2))
        print("Capital CN: ", round(result.x[3], 2))
        print("Yw DK:      ", round(Yw_DK, 2))
        print("Yx DK:      ", round(Yx_DK, 2))
        print("Yw CN:      ", round(Yw_CN, 2))
        print("Yx CN:      ", round(Yx_CN, 2))
        print("Utility DK: ", round(utility_DK, 2))
        print("Utility CN: ", round(utility_CN, 2))
        return

    def optimize_production_with_trade(self):
        def objective_function(x):
            Yw_DK = x[0]
            Yx_DK = x[1]
            Yw_CN = x[2]
            Yx_CN = x[3]
            return -(self.Uw(Yw_DK, Yx_DK) + self.Ux(Yw_CN, Yx_CN))

        x0 = [0.25 * self.Yww_max, 0.25 * self.Yxw_max, 0.25 * self.Ywx_max, 0.25 * self.Yxx_max]  # Initial guess for consumption

        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: self.Yw(self.Lw, self.Kw) - x[0] - x[2]},  # Wind production constraint
                       {'type': 'eq', 'fun': lambda x: self.Yx(self.Lw, self.Kw) - x[1] - x[3]},  # Textile production constraint
                       {'type': 'eq', 'fun': lambda x: self.Yw(self.Lx, self.Kx) - x[2] - x[0]},  # Wind production constraint
                       {'type': 'eq', 'fun': lambda x: self.Yx(self.Lx, self.Kx) - x[3] - x[1]})  # Textile production constraint

        # Optimization
#        result = minimize(objective_function, x0, constraints=constraints)
#        result = minimize(self.Uw, x0, constraints=constraints)
 
        result = minimize(lambda x: -(self.Uw(self.Yw(x[0], x[1]), self.Yx(x[2], x[3])) + self.Ux(self.Yw(self.Lx - x[2], self.Kx - x[3]), self.Yx(x[2], x[3]))), x0, constraints=constraints)


        Yw_DK = result.x[0]
        Yx_DK = result.x[1]
        Yw_CN = result.x[2]
        Yx_CN = result.x[3]
        utility_DK = self.Uw(Yw_DK, Yx_DK)
        utility_CN = self.Ux(Yw_CN, Yx_CN)
        
    # Print the additional information
        print()  # Add an empty print statement for indentation
        print("Production with trade:")
        print("Labor DK:   ", round(result.x[0], 2))
        print("Capital DK: ", round(result.x[1], 2))
        print("Labor CN:   ", round(result.x[2], 2))
        print("Capital CN: ", round(result.x[3], 2))
        print("Yw DK:      ", round(Yw_DK, 2))
        print("Yx DK:      ", round(Yx_DK, 2))
        print("Yw CN:      ", round(Yw_CN, 2))
        print("Yx CN:      ", round(Yx_CN, 2))
        print("Utility DK: ", round(utility_DK, 2))
        print("Utility CN: ", round(utility_CN, 2))
        # Return the results

        return
  
    def optimize_production_without_trade_analytical(self):
        def production_constraint(x):
            return self.Lw - x[0] - x[1], self.Kw - x[0] - x[1], self.Lx - x[2] - x[3], self.Kx - x[2] - x[3]

        x0 = [0.25 * self.Lw, 0.25 * self.Kw, 0.25 * self.Lx, 0.25 * self.Kx]  # Initial guess for labor and capital allocation

        print("Initial guess (x0):", x0)
        # Constraints
        constraints = ({'type': 'eq', 'fun': production_constraint})

        # Optimization using fsolve (analytical solution)
        result = fsolve(lambda x: -np.array([self.Uw(x[0], x[1]) + self.Ux(x[2], x[3]), 0, 0, 0]), x0)

        Yw_DK = self.Yw(result[0], result[1])
        Yx_DK = self.Yx(result[2], result[3])
        utility_DK = self.Uw(Yw_DK, Yx_DK)

        Yw_CN = self.Yw(self.Lx - result[2], self.Kx - result[3])
        Yx_CN = self.Yx(result[2], result[3])
        utility_CN = self.Ux(Yw_CN, Yx_CN)

        # Print the additional information
        print()  # Add an empty print statement for indentation
        print("Production without trade (analytical):")
        print("Labor DK:   ", round(result[0], 2))
        print("Capital DK: ", round(result[1], 2))
        print("Labor CN:   ", round(result[2], 2))
        print("Capital CN: ", round(result[3], 2))
        print("Yw DK:      ", round(Yw_DK, 2))
        print("Yx DK:      ", round(Yx_DK, 2))
        print("Utility DK: ", round(utility_DK, 2))
        print("Utility CN: ", round(utility_CN, 2))

        return

    def optimize_production_with_trade_analytical(self):
        def objective_function(x):
            Yw_DK = x[0]
            Yx_DK = x[1]
            Yw_CN = x[2]
            Yx_CN = x[3]
            return -(self.Uw(Yw_DK, Yx_DK) + self.Ux(Yw_CN, Yx_CN))

        x0 = [0.25 * self.Yww_max, 0.25 * self.Yxw_max, 0.25 * self.Ywx_max, 0.25 * self.Yxx_max]  # Initial guess for consumption
        

        constraints = ({'type': 'eq', 'fun': lambda x: self.Yw(self.Lw, self.Kw) - x[0] - x[2]},  # Wind production constraint
                    {'type': 'eq', 'fun': lambda x: self.Yx(self.Lw, self.Kw) - x[1] - x[3]},  # Textile production constraint
                    {'type': 'eq', 'fun': lambda x: self.Yw(self.Lx, self.Kx) - x[2] - x[0]},  # Wind production constraint
                    {'type': 'eq', 'fun': lambda x: self.Yx(self.Lx, self.Kx) - x[3] - x[1]})  # Textile production constraint

        # Optimization using fsolve (analytical solution)
        result = fsolve(lambda x: -np.array([self.Uw(x[0], x[1]) + self.Ux(x[2], x[3]), 0, 0, 0]), x0)

        Yw_DK = result[0]
        Yx_DK = result[1]
        Yw_CN = result[2]
        Yx_CN = result[3]
        utility_DK = self.Uw(Yw_DK, Yx_DK)
        utility_CN = self.Ux(Yw_CN, Yx_CN)

        # Print the additional information
        print()  # Add an empty print statement for indentation
        print("Production with trade (analytical):")
        print("Labor DK:   ", round(result[0], 2))
        print("Capital DK: ", round(result[1], 2))
        print("Labor CN:   ", round(result[2], 2))
        print("Capital CN: ", round(result[3], 2))
        print("Yw DK:      ", round(Yw_DK, 2))
        print("Yx DK:      ", round(Yx_DK, 2))
        print("Utility DK: ", round(utility_DK, 2))
        print("Utility CN: ", round(utility_CN, 2))

        return
    
    def plot_utility_surface(self, country='DK', trade=True):
        # Define ranges for capital and labor inputs based on the selected country
        if country == 'DK':
            L_range = np.linspace(0, self.Lw, 50)
            K_range = np.linspace(0, self.Kw, 50)
            max_utility_function = self.Uw
            title = "Utility Surface for Denmark"
        else:
            L_range = np.linspace(0, self.Lx, 50)
            K_range = np.linspace(0, self.Kx, 50)
            max_utility_function = self.Ux
            title = "Utility Surface for China"
        
        # Create meshgrid
        L_mesh, K_mesh = np.meshgrid(L_range, K_range)

        Yw_country = lambda L, K: self.Yw(L, K)
        Yx_country = lambda L, K: self.Yx(np.maximum(0, self.Lw - L), np.maximum(0, self.Kw - K)) if country == 'DK' else lambda L, K: self.Yx(np.maximum(0, L - self.Lx), np.maximum(0, K - self.Kx))

        utility = np.zeros_like(L_mesh, dtype=float)
        for i in range(len(L_range)):
            for j in range(len(K_range)):
                Yw = Yw_country(L_mesh[i, j], K_mesh[i, j])
                Yx = Yx_country(L_mesh[i, j], K_mesh[i, j])
                utility[i, j] = max_utility_function(Yw, Yx)

        # Find the indices of the maximum utility
        max_utility_index = np.unravel_index(np.nanargmax(utility), utility.shape)
        max_utility_point = (L_mesh[max_utility_index], K_mesh[max_utility_index])

        # Plot the surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(L_mesh, K_mesh, utility, cmap='viridis')
        ax.scatter(max_utility_point[0], max_utility_point[1], max_utility_function(max_utility_point[0], max_utility_point[1]), color='red', label='Max Utility')
        ax.set_xlabel('Labor')
        ax.set_ylabel('Capital')
        ax.set_zlabel('Utility')
        ax.set_title(title)
        ax.legend()
        plt.show()

        print(f"Max Utility Point for {country}: Labor={max_utility_point[0]}, Capital={max_utility_point[1]}, Utility={utility[max_utility_index]}")