from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import numpy as np
from scipy import optimize
import itertools
from scipy.optimize import minimize

class HOmodelClass():

    def __init__(self):

        par = self.par = SimpleNamespace()

        #Baseline parameters are defined
        #Defining parametervalues the economy and initial values
        par.alpha = 1/3    # weight on capital for Denmarks (DK) production
        par.beta = 2/3     # weight on capital for Chinas (CN) production
        par.phi = 0.5       # Substitutionelasticity between the two goods for DK(CES-function)
        par.psi = 0.5       # Substitutionelasticity between the two goods for CN (CES-function)
        par.w = 1           # Wage
        par.r = 0.05        # Interest rate
        par.Aw = 1          # Productivity for DK
        par.Ax = 1          # Productivity for CN

        #Defining the initial values
        self.Lw = 800           # Labor for DK
        self.Lx = 1000           # Labor for CN
        self.Kw = 1000           # Capital for DK
        self.Kx = 800           # Capital for CN

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

        print(f"alpha: {par.alpha:.4f}, beta: {par.beta:.4f}, phi: {par.phi:.4f}, psi: {par.psi:.4f}, w: {par.w:.4f}, r: {par.r:.4f}, Aw: {par.Aw:.4f}, Ax: {par.Ax:.4f}, Lw: {self.Lw:.4f}, Lx: {self.Lx:.4f}, Kw: {self.Kw:.4f}, Kx: {self.Kx:.4f}, Uw: {self.Uw(self.Yww_max, self.Yxx_max):.4f}, Ux: {self.Ux(self.Yww_max, self.Yxx_max):.4f}, Yw: {self.Yw(self.Lw,self.Kw):.4f}, Yx: {self.Yx(self.Lx, self.Kx):.4f}")

        return 
    
    def max_utility_without_trade(self):
        par = self.par 
        Uw_max = -np.inf
        Ux_max = -np.inf
        Lw_opt = 0
        Kw_opt = 0
        Lx_opt = 0
        Kx_opt = 0

        for l in range (self.Lw):
            for k in range (self.Kw):
                utility_w = self.Uw(l, k)
                if utility_w > Uw_max:
                    Uw_max = utility_w
                    Lw_opt = l
                    Kw_opt = k
    
        for l in range (self.Lx):
            for k in range (self.Kx):
                utility_x = self.Uw(l, k)
                if utility_x > Ux_max:
                    Ux_max = utility_x
                    Lx_opt = l
                    Kx_opt = k

#Calculation the production based on the optmum values for Lw, Kw, Lx, Kx
        self.Yw_opt = self.Yw(Lw_opt, Kw_opt)
        self.Yx_opt = self.Yx(Lx_opt, Kx_opt)

        print(f"Uw_max: {Uw_max:.4f}, Ux_max: {Ux_max:.4f}, Lw_opt: {Lw_opt:.4f}, Lx_opt: {Lx_opt:.4f}, Kw_opt: {Kw_opt:.4f}, Kx_opt: {Kx_opt:.4f}, Yw_opt: {self.Yw_opt:.4f}, Yx_opt: {self.Yx_opt:.4f}")
        
        return #round(Lw_opt, 2), round(Kw_opt, 2), round(Lx_opt, 2), round(Kx_opt, 2), round(self.Yw_opt, 2), round(self.Yx_opt, 2)
    
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
    
    def max_utility_with_trade(self):
        par = self.par
        self.MPLx = (1-par.alpha)*par.Ax*(self.Kx/self.Lx)**par.alpha
        self.MPKx = par.alpha*par.Ax*(self.Lx/self.Kx)**(1-par.alpha)

        self.MPLw = (1-par.beta)*par.Aw*(self.Kw/self.Lw)**par.beta
        self.MPKw = par.beta*par.Aw*(self.Lw/self.Kw)**(1-par.beta)
        
        self.Pw = par.w/self.MPLw
        self.Px = par.w/self.MPLx

        self.Px = par.r/self.MPKx
        self.Pw = par.r/self.MPKw

        #Relative prices
        self.Pr =  self.MPLx/self.MPLw
        self.Pr =  self.MPKx/self.MPKw
        self.Pr = self.Pw/self.Px

        #solving
        par.r = self.MPKw/self.MPLw*par.w

        #Defining the utility functions with prices
        self.Uw = lambda Yw, Yx : self.Px*Yx**(par.phi)*Yw**(1-par.phi)
        self.Ux = lambda Yw, Yx : self.Pw*Yx**(par.psi)*Yw**(1-par.psi)

        #Finding the optimal values for Lw, Kw, Lx, Kx with trade
        Uw_max = -np.inf
        Ux_max = -np.inf
        Lw_opt = 0
        Kw_opt = 0
        Lx_opt = 0
        Kx_opt = 0

        combinations = itertools.product(range(self.Lw + 1), range(self.Kw + 1), range(self.Lx + 1), range(self.Kx + 1))

        for Lw, Kw, Lx, Kx in combinations:
            # Check if the combination violates resource constraints
            if Lw + Lx <= self.Lw and Kw + Kx <= self.Kw:
                # Calculate utility for country w
                utility_w = self.Uw(self.Yw(Lw, Kw), self.Yx(Lx, Kx))
                if utility_w > Uw_max:
                    Uw_max = utility_w
                    Lw_opt = Lw
                    Kw_opt = Kw
                    Lx_opt = Lx
                    Kx_opt = Kx

        #Calculation the production based on the optmum values for Lw, Kw, Lx, Kx
        self.Yw_opt = self.Yw(Lw_opt, Kw_opt)
        self.Yx_opt = self.Yx(Lx_opt, Kx_opt)

        print(f"Uw_max: {Uw_max:.4f}, Ux_max: {Ux_max:.4f}, Lw_opt: {Lw_opt:.4f}, Lx_opt: {Lx_opt:.4f}, Kw_opt: {Kw_opt:.4f}, Kx_opt: {Kx_opt:.4f}, Yw_opt: {self.Yw_opt:.4f}, Yx_opt: {self.Yx_opt:.4f}")

        return
    
    def optimize_production_without_trade(self):
        """
        Optimize production without trade
        
        Returns:
        results (dict): Dictionary containing results
        """
        def production_constraint(x):
            return self.Lw - x[0] - x[1], self.Kw - x[0] - x[1], self.Lx - x[2] - x[3], self.Kx - x[2] - x[3]

        x0 = [0.25 * self.Lw, 0.25 * self.Kw, 0.25 * self.Lx, 0.25 * self.Kx]  # Initial guess for labor and capital allocation

        # Constraints
        constraints = ({'type': 'eq', 'fun': production_constraint})

        # Optimization
        result = minimize(lambda x: -(self.Yw(x[0], x[1]) + self.Yx(x[2], x[3])), x0, constraints=constraints)
        
        Yw_DK = self.Yw(result.x[0], result.x[1])
        Yx_DK = self.Yx(result.x[2], result.x[3])
        utility_DK = self.Uw(Yw_DK, Yx_DK)
        
        Yw_CN = self.Yw(self.Lx - result.x[2], self.Kx - result.x[3])
        Yx_CN = self.Yx(result.x[2], result.x[3])
        utility_CN = self.Ux(Yw_CN, Yx_CN)

    # Print the additional information
        print("Production with trade:")
        print("Labor_DK:", result.x[0])
        print("Capital_DK:", result.x[1])
        print("Labor_CN:", result.x[2])
        print("Capital_CN:", result.x[3])
        print("Yw_DK:", Yw_DK)
        print("Yx_DK:", Yx_DK)
        print("Utility_DK:", utility_DK)
        print("Utility_CN:", utility_CN)
                
        
        return# {'Labor_DK': result.x[0], 'Capital_DK': result.x[1], 
               # 'Labor_CN': result.x[2], 'Capital_CN': result.x[3],
               # 'Yw_DK': Yw_DK, 'Yx_DK': Yx_DK,
               # 'Utility_DK': utility_DK, 'Utility_CN': utility_CN}

    def optimize_production_with_trade(self):
        """
        Optimize production with trade
        
        Returns:
        results (dict): Dictionary containing results
        """
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
        result = minimize(objective_function, x0, constraints=constraints)

        Yw_DK = result.x[0]
        Yx_DK = result.x[1]
        Yw_CN = result.x[2]
        Yx_CN = result.x[3]
        utility_DK = self.Uw(Yw_DK, Yx_DK)
        utility_CN = self.Ux(Yw_CN, Yx_CN)
        
    # Print the additional information
        print("Production with trade:")
        print("Labor_DK:", result.x[0])
        print("Capital_DK:", result.x[1])
        print("Labor_CN:", result.x[2])
        print("Capital_CN:", result.x[3])
        print("Yw_DK:", Yw_DK)
        print("Yx_DK:", Yx_DK)
        print("Utility_DK:", utility_DK)
        print("Utility_CN:", utility_CN)
        
        # Return the results
        return# {'Yw_DK': Yw_DK, 'Yx_DK': Yx_DK, 
               # 'Yw_CN': Yw_CN, 'Yx_CN': Yx_CN,
                #'Utility_DK': utility_DK, 'Utility_CN': utility_CN}
