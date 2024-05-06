from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import numpy as np
from scipy import optimize
import itertools

class HOmodelClass():

    def __init__(self):

        par = self.par = SimpleNamespace()

        #Baseline parameters are defined
        #Defining parametervalues the economy and initial values
        par.alpha = 0.33    # weight on capital for Denmarks (DK) production
        par.beta = 0.66     # weight on capital for Chinas (CN) production
        par.phi = 0.5       # Substitutionelasticity between the two goods for DK(CES-function)
        par.psi = 0.5       # Substitutionelasticity between the two goods for CN (CES-function)
        par.w = 1           # Wage
        par.r = 0.05        # Interest rate
        par.Aw = 1          # Productivity for DK
        par.Ax = 1          # Productivity for CN

        #Defining the initial values
        self.Lw = 400           # Labor for DK
        self.Lx = 650           # Labor for CN
        self.Kw = 700           # Capital for DK
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
    
    

        


