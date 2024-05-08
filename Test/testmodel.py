from scipy import optimize
import numpy as np
from types import SimpleNamespace

class HOmodelClass:
    def __init__(self):
        par = self.par = SimpleNamespace()

        # Parameter initialization
        par.alpha = 0.6  # capital productivity in wind power production (capital-intensive)
        par.beta = 0.3   # capital productivity in textile production (labor-intensive)
        par.aw = 1.0    # productivity for wind power 
        par.ax = 1.0    # productivity for textile
        par.ww = 1.0    # wage rate in wind power sector
        par.wx = 1.0    # wage rate in textile sector
        par.rw = 1.0    # rental rate in wind power sector
        par.rx = 1.0    # rental rate in textile sector
        par.phi = 0.5   # weight on wind power in the utility function
        par.psi = 0.5   # weight on textile in the utility function
        par.pw = 2.0    # price of wind power
        par.px = 1.0    # price of textile


        # Total resources for each country
        self.K_total_DK = 800  # total capital in Denmark (abundant in capital)
        self.L_total_DK = 600  # total labor in Denmark
        self.K_total_CN = 400  # total capital in China
        self.L_total_CN = 700  # total labor in China (abundant in labor)

        # Initial values for capital and labor in the two sectors in the two countries
        self.Kw_DK, self.Kx_DK = 500, 300  
        self.Lw_DK, self.Lx_DK = 200, 400  

        self.Kw_CN, self.Kx_CN = 300, 100  
        self.Lw_CN, self.Lx_CN = 250, 450 

        # Production functions for Denmark
        self.Yw_DK = lambda Kw_DK, Lw_DK: self.par.aw * (Kw_DK ** self.par.alpha) * (Lw_DK ** (1 - self.par.alpha))
        self.Yx_DK = lambda Kx_DK, Lx_DK: self.par.ax * (Kx_DK ** self.par.beta) * (Lx_DK ** (1 - self.par.beta))

        # Production functions for China
        self.Yw_CN = lambda Kw_CN, Lw_CN: self.par.aw * (Kw_CN ** self.par.alpha) * (Lw_CN ** (1 - self.par.alpha))
        self.Yx_CN = lambda Kx_CN, Lx_CN: self.par.ax * (Kx_CN ** self.par.beta) * (Lx_CN ** (1 - self.par.beta))

        # Utility functions for the two countries
        self.U_DK = lambda Yw_DK, Yx_DK: Yw_DK**par.phi * Yx_DK**(1-par.phi)
        self.U_CN = lambda Yw_CN, Yx_CN: Yw_CN**par.psi * Yx_CN**(1-par.psi)

    def capital_labor_ratios(self):
        # Define the capital-labor ratios in the two countries
        self.wratio_DK = self.Kw_DK / self.Lw_DK # capital-labor ratio in wind power sector in Denmark
        self.xratio_DK = self.Kx_DK / self.Lx_DK # capital-labor ratio in textile sector in Denmark
        self.wratio_CN = self.Kw_CN / self.Lw_CN # capital-labor ratio in wind power sector in China
        self.xratio_CN = self.Kx_CN / self.Lx_CN # capital-labor ratio in textile sector in China

        print(f"Total capital-labor ratio in Denmark: {self.K_total_DK / self.L_total_DK}")
        print(f"Total capital-labor ratio in China: {self.K_total_CN / self.L_total_CN}")
    
    def market_clearing(self):
        par = self.par = SimpleNamespace()
        #Define optimal use of capital and labor in the two sectors 
        self.MPKw = self.par.alpha*self.par.aw*(self.Lw_DK/self.Kw_DK)**(1-self.par.alpha)
        self.MPLw = (1-self.par.alpha)*self.par.aw*(self.Kw_DK/self.Lw_DK)**self.par.alpha

        self.MPKx = self.par.beta*self.par.ax*(self.Lx_DK/self.Kx_DK)**(1-self.par.beta) 
        self.MPLx = (1-self.par.beta)*self.par.ax*(self.Kx_DK/self.Lx_DK)**self.par.beta

        # Define the autarchy equilibrium
        # The labor market clearing conditions
        self.ww = self.par.pw*self.MPLw
        self.wx = self.par.px*self.MPLx
        # The capital market clearing conditions
        self.rw = self.par.pw*self.MPKw
        self.rx = self.par.px*self.MPKx


