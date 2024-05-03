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
        Lw = 400           # Labor for DK
        Lx = 650           # Labor for CN
        Kw = 700           # Capital for DK
        Kx = 300           # Capital for CN

    def equations(self, alpha, beta, phi, psi, w, r, Aw, Ax, Lw, Lx, Kw, Kx):
        #Defining the production functions
        Yw = lambda Lw, Kw : Aw *(Lw**(1-alpha))*(Kw**alpha)
        Yx = lambda Lx, Kx : Ax *(Lx**(1-beta))*(Kx**beta)

        #Defining the resource constraints
        Yww_max = Yw(Lw, Kw) #Maximum production for DK of wind (Yww = Production of w from w producer)
        Yxw_max = Yx(Lw,0, Kw,0) #Maximum production for DK of textile (Yxw = Production of x from w producer)
        Yxx_max = Yw(Lx, Kx)  #Maximum production for CN of textile (Yxx = Production of x from x producer)
        Ywx_max = Yx(Lx,0, Kx,0) #Maximum production for CN of wind (Ywx = Production of w from x producer)

        #Defining the utility functions
        Uw = lambda Yw, Yx : Yx**(phi)*Yw**(1-phi) 
        Ux = lambda Yw, Yx : Yx**(psi)*Yw**(1-psi)

        return Yww_max, Yxw_max, Yxx_max, Ywx_max

