""" Very Simple Aerodynamic Model Linearisation"""

import numpy as np
import sys
sys.path.append('/home/ng213/code/sharpy/')
import sharpy.linear.src.libss as libss

sys.path.append('/home/ng213/sharpy_cases/08_SimpleGlider')


class SimpleAero(object):

    def __init__(self, tsaero, tsstr):

        self.rho = tsaero.rho
        self.u_inf = 28
        self.S = 6.11
        self.b = 15
        self.c = self.S / self.b
        self.lt = 4.63
        self.lw = 0.30
        self.St = 1.14
        self.AR = self.b / self.c

        self.CLa = 5.55
        self.Clat = 4.3
        e = 0.8
        self.alpha0 = tsstr.euler[1]

        self.ss = None

    def assemble(self):

        qinf = 0.5 * self.rho * self.u_inf ** 2
        e = 0.8

        L0 = 318*9.81 * np.cos(2*np.pi / 180) # qinf * self.S * self.CLa * self.alpha0
        CL0 = self.CLa * self.alpha0
        D0 = qinf * self.S * (0.005 + CL0 ** 2 / e / np.pi / self.AR)

        D = np.zeros((3, 3))

        D[0, 0] = 2 * L0 / self.u_inf
        D[0, 1] = qinf * self.S / self.u_inf * self.CLa
        D[0, 2] = qinf * self.St * self.Clat * self.lt / self.u_inf

        D[1, 0] = 2 * D0 / self.u_inf
        D[1, 1] = qinf * self.S * 2 * self.S*CL0*self.CLa/np.pi/self.b**2/e/self.u_inf

        D[2, 1] = qinf * self.S * self.c * (self.lw/self.c*self.CLa - self.St/self.S*self.lt/self.c*self.Clat)/self.u_inf
        D[2, 2] = qinf * self.S * self.c * self.St * self.lt / self.S / self.c * self.Clat * self.lt / self.u_inf

        self.ss = libss.ss(np.zeros((3, 3)),
                           np.zeros((3, 3)),
                           np.zeros((3, 3)),
                           D)



