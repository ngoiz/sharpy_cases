#Transmit gust across chord

import numpy as np
import sys
import scipy as sc

sys.path.append('~/code/sharpy/')

M = 5
dt = 0.1

A_gust = np.zeros((M, M))
A_gust[1:, :-1] = np.eye(M-1)
B_gust = np.zeros((M, 1))
B_gust[0] = 1
C_gust = np.eye(M)
D_gust = None
# ss_gust = libss.ss(A_gust, B_gust, C_gust, D_gust, dt=dt)
ss_gust = sc.signal.dlti(A_gust, B_gust, C_gust, D_gust, dt=dt)

# Input
Nsteps = 100
time_dom = np.linspace(0, dt*Nsteps, Nsteps+1)
uz = np.ones_like(time_dom)

out = sc.signal.dlsim(ss_gust, uz, time_dom)

print('End')
