import numpy as np
import sys
sys.path.append('/home/ng213/code/sharpy/')
import matplotlib.pyplot as plt
import scipy as sc
import sharpy.linear.src.lingebm as lingebm

import sharpy.sharpy_main


solver_file_path = './cases/flexible_beam_static.solver.txt'

data = sharpy.sharpy_main.main(['', solver_file_path])

tsstruct0 = data.structure.timestep_info[-1]

# dt = 0.0001

beam = lingebm.FlexDynamic(tsstruct0, dt=None, wv=None, use_euler=True)

# Rigid body equation only
beam.U = np.zeros((69, 2))

# omega_y mode
beam.U[-5, 0] = 1
beam.U[-2, 1] = 1

# Normalise eigenmodes
beam.U[-5, 0] = 1/np.sqrt(beam.U[:, 0].T.dot(beam.Mstr.dot(beam.U[:, 0])))
beam.U[-2, 1] = 1/np.sqrt(beam.U[:, 1].T.dot(beam.Mstr.dot(beam.U[:, 1])))

beam.modal = True
beam.inout_coords = 'modes'
beam.Nmodes = 2
beam.freq_natural = np.zeros(2)


Crrgrav = beam.linearise_gravity_forces(tsstruct0)
beam.Cstr[-9:, -9:] += Crrgrav


beam.Ccut = beam.U.T.dot(beam.Cstr.dot(beam.U))

beam.assemble()

# Output theta only
theta_out = np.zeros((1, 4))
theta_out[-1, -1] = 1
beam.SScont.addGain(theta_out, where='out')

# Simulation
ltisystem = sc.signal.lti(beam.SScont.A,
                       beam.SScont.B,
                       beam.SScont.C,
                       beam.SScont.D)

eigs = np.linalg.eig(beam.SScont.A)

T = 4
N = 4000
theta_init = 5*np.pi/180
t_dom = np.linspace(0, T, N)
U = np.zeros((N, 2))
X0 = np.zeros(4)
X0[-1] = theta_init

out = ltisystem.output(U, t_dom, X0)

fig = plt.figure()
plt.plot(t_dom, out[1] * 180 / np.pi)
plt.xlabel('Time, t [s]')
plt.ylabel(r'Angular Displacement, $\theta$ [deg]')
plt.savefig('./PLOTS/LinearRigid.pdf')

print('End of Routine')