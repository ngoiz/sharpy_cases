import numpy as np
import sys
sys.path.append('/home/ng213/code/sharpy/')
import matplotlib.pyplot as plt
import scipy as sc
import sharpy.linear.src.lingebm as lingebm
import pandas as pd

import sharpy.sharpy_main


solver_file_path = '../cases/flexible_beam_static.solver.txt'

data = sharpy.sharpy_main.main(['', solver_file_path])

tsstruct0 = data.structure.timestep_info[-1]

# dt = 0.0001
beam_settings = {'modal_projection': True,
                 'inout_coords': 'modes',
                 'discrete_time': False,
                 'proj_modes': 'undamped',
                 'use_euler': True,
                 'num_modes': 13}
beam = lingebm.FlexDynamic(tsstruct0, structure=data.structure, custom_settings=beam_settings)

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


beam.linearise_gravity_forces(tsstruct0)
# beam.Cstr[-9:, -9:] += Crrgrav


beam.Ccut = beam.U.T.dot(beam.Cstr.dot(beam.U))

beam.assemble()

# Remove integrals
rem_int = np.zeros((4, 2))
rem_int[-2:, -2:] = np.eye(2)
if beam.SScont:
    beam.SScont.A = rem_int.T.dot(beam.SScont.A.dot(rem_int))
    beam.SScont.B = rem_int.T.dot(beam.SScont.B)
    beam.SScont.C = beam.SScont.C.dot(rem_int)

    # Output theta only
    theta_out = np.zeros((1, 4))
    theta_out[-1, -1] = 1
    beam.SScont.addGain(theta_out, where='out')
else:
    beam.SSdisc.A = rem_int.T.dot(beam.SSdisc.A.dot(rem_int))
    beam.SSdisc.B = rem_int.T.dot(beam.SSdisc.B)
    beam.SSdisc.C = beam.SSdisc.C.dot(rem_int)

    # Output theta only
    theta_out = np.zeros((1, 4))
    theta_out[-1, -1] = 1
    beam.SSdisc.addGain(theta_out, where='out')

# Simulation
if beam.SScont:
    ltisystem = sc.signal.lti(beam.SScont.A,
                           beam.SScont.B,
                           beam.SScont.C,
                           beam.SScont.D)
    eigs = np.linalg.eig(beam.SScont.A)
else:
    ltisystem = sc.signal.dlti(beam.SSdisc.A,
                              beam.SSdisc.B,
                              beam.SSdisc.C,
                              beam.SSdisc.D,
                               dt=beam.SSdisc.dt)
    eigs = np.linalg.eig(beam.SSdisc.A)
    eig_vals = np.log(eigs[0])/beam.SSdisc.dt

T = 4
N = 4000
theta_init = 5*np.pi/180
t_dom = np.linspace(0, T, N)
U = np.zeros((N, 2))
X0 = np.zeros(2)
X0[-1] = theta_init

if beam.SScont:
    out = ltisystem.output(U, t_dom, X0)
else:
    out = sc.signal.dlsim(ltisystem, U, t_dom, X0)


# Non linear results
datanl = pd.read_csv('/home/ng213/sharpy_cases/07_RigidBodyModes/04_Pendulum/PLOTS/nonlinear_theta5_sim.csv')
dt = T/N
t_sim = datanl['Time'] * dt
lx0 = datanl['local_x (0) (stats)']
lx2 = datanl['local_x (2) (stats)']

theta_nl = np.arctan(-lx0/lx2) * 180 / np.pi

fig = plt.figure()
plt.plot(out[0], out[1] * 180 / np.pi, color='k', lw=1.5, ls='--', label='LinearRigid')
plt.plot(t_sim, theta_nl, color='b', lw=4, alpha=0.5, label='NonlinearFlex')
plt.xlabel('Time, t [s]')
plt.ylabel(r'Angular Displacement, $\theta$ [deg]')
plt.grid()
plt.legend()

plt.show()
# plt.savefig('/home/ng213/sharpy_cases/07_RigidBodyModes/04_Pendulum/PLOTS/LinearRigid.pdf')

print('End of Routine')