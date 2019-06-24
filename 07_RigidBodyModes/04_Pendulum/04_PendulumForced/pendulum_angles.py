import numpy as np
import sys
sys.path.append('/home/ng213/code/sharpy/')
import matplotlib.pyplot as plt
import scipy as sc
import sharpy.linear.src.lingebm as lingebm
import pandas as pd
import sharpy.sharpy_main

def plot_mode(U):
    num_node = int((U.shape[0] - 9) / 6)
    Ux = np.array([U[6*i] for i in range(num_node-1)])
    Uz = np.array([U[6*i + 2] for i in range(num_node-1)])

    plt.figure()
    plt.plot(Ux, Uz)
    plt.show()


solver_file_path = '../cases/flexible_beam_static.solver.txt'

data = sharpy.sharpy_main.main(['', solver_file_path])

tsstruct0 = data.structure.timestep_info[-1]

beam_settings = {'modal_projection': True,
                 'inout_coords': 'modes',
                 'discrete_time': False,
                 'proj_modes': 'undamped',
                 'use_euler': True,
                 'num_modes': 13}

# Modify the A frame orientation from the NL SHARPy solution
euler_init = np.array([5*np.pi/180, 5*np.pi/180, 5*np.pi/180])
# tsstruct0.quat = algebra.euler2quat_ag(euler_init)
# tsstruct0.mb_quat[0] = algebra.euler2quat_ag(euler_init)
tsstruct0.quat = tsstruct0.mb_quat[0]

beam = lingebm.FlexDynamic(tsstruct0, structure=data.structure,
                           custom_settings=beam_settings)

beam.update_modal()
beam.Nmodes = 7

# Retain the rotational rigid degrees of freedom
phi_rr = np.zeros((69, 6))

# Rigid body modes of interest
phi_rr[-6:, :] = np.eye(6)

# Get the flexible modes
phi = beam.U[:, 10:11]
phi = np.hstack((phi, phi_rr))

beam.U = phi
beam.freq_natural = np.array([beam.freq_natural[10], 0, 0, 0, 0, 0, 0])  # Update natural frequencies

# Normalise eigenmodes
for i in range(len(beam.freq_natural)):
    diag_factor = beam.U[:, i].T.dot(beam.Mstr.dot(beam.U[:, i]))
    beam.U[:, i] = 1 / np.sqrt(diag_factor) * beam.U[:, i]

beam.linearise_gravity_forces(tsstruct0)

beam.assemble()

eigs = np.linalg.eig(beam.SScont.A)

out_matrix = np.zeros((3, beam.SScont.outputs))
out_matrix[:, -3:] = np.eye(3)
beam.SScont.addGain(out_matrix, where='out')

# Simulation --------------------------------------------------------------------------------------------
ltisystem = sc.signal.lti(beam.SScont.A,
                       beam.SScont.B,
                       beam.SScont.C,
                       beam.SScont.D)

T = 4  # Total time
N = 4000  # Number of time steps

theta_init = euler_init[1]

# Initial condition - modal coordinate
eta0 = tsstruct0.q[:-4]  # Remove the quaternion from the original state vector
eta0 = np.concatenate((eta0, np.array([0, 0, 0])))

 # Initial condition - modal coordinate derivative
eta_dot_0 = np.zeros_like(eta0)
eta_dot_0[-3:] = euler_init
q0 = beam.U.T.dot(eta0)
q0_dot = beam.U.T.dot(eta_dot_0)
q0 = np.zeros_like(q0_dot)

t_dom = np.linspace(0, T, N)
U = np.zeros((N, beam.SScont.inputs))
# U[0, 3] = 0.1
X0 = np.zeros(beam.SScont.states)
X0[:beam.Nmodes] = q0
X0[beam.Nmodes:] = q0_dot

out = ltisystem.output(U, t_dom, X0)

phi = out[1][:, 0]
theta = out[1][:, 1]
psi = out[1][:, 2]


# Non linear results
datanl = pd.read_csv('/home/ng213/sharpy_cases/07_RigidBodyModes/04_Pendulum/PLOTS/nonlinear_theta5_sim.csv')
dt = T/N
t_sim = datanl['Time'] * dt
lx0 = datanl['local_x (0) (stats)']
lx2 = datanl['local_x (2) (stats)']

theta_nl = np.arctan(-lx0/lx2) * 180 / np.pi

fig, ax = plt.subplots(nrows=3, sharex=True, constrained_layout=True)
ax[0].plot(t_dom, phi * 180 / np.pi, color='k', ls='-', label='LinearFlexible')

ax[1].plot(t_dom, theta * 180 / np.pi, color='k', ls='-', label='LinearFlexible')
# ax[1].plot(t_sim, theta_nl, color='b', lw=4, alpha=0.5, label='NonlinearFlex')

ax[2].plot(t_dom, psi * 180 / np.pi, color='k', ls='-', label='LinearFlexible')

ax[2].set_xlabel('Time, t [s]')
ax[0].set_ylabel(r'$\phi$ [deg]')
ax[1].set_ylabel(r'$\theta$ [deg]')
ax[2].set_ylabel(r'$\psi$ [deg]')
# plt.grid()
# plt.legend()
# fig.show()
plt.savefig('/home/ng213/sharpy_cases/07_RigidBodyModes/04_Pendulum/PLOTS/LinearFlex_Init_RollYawPitch.pdf')

print('End of Routine')