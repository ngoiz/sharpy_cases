import numpy as np
import sys
sys.path.append('/home/ng213/code/sharpy/')
import matplotlib.pyplot as plt
import scipy as sc
import sharpy.linear.src.lingebm as lingebm
import pandas as pd
import sharpy.utils.algebra as algebra
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

beam_settings = {'modal_projection': true,
                 'inout_coords': 'modes',
                 'discrete_time': true,
                 'newmark_damp': 0.15*0,
                 'discr_method': 'newmark',
                 'dt': 0.001,
                 'proj_modes': 'undamped',
                 'use_euler': True,
                 'num_modes': 13}

# Modify the A frame orientation from the NL SHARPy solution
euler_init = np.array([0, 5*np.pi/180, 0])
tsstruct0.quat = algebra.euler2quat_ag(euler_init)
tsstruct0.mb_quat[0] = algebra.euler2quat_ag(euler_init)

beam = lingebm.FlexDynamic(tsstruct0, structure=data.structure,
                           custom_settings=beam_settings)

beam.update_modal()
beam.Nmodes = 3

# Retain the rotational rigid degree of freedom only through the two rigid body modes
phi_rr = np.zeros((69, 2))

# Rigid body modes of interest
phi_rr[-5, 0] = 1  # Omega_y mode
phi_rr[-2, 1] = 1  # Theta mode

# Get the flexible modes
phi = beam.U[:, 10:10+beam.Nmodes-2]
phi = np.hstack((phi, phi_rr))

beam.U = phi
beam.freq_natural = np.array([beam.freq_natural[10], 0, 0])  # Update natural frequencies

# Normalise eigenmodes
for i in range(beam.U.shape[1]):
    diag_factor = beam.U[:, i].T.dot(beam.Mstr.dot(beam.U[:, i]))
    beam.U[:, i] = 1 / np.sqrt(diag_factor) * beam.U[:, i]

beam.linearise_gravity_forces(tsstruct0)

beam.assemble()

theta_out = np.zeros((1, beam.SSdisc.outputs))
theta_out[-1, -1] = 1
beam.SSdisc.addGain(theta_out, where='out')

# Simulation --------------------------------------------------------------------------------------------
ltisystem = sc.signal.dlti(beam.SSdisc.A,
                       beam.SSdisc.B,
                       beam.SSdisc.C,
                       beam.SSdisc.D, dt=0.001)

T = 4  # Total time
N = 4000  # Number of time steps

theta_init = euler_init[1]

# Initial condition - modal coordinate
eta0 = tsstruct0.q[:-4]  # Remove the quaternion from the original state vector
eta0 = np.concatenate((eta0, np.array([0, 0, 0])))

 # Initial condition - modal coordinate derivative
eta_dot_0 = np.zeros_like(eta0)
eta_dot_0[-3:] = np.array([0, theta_init, 0])
q0 = beam.U.T.dot(eta0)
q0_dot = beam.U.T.dot(eta_dot_0)
q0 = np.zeros_like(q0_dot)

t_dom = np.linspace(0, T, N + 1)
U = np.zeros((N + 1, beam.SSdisc.inputs))
X0 = np.zeros(beam.SSdisc.states)
X0[:beam.Nmodes] = q0
X0[beam.Nmodes:] = q0_dot

out = ltisystem.output(U, t_dom, X0)


# Non linear results
datanl = pd.read_csv('/home/ng213/sharpy_cases/07_RigidBodyModes/04_Pendulum/PLOTS/nonlinear_theta5_sim.csv')
dt = T/N
t_sim = datanl['Time'] * dt
lx0 = datanl['local_x (0) (stats)']
lx2 = datanl['local_x (2) (stats)']

theta_nl = np.arctan(-lx0/lx2) * 180 / np.pi

fig = plt.figure()
plt.plot(t_dom, out[1] * 180 / np.pi, color='k', ls='--', label='LinearFlexible')
plt.plot(t_sim, theta_nl, color='b', lw=4, alpha=0.5, label='NonlinearFlex')
plt.xlabel('Time, t [s]')
plt.ylabel(r'Angular Displacement, $\theta$ [deg]')
plt.grid()
plt.legend()
plt.show()
# plt.savefig('/home/ng213/sharpy_cases/07_RigidBodyModes/04_Pendulum/PLOTS/LinearFlexible.pdf')

print('End of Routine')