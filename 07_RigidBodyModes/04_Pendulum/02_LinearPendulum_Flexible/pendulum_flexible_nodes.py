import numpy as np
import sys
sys.path.append('/home/ng213/code/sharpy/')
import matplotlib.pyplot as plt
import scipy.linalg as sclalg
import scipy as sc
import sharpy.linear.src.lingebm as lingebm

import sharpy.sharpy_main


solver_file_path = '../cases/flexible_beam_static.solver.txt'

data = sharpy.sharpy_main.main(['', solver_file_path])

tsstruct0 = data.structure.timestep_info[-1]

# dt = 0.001

beam_settings = {'modal_projection': False,
                 'inout_coords': 'nodes',
                 'discrete_time': False,
                 'proj_modes': 'undamped',
                 'use_euler': True}

beam = lingebm.FlexDynamic(tsstruct0, structure=data.structure, custom_settings=beam_settings)

# ## Comparing eigenvalues
# e_modal = beam.eigs
# A_m = np.zeros((2*beam.Mstr.shape[0], 2*beam.Mstr.shape[0]))
# num_dof = beam.Mstr.shape[0]
# A_m[num_dof:, :num_dof] = -np.linalg.inv(beam.Mstr).dot(beam.Kstr)
# A_m[num_dof:, num_dof:] = -np.linalg.inv(beam.Mstr).dot(beam.Cstr)*0
# A_m[:num_dof, num_dof:] = np.eye(num_dof)
# e_A = np.linalg.eig(A_m)[0]
# order = np.argsort(e_A.imag)
# e_A = e_A[order]
# omega_modal = np.sqrt(e_modal[e_modal>0])
# omega_A = e_A[e_A.imag > 0].imag
#
# omega_diff = np.abs(omega_modal.T-omega_A[:len(omega_modal)])
#
# fig = plt.figure
# plt.scatter(e_A.real, e_A.imag, marker='s', color='r',label='StateSpace')
# plt.scatter(np.zeros_like(omega_modal), omega_modal, marker='x', color='k')
# plt.scatter(np.zeros_like(omega_modal), -omega_modal, marker='x', color='k', label='Modal')
# plt.legend()
# # plt.ylim([-np.max(omega_modal), np.max(omega_modal)])
# plt.ylim([-2000, 2000])
# # plt.savefig('./Evals.pdf')
# ## End comparison


# beam.Cstr[:-3, :] = beam.Cstr[:-3, :] * 0
# beam.Cstr = beam.Cstr * 0

# beam.modal = False
# beam.inout_coords = 'nodes'

beam.update_modal()

# Remove rigid degrees of freedom except omega_y and theta
phi_rr = np.zeros((9, 2))
# phi_rr = np.eye(9)

# omega_y mode
phi_rr[-5, 0] = 1
phi_rr[-2, 1] = 1

# for i_node in tsstruct0.num_node:

beam.linearise_gravity_forces(tsstruct0)

phi = sclalg.block_diag(np.eye(60), phi_rr)

A_m = np.zeros((2*beam.Mstr.shape[0], 2*beam.Mstr.shape[0]))
num_dof = beam.Mstr.shape[0]
A_m[num_dof:, :num_dof] = -np.linalg.inv(beam.Mstr).dot(beam.Kstr)
A_m[num_dof:, num_dof:] = -np.linalg.inv(beam.Mstr).dot(beam.Cstr)
A_m[:num_dof, num_dof:] = np.eye(num_dof)
e_A = np.linalg.eig(A_m)[0]
print(np.max(e_A.real))
# phi = np.eye(69,60)

# Normalise eigenmodes
# for i in range(beam.Nmodes):
#     beam.U[:, i] = 1/np.sqrt(beam.U[:, i].T.dot(beam.Mstr.dot(beam.U[:, i])))
# for i in range(len(beam.freq_natural)):
#     diag_factor = beam.U[:, i].T.dot(beam.Mstr.dot(beam.U[:, i]))
#     beam.U[:, i] = 1 / np.sqrt(diag_factor) * beam.U[:, i]


beam.Mstr = phi.T.dot(beam.Mstr.dot(phi))
beam.Cstr = phi.T.dot(beam.Cstr.dot(phi))
beam.Kstr = phi.T.dot(beam.Kstr.dot(phi))

# beam.Ccut = beam.U.T.dot(beam.Cstr.dot(beam.U))

beam.assemble()

eigs = np.linalg.eig(beam.SScont.A)

order = np.argsort(eigs[0].real)[::-1]
eigvals = eigs[0][order]
evecs = eigs[1][:, order]

# Output theta only
theta_out = np.zeros((1, beam.SScont.outputs))
theta_out[-1, -1] = 1

# theta_out = np.zeros((2, beam.SScont.outputs))
# theta_out[0, 2] = 1
# theta_out[1, 0] = 1
beam.SScont.addGain(theta_out, where='out')

# Simulation
ltisystem = sc.signal.lti(beam.SScont.A,
                       beam.SScont.B,
                       beam.SScont.C,
                       beam.SScont.D)

T = 4
N = 4000
theta_init = 5*np.pi/180
t_dom = np.linspace(0, T, N)
U = np.zeros((N, beam.SScont.inputs))
# U[0, -1] = theta_init

# Initial position
q0 = tsstruct0.q[:-10]
q0x = np.array([q0[6*i] for i in range(tsstruct0.num_node-1)])
q0z = np.array([q0[6*i+2] for i in range(tsstruct0.num_node-1)])
X0 = np.zeros(beam.SScont.states)
X0[:60] = q0
# X0[-1] = 0*-theta_init

out = ltisystem.output(U, t_dom, X0)

# theta = np.arctan(-out[1][:,1]/out[1][:, 0])
theta = out[1]
qout = out[2]

x_tip = qout[:, 54]

fig = plt.figure()
plt.plot(t_dom, theta * 180 / np.pi)
plt.xlabel('Time, t [s]')
plt.ylabel(r'Angular Displacement, $\theta$ [deg]')
plt.show()
# plt.savefig('./LinearFlexible.pdf')

fig = plt.figure()
# for i in range(tsstruct0.num_node-1):

plt.plot(t_dom, x_tip)
    # plt.plot(t_dom, qout[:, 6*i+2])
plt.xlabel('Time, t [s]')
# plt.ylabel(r'Angular Displacement, $\theta$ [deg]')
plt.show()
# plt.savefig('./LinearFlexible.pdf')
print('End of Routine')