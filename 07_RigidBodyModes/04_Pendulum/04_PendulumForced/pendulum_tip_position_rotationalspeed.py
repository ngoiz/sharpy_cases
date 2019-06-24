import numpy as np
import sys
sys.path.append('/home/ng213/code/sharpy/')
import matplotlib.pyplot as plt
import scipy as sc
import sharpy.linear.src.lingebm as lingebm
import pandas as pd
import sharpy.utils.algebra as algebra
import sharpy.sharpy_main
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.rc('font', family='serif', serif='Times')
# plt.rc('text', usetex=True)
# plt.rc('xtick', labelsize=8)
# plt.rc('ytick', labelsize=8)
# plt.rc('axes', labelsize=8)

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

# beam_settings = {'modal_projection': True,
#                  'inout_coords': 'modes',
#                  'discrete_time': False,
#                  'proj_modes': 'undamped',
#                  'use_euler': True,
#                  'num_modes': 13}

beam_settings = {'modal_projection': False,
                 'inout_coords': 'nodes',
                 'discrete_time': True,
                 'newmark_damp': 0.15*1,
                 'discr_method': 'newmark',
                 'dt': 0.001,
                 'proj_modes': 'undamped',
                 'use_euler': True,
                 'num_modes': 13}

# Modify the A frame orientation from the NL SHARPy solution
euler_init = np.array([0*np.pi/180, 5*np.pi/180, 0*np.pi/180])
# tsstruct0.quat = algebra.euler2quat_ag(euler_init)
# tsstruct0.mb_quat[0] = algebra.euler2quat_ag(euler_init)
tsstruct0.quat = tsstruct0.mb_quat[0]

beam = lingebm.FlexDynamic(tsstruct0, structure=data.structure,
                           custom_settings=beam_settings)

# beam.update_modal()
# beam.Nmodes = 7

# Retain the rotational rigid degrees of freedom
phi_rr = np.zeros((69, 6))

phi_rot = np.eye(beam.Mstr.shape[0])
phi_rot[-9:-6] = 0

beam.linearise_gravity_forces()
beam.Mstr = phi_rot.dot(beam.Mstr)
beam.Cstr = phi_rot.dot(beam.Cstr)
beam.Kstr = phi_rot.dot(beam.Kstr)

def rem_v(M):
    Mss = M[:-9, :-9]
    Msr = M[:-9, -6:]
    Mrs = M[-6:, :-9]
    Mrr = M[-6:, -6:]
    return np.block([[Mss, Msr], [Mrs, Mrr]])

beam.Mstr = rem_v(beam.Mstr)
beam.Cstr = rem_v(beam.Cstr)
beam.Kstr = rem_v(beam.Kstr)

# Rigid body modes of interest
phi_rr[-6:, :] = np.eye(6)


beam.assemble()

eigs = np.linalg.eig(beam.SSdisc.A)
order = np.argsort(np.abs(eigs[0]))[::-1]
eigs_ct = np.log(eigs[0])/beam.SSdisc.dt
eigs_ct = eigs_ct[order]
evecs = eigs[1][:, order]

# out_matrix = np.zeros((6, beam.SSdisc.outputs))
# out_matrix[:3, 54:57] = np.eye(3)
# out_matrix[3:, -3:] = np.eye(3)

out_matrix = np.eye(beam.SSdisc.outputs)

# out_matrix[:, -3:] = np.eye(3)
beam.SSdisc.addGain(out_matrix, where='out')

# Simulation --------------------------------------------------------------------------------------------
ltisystem = sc.signal.dlti(beam.SSdisc.A,
                       beam.SSdisc.B,
                       beam.SSdisc.C,
                       beam.SSdisc.D,
                        dt=beam.SSdisc.dt)

T = 4  # Total time
N = 4000  # Number of time steps

theta_init = euler_init[1]

# Initial condition - move from linearisation point
rot = algebra.euler2rotation_ag(euler_init)
eta0 = tsstruct0.q[:-4]  # Remove the quaternion from the original state vector
# for i in range(tsstruct0.num_node-1):
#     jj_tra = 6*i + np.array([0, 1, 2])
#     orig_eta = eta0[jj_tra]
#     eta_d = rot.dot(orig_eta)
#     eta0[jj_tra] = eta_d

eta0 = np.concatenate((eta0[:-6], np.array([0, 0, 0, 0, 0, 0])))

 # Initial condition - modal coordinate derivative
eta_dot_0 = np.zeros_like(eta0)
eta_dot_0[-3:] = euler_init
# eta_dot_0[-4] = 0.1


# t_dom = np.linspace(0, T, N+1)
U = np.zeros((N, beam.SSdisc.inputs))
U[:, 55] = 10
X0 = np.zeros(beam.SSdisc.states)
X0[:beam.SSdisc.states//2] = eta0*0
X0[beam.SSdisc.states//2:] = eta_dot_0

out = sc.signal.dlsim(ltisystem, U, x0=X0)

t_dom = out[0]
# x = out[1][:, 0]
# y = out[1][:, 1]
# z = out[1][:, 2]
# theta = out[1][:, 4]
eta_out = eta0 + out[1][:, :beam.SSdisc.states//2]
eta_dot_out = out[1][:, beam.SSdisc.states//2:]

x = eta_out[:, 54]
y = eta_out[:, 55]
z = eta_out[:, 56]
euler_out = eta_dot_out[:, -3:]

xg = np.zeros(len(t_dom))
yg = np.zeros(len(t_dom))
zg = np.zeros(len(t_dom))

for n in range(N):
    euler = out[1][n, -3:]
    Cag = algebra.euler2rotation_ag(euler)
    Cga = Cag.T
    pos = Cag.dot(np.array([x[n], y[n], z[n]]))  # Project xa onto xg (thats why Cag is used)
    xg[n] = pos[0]
    yg[n] = pos[1]
    zg[n] = pos[2]


# Non linear results
datanl = pd.read_csv('/home/ng213/sharpy_cases/07_RigidBodyModes/04_Pendulum/PLOTS/nonlinear_theta5_sim.csv')
dt = T/N
t_sim = datanl['Time'] * dt
lx0 = datanl['local_x (0) (stats)']
lx1 = datanl['local_x (1) (stats)']
lx2 = datanl['local_x (2) (stats)']
xg_nl = -datanl['X (stats)']
yg_nl = datanl['Y (stats)']
zg_nl = datanl['Z (stats)']
xa_nl_o = datanl['coords_a (0) (stats)']
ya_nl_o = datanl['coords_a (1) (stats)']
za_nl_o = datanl['coords_a (2) (stats)']
phi_nl = np.arctan(-lx1/lx2) * 180 / np.pi
theta_nl = np.arctan(-lx0/lx2) * 180 / np.pi
psi_nl = np.arctan(-lx1/lx0) * 180 / np.pi

# Original NL sim is with the nodes rotated wrt to the A frame
xa_nl = np.zeros(len(t_sim))
ya_nl = np.zeros(len(t_sim))
za_nl = np.zeros(len(t_sim))
for n in range(len(t_sim)):
    euler_nl = np.array([phi_nl[0], theta_nl[0], psi_nl[0]])*np.pi/180
    rot = algebra.euler2rotation_ag(-euler_nl)
    pos = rot.T.dot(np.array([xa_nl_o[n], ya_nl_o[n], za_nl_o[n]]))
    xa_nl[n] = pos[0]
    ya_nl[n] = pos[1]
    za_nl[n] = pos[2]

fig, ax = plt.subplots(nrows=3, sharex=True)
fig.subplots_adjust(left=.25, bottom=.16, right=.99, top=.97)
ax[0].plot(t_dom, x, color='k', ls='-', label='LinearFlexible')
# ax[0].plot(t_sim, xa_nl, color='b', lw=4, alpha=0.5, label='NonlinearFlex')
ax[1].plot(t_dom, y, color='k', ls='-', label='LinearFlexible')
# ax[1].plot(t_sim, ya_nl, color='b', lw=4, alpha=0.5, label='NonlinearFlex')
ax[2].plot(t_dom, z, color='k', ls='-', label='LinearFlexible')
# ax[2].plot(t_sim, za_nl, color='b', lw=4, alpha=0.5, label='NonlinearFlex')

ax[2].set_xlabel('Time, t [s]')
ax[0].set_ylabel(r'$x_A$')
ax[1].set_ylabel(r'$y_A$')
ax[2].set_ylabel(r'$z_A$')
# plt.grid()
# plt.legend()
# plt.savefig('/home/ng213/sharpy_cases/07_RigidBodyModes/04_Pendulum/PLOTS/LinearFlex_BodyFrame.pdf')
fig.show()

fig, ax = plt.subplots(nrows=3, sharex=True)
fig.subplots_adjust(left=.25, bottom=.16, right=.99, top=.97)
ax[0].plot(t_dom, xg, color='k', ls='-', label='LinearFlexible')
# ax[0].plot(t_sim, xg_nl, color='b', lw=4, alpha=0.5, label='NonlinearFlex')

ax[1].plot(t_dom, yg, color='k', ls='-', label='LinearFlexible')
# ax[1].plot(t_sim, yg_nl, color='b', lw=4, alpha=0.5, label='NonlinearFlex')

ax[2].plot(t_dom, zg, color='k', ls='-', label='LinearFlexible')
# ax[2].plot(t_sim, zg_nl, color='b', lw=4, alpha=0.5, label='NonlinearFlex')

ax[2].set_xlabel('Time, t [s]')
ax[0].set_ylabel(r'$x_G$')
ax[1].set_ylabel(r'$y_G$')
ax[2].set_ylabel(r'$z_G$')
# plt.grid()
# plt.legend()
# plt.savefig('/home/ng213/sharpy_cases/07_RigidBodyModes/04_Pendulum/PLOTS/LinearFlex_InertialFrame.pdf')
fig.show()

fig, ax = plt.subplots(nrows=3, sharex=True)
fig.subplots_adjust(left=.25, bottom=.16, right=.99, top=.97)
ax[0].plot(t_dom, euler_out[:, 0]*180/np.pi, color='k', ls='-', label='LinearFlexible')
# ax[0].plot(t_sim, phi_nl, color='b', lw=4, alpha=0.5, label='NonlinearFlex')
ax[1].plot(t_dom, euler_out[:, 1]*180/np.pi, color='k', ls='-', label='LinearFlexible')
# ax[1].plot(t_sim, theta_nl, color='b', lw=4, alpha=0.5, label='NonlinearFlex')
ax[2].plot(t_dom, euler_out[:, 2]*180/np.pi, color='k', ls='-', label='LinearFlexible')
# ax[2].plot(t_sim, psi_nl, color='b', lw=4, alpha=0.5, label='NonlinearFlex')
ax[0].set_ylabel(r'$\phi$')
ax[1].set_ylabel(r'$\theta$')
ax[2].set_ylabel(r'$\psi$')
ax[2].set_xlabel('Time, t [s]')
# plt.savefig('/home/ng213/sharpy_cases/07_RigidBodyModes/04_Pendulum/PLOTS/LinearFlex_EulerAngles.pdf',
#             bbox_inches="tight")
plt.show()

print('End of Routine')