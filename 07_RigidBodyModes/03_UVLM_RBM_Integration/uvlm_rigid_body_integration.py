
import numpy as np
import os
import sys
sys.path.append('/home/ng213/code/sharpy/')
import matplotlib.pyplot as plt
import sharpy.linear.src.libsparse as libsp
import scipy.linalg as sclalg
import scipy as sc
import sharpy.linear.src.lin_aeroelastic as linaeroelastic
import sharpy.linear.src.libss as libss

import cases.templates.flying_wings as wings
import sharpy.sharpy_main

# Problem Set up
u_inf = 10.
alpha_deg = 4
rho = 1.02
use_euler = True
if use_euler:
    num_modes = 9
    rigid_dof = 9
else:
    num_modes = 10
    rigid_dof = 10
num_flex_modes = num_modes - rigid_dof

# Lattice Discretisation
M = 4
N = 12
M_star_fact = 10
# #
# M = 16
N = 24
# M_star_fact = 10

# Linear UVLM settings
integration_order = 2
remove_predictor = False
use_sparse = False

# ROM Properties
algorithm = 'mimo_rational_arnoldi'
r = 2
frequency_continuous_k = np.array([0.])

# Case Admin - Create results folders
case_name = 'goland'
case_nlin_info = 'M%dN%dMs%d_nmodes%d' %(M, N, M_star_fact, num_modes)
case_rom_info = 'rom_MIMORA_r%d_sig%04d_%04dj' % (r, frequency_continuous_k[0].real*1000, frequency_continuous_k[0].imag*1000)
fig_folder = './figures/'
os.system('mkdir -p %s' % fig_folder)


# SHARPy nonlinear reference solution
ws = wings.Goland(M=M,
                  N=N,
                  Mstar_fact=M_star_fact,
                  u_inf=u_inf,
                  alpha=alpha_deg,
                  beta=0,
                  # aspect_ratio=32,
                  rho=rho,
                  sweep=0,
                  physical_time=2,
                  n_surfaces=2,
                  route='cases',
                  case_name=case_name)

ws.gust_intensity = 0.01
ws.sigma = 1

ws.clean_test_files()
ws.update_derived_params()
ws.update_aero_prop()
ws.n_tstep = 20
ws.update_fem_prop()
ws.set_default_config_dict()

ws.generate_aero_file()
ws.generate_fem_file()

ws.config['SHARPy']['flow'] = ['BeamLoader', 'AerogridLoader',
                               # 'StaticUvlm',
                               'StaticCoupled',
                               'AerogridPlot', 'BeamPlot',
                               # 'DynamicCoupled',
                               'Modal',
                               'SaveData']
ws.config['SHARPy']['write_screen'] = 'on'
ws.config['Modal']['NumLambda'] = 40
ws.config['Modal']['rigid_body_modes'] = True
ws.config['StaticCoupled']['structural_solver_settings']['gravity_on'] = True
ws.config['DynamicCoupled']['aero_solver_settings']['velocity_field_input']['gust_length'] = 5
ws.config.write()

data = sharpy.sharpy_main.main(['',ws.route+ws.case_name+'.solver.txt'])


# Linearise - reduce UVLM and project onto modes
scaling_factors = {'length': 1,  #0.5 * ws.c_ref,
                   'speed': 1,  #u_inf,
                   'density': 1} #rho}
dt = ws.dt
lin_settings = {'dt': dt,
                'integr_order': integration_order,
                'density': rho,
                'remove_predictor': remove_predictor,
                'use_sparse': use_sparse,
                'ScalingDict': scaling_factors,
                'rigid_body_motion': True,
                'use_euler': use_euler}

# Linearisation
# Original data point
tsaero0 = data.aero.timestep_info[-1]
tsstruct0 = data.structure.timestep_info[-1]
tsaero0.rho = rho
flex_dof = data.structure.num_dof.value

# Create aeroelastic system
aeroelastic_system = linaeroelastic.LinAeroEla(data, lin_settings)

beam = aeroelastic_system.lingebm_str
uvlm = aeroelastic_system.linuvlm

# Parametrise orientation in terms of euler angles
if use_euler:
    beam.Mstr = beam.Mstr[:-1, :-1]
    beam.Cstr = beam.Cstr[:-1, :-1]
    beam.Kstr = beam.Kstr[:-1, :-1]

# structure to UVLM gains
aeroelastic_system.get_gebm2uvlm_gains()

beam.Kstr[:flex_dof, :flex_dof] += aeroelastic_system.Kss

C_rigid_effects = np.block([[np.zeros((flex_dof, flex_dof)), aeroelastic_system.Csr],
                            [aeroelastic_system.Crs, aeroelastic_system.Crr]])

beam.Cstr[-rigid_dof:, :] = 0  # Erase quaternion equations - using euler parametrisation
beam.Cstr += C_rigid_effects

# Get rid of all dofs except the rolling motion
beam.update_modal()

# Assemble Ksa and Kas
zero_matrix = np.zeros((3*uvlm.Kzeta, beam.num_dof))
Kas = np.block([[aeroelastic_system.Kdisp, zero_matrix],
                [aeroelastic_system.Kvel_disp, aeroelastic_system.Kvel_vel]])

Ksa = aeroelastic_system.Kforces

# Assemble UVLM
uvlm.assemble_ss()

# UVLM remove gust input - gusts not required in analysis
uvlm.SS.B = libsp.csc_matrix(aeroelastic_system.linuvlm.SS.B[:, :6*aeroelastic_system.linuvlm.Kzeta])
uvlm.SS.D = libsp.csc_matrix(aeroelastic_system.linuvlm.SS.D[:, :6*aeroelastic_system.linuvlm.Kzeta])

# Extract the state information: add np.eye(K) to ulvm.SS.C
gamma_out_matrix = np.zeros((uvlm.K, 3*uvlm.K + uvlm.K_star))
gamma_out_matrix[:uvlm.K, :uvlm.K] = np.eye(uvlm.K)
uvlm.SS.C = np.vstack((uvlm.SS.C, gamma_out_matrix))


zero_matrix_D = np.zeros((uvlm.K, uvlm.SS.D.shape[1]))
print(uvlm.SS.D.shape)
print(zero_matrix_D.shape)
uvlm.SS.D = np.vstack((uvlm.SS.D.todense(), zero_matrix_D))

# Add coupling matrices - Input
uvlm.SS.addGain(Kas, where='in')

# Add output matrix (to nodal forces)
Ksa_gamma = sclalg.block_diag(Ksa, np.eye(uvlm.K))
uvlm.SS.addGain(Ksa_gamma, where='out')

# Keep only the omega_x mode
# beam.freq_natural = np.array([0.])
# beam.U = np.zeros((flex_dof+rigid_dof, 1))
# beam.U[-6] = 1/np.sqrt(beam.Mstr[-6, -6]) # Scale evect for unity


beam.dlti = True
beam.newmark_damp = 5e-3
beam.modal = True
beam.proj_modes = 'undamped'
beam.Nmodes = 1
beam.discr_method = 'newmark'
beam.inout_coords = 'modes'

# beam.assemble()


# Mode 1
# Keep only the Vx mode
label_list_integr = [r'$\bar{V_x}$',r'$\bar{V_y}$', r'$\bar{V_z}$',
                     r'$\Phi$', r'$\Theta$', r'$\Psi$',
                     r'$\bar{\Phi}$', r'$\bar{\Theta}$', r'$\bar{\Psi}$']
label_list_veloc = [r'$V_x$', r'$V_y$', r'$V_z$',
                    r'$\omega_x$',r'$\omega_y$',r'$\omega_z$',
                    r'$\Phi$', r'$\Theta$', r'$\Psi$']
label_list_force = ['X', 'Y', 'Z', 'L', 'M', 'N', 'QL', 'QM', 'QN']

mode = 6

beam.freq_natural = np.array([0.])
beam.U = np.zeros((flex_dof+rigid_dof, 1))
beam.U[-9+mode] = 1/np.sqrt(beam.Mstr[-9+mode, -9+mode])  # Scale evect for unity

# uvlm.assemble_ss()
# Assemble UVLM
# uvlm.SS.addGain(Kas, where='in')
# uvlm.SS.addGain(Ksa, where='out')

beam.assemble()

# Aero to modal and retain nodal forces
phi = beam.U
out_mat = np.vstack((np.eye(uvlm.SS.outputs), np.hstack((phi.T, np.zeros((phi.T.shape[0], uvlm.K))))))
# out_mat = sclalg.block_diag(out_mat, np.eye(uvlm.K))

mod_mat = sclalg.block_diag(phi*0, phi)
uvlm.SS.addGain(mod_mat, where='in')
uvlm.SS.addGain(out_mat, where='out')

# Couple
Tsa = np.zeros((beam.SSdisc.inputs, uvlm.SS.outputs))
Tsa[-1, -1] = 1
Tas = np.eye(2)
ss_coupled = libss.couple(uvlm.SS, beam.SSdisc, Tas, Tsa)  # Coupled system with modal i/o
nodal_gain_out = 1*np.eye(3) * beam.U[-9+mode]
ss_coupled.addGain(nodal_gain_out, where='in')

# A = uvlm.SS.A
# B = uvlm.SS.B
# C = uvlm.SS.C
# D = uvlm.SS.D

A = ss_coupled.A
B = ss_coupled.B
C = ss_coupled.C
D = ss_coupled.D

dlti = sc.signal.dlti(A, B, C, D, dt=ws.dt)
T = 30 #s
out = sc.signal.dstep(dlti, n=np.ceil(T/ws.dt))
tout = out[0]
# Q_out = out[1][1]
phi_out = out[1][0]
Vx_out = out[1][1]
Q_out = out[1][2]

# Aerodynamic forces
def aero_forces(Q_out):
    num_node = data.structure.num_node
    Fz_tf = np.array([Q_out[-1, 6*i+2] for i in range(num_node)])
    qy = np.array([data.structure.timestep_info[0].q[6*i+1] for i in range(num_node)])
    w1 = np.concatenate((np.array([12]), np.arange(0, num_node//2)))
    w2 = np.arange(num_node//2, num_node)
    index_W = np.arange(0, num_node//2+1)

    return qy, Fz_tf

forces = aero_forces(Q_out)
forces[1][-1] -= np.sum(forces[1][:-1])

fig = plt.figure()
plt.title('Mode %d Aerodynamics' % mode)
plt.scatter(forces[0], forces[1]/np.abs(forces[0][0]-forces[0][-1]))
# nl_forces = np.array([tsstruct0.steady_applied_forces[6*i+2] for i in range(data.structure.num_node-1)])
# plt.scatter(forces[0], tsstruct0.steady_applied_forces[:,2]/np.abs(forces[0][0]-forces[0][-1]))
plt.ylabel('Aerodynamic Force per unit span, $F_z$ [N/m]')
plt.xlabel('Spanwise coordinate, y [m]')
plt.savefig('./figures/Mode%d_aero.pdf' % mode)
plt.show()


# Gamma
gamma_vec = Q_out[-1, 6*(data.structure.num_node-1)+9:6*(data.structure.num_node-1)+9+uvlm.K]
gamma = []
worked_panels = 0
for i_surf in range(data.aero.timestep_info[-1].n_surf):
    dimensions_gamma = data.aero.aero_dimensions[i_surf]
    panels_in_surface = data.aero.timestep_info[-1].gamma[i_surf].size
    gamma.append(gamma_vec[worked_panels:worked_panels+panels_in_surface].reshape(
        dimensions_gamma, order='C'))
    worked_panels += panels_in_surface

X,Y = np.meshgrid(range(int(N/2)), range(M))
fig = plt.figure()
plt.contour(X, Y, gamma[0])
plt.show()
fig = plt.figure()
plt.contour(X, Y, gamma[1])
plt.show()

# Step Results
fig, ax = plt.subplots(nrows=3, constrained_layout=True, sharex=True)
fig.suptitle('Mode %d' %mode)
ax[0].plot(tout, np.ones_like(tout), color='b', ls='-.', label='Input')
ax[0].plot(tout, Q_out[:, -3]/beam.U[-9+mode], color='k', label='Aerodynamic Force')
ax[0].legend()
ax[1].plot(tout, Q_out[:, -2]*beam.U[-9+mode], color='k')
ax[2].plot(tout, Q_out[:, -1]*beam.U[-9+mode], color='k', label='Linear SHARPy')

ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
ax[2].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
ax[0].set_ylabel(label_list_force[mode])
ax[1].set_ylabel(label_list_integr[mode])
ax[2].set_ylabel(label_list_veloc[mode])
ax[2].set_xlabel('Time, t [s]')
fig.savefig('./figures/Mode%d_output.pdf' % mode)
fig.show()
print('End')
