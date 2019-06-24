import numpy as np
import os
import sys
sys.path.append('/home/ng213/code/sharpy/')
import matplotlib.pyplot as plt
import sharpy.linear.src.libsparse as libsp
import scipy.linalg as sclalg
import sharpy.linear.src.lin_aeroelastic as linaeroelastic
import sharpy.linear.src.libss as libss
import scipy as sc

sys.path.append('/home/ng213/sharpy_cases/08_SimpleGlider')
import glider
import sharpy.sharpy_main

# Problem Set up
u_inf = 28.
alpha_deg = 11.35
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
#
# M = 8
# N = 16
# M_star_fact = 20

# Linear UVLM settings
integration_order = 2
remove_predictor = False
use_sparse = True


# Case Admin - Create results folders
case_name = 'glider'
case_nlin_info = 'M%dN%dMs%d_nmodes%d' %(M, N, M_star_fact, num_modes)
fig_folder = './figures/'
os.system('mkdir -p %s' % fig_folder)


# SHARPy nonlinear reference solution
ws = glider.Glider(M=M,
                   N=N,
                   Mstarfact=M_star_fact,
                   u_inf=u_inf,
                   alpha_deg=alpha_deg,
                   rho=rho,
                   physical_time=100,
                   beta_deg=0,
                   case_route='./cases/',
                   case_name=case_name)

ws.sigma = 1

ws.clean_test_files()
ws.update_mass_stiffness(sigma=1, lumped_mass_node=0, lumped_mass_=100*0,
                             lumped_mass_inertia_= 0*np.diag([1,1,1]))
ws.update_fem_prop()
ws.generate_fem_file()
ws.update_aero_properties()
ws.generate_aero_file()
ws.set_default_config_dict()

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
ws.config.write()

data = sharpy.sharpy_main.main(['',ws.case_route+'/'+ws.case_name+'.solver.txt'])

# Linearise - reduce UVLM and project onto modes
scaling_factors = {'length': 1,  #0.5 * ws.c_ref,
                   'speed': 1,  #u_inf,
                   'density': 1} #rho}
dt = ws.dt

beam_settings = {'modal_projection': True,
                 'inout_coords': 'modes',
                 'discrete_time': True,
                 'newmark_damp': 1.5e-3,
                 'discr_method': 'newmark',
                 'dt': dt,
                 'proj_modes': 'undamped',
                 'use_euler': use_euler,
                 'num_modes': 13,
                 'print_info': 'on',
                 'gravity':'off'}
lin_settings = {'dt': dt,
                'integr_order': integration_order,
                'density': rho,
                'remove_predictor': remove_predictor,
                'use_sparse': use_sparse,
                'ScalingDict': scaling_factors,
                'rigid_body_motion': True,
                'use_euler': use_euler,
                'beam_settings': beam_settings}

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
# if use_euler:
#     beam.Mstr = beam.Mstr[:-1, :-1]
#     beam.Cstr = beam.Cstr[:-1, :-1]
#     beam.Kstr = beam.Kstr[:-1, :-1]

# structure to UVLM gains
aeroelastic_system.get_gebm2uvlm_gains()

beam.update_modal()

print('Iyy = %.4f kg m2' %(beam.Mstr[-5, -5]))
beam.Kstr[:flex_dof, :flex_dof] += aeroelastic_system.Kss
beam.Kstr[flex_dof:, :flex_dof] += aeroelastic_system.Krs

C_rigid_effects = np.block([[np.zeros((flex_dof, flex_dof)), aeroelastic_system.Csr],
                            [aeroelastic_system.Crs, aeroelastic_system.Crr]])

beam.Cstr += C_rigid_effects
beam.linearise_gravity_forces(tsstruct0)

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

# Add coupling matrices
uvlm.SS.addGain(Kas, where='in')
uvlm.SS.addGain(Ksa, where='out')

# Keep longitudinal dynamic modes only - Vx Vz, omega_y and theta
phi = np.zeros((beam.U.shape[0], 4))
# phi = np.zeros((beam.U.shape[0], 5))
# phi = np.zeros((beam.U.shape[0], 9))
phi[-9, 0] = 1
phi[-7, 1] = 1
phi[-5, 2] = 1
phi[-2, 3] = 1
# Lateral Modes
# phi[-8, 0] = 1
# phi[-6, 1] = 1
# phi[-4, 2] = 1
# phi[-3, 3] = 1
# phi[-1, 4] = 1

# All
# phi[-9:,-9:] = np.eye(9)

# beam.U = np.hstack((phi, beam.U[:, 10:15]))
beam.U = phi
beam.freq_natural = np.zeros((4,))
# beam.freq_natural = np.concatenate((np.zeros((4,)), beam.freq_natural[10:15]))
beam.Ccut = beam.U.T.dot(beam.Cstr.dot(beam.U))
# beam.freq_natural = np.zeros(5)
# beam.freq_natural = np.zeros(9)

for i in range(len(beam.freq_natural)):
    diag_factor = beam.U[:, i].T.dot(beam.Mstr.dot(beam.U[:, i]))
    beam.U[:, i] = 1 / np.sqrt(diag_factor) * beam.U[:, i]

# beam_ct = beam
beam.dlti = True
beam.newmark_damp = 5e-3
beam.modal = True
beam.proj_modes = 'undamped'
beam.Nmodes = len(beam.freq_natural)
beam.discr_method = 'newmark'
beam.inout_coords = 'modes'

# beam_ct.dlti = False
# beam_ct.modal = True
# beam_ct.proj_modes = 'undamped'
# beam_ct.inout_coords = 'modes'
# beam_ct.Nmodes = len(beam.freq_natural)
# beam_ct.assemble()

beam.assemble()
# Remove integral of RBMs
psi = np.zeros((beam.SSdisc.states, 4))
psi[-4:, -4:] = np.eye(4)

beam.SSdisc.A = psi.T.dot(beam.SSdisc.A.dot(psi))
beam.SSdisc.B = psi.T.dot(beam.SSdisc.B)
beam.SSdisc.C = beam.SSdisc.C.dot(psi)

beam_eigs = np.log(np.linalg.eig(beam.SSdisc.A)[0])/beam.SSdisc.dt
# Add modal input/output to UVLM
out_mat = beam.U.T
out_mat[-1, :] = 0
mode_integr = beam.U.copy()
mode_integr[-2,3] = 0

# Change some signs
mode_integr[:, 0] *= +1
mode_integr[:, 2] *= -1
out_mat[0, :] *= 1
out_mat[2, :] *= 1
in_mat = sclalg.block_diag(beam.U*0, mode_integr)
uvlm.SS.addGain(in_mat, where='in')
uvlm.SS.addGain(out_mat, where='out')

# Couple UVLM and BEAM
Tsa = np.eye(beam.SSdisc.inputs, uvlm.SS.outputs)
Tas = np.eye(uvlm.SS.inputs, beam.SSdisc.outputs)
# Tas[0, 0] = -1
# Tas[2, 2] = -1
# Tas[4, 4] = -1
# Tas[6, 6] = -1
# Tsa = Tas[:4, :4]
# Remove the couple between orientation and aerodynamics
Tas[-1, -1] = 0
Tas[3, 3] = 0
Tsa[-1, -1] = 0
ss_coupled = libss.couple(uvlm.SS, beam.SSdisc, Tas, Tsa)

eigs_dt, fd_modes = np.linalg.eig(ss_coupled.A)
eigs_ct = np.log(eigs_dt) / ws.dt

# Sort eigvals in real mag
order = np.argsort(eigs_ct.real)[::-1]
eigs_ct = eigs_ct[order]
fd_modes = fd_modes[:, order]

plt.figure()
for eig in range(len(eigs_ct)):
    if np.argmax(np.abs(fd_modes[:, eig]), axis=0) < uvlm.SS.states:
        C = 'b'
        S = 2
        m = 's'
    else:
        C = 'r'
        S = 10
        m = '^'
        print("%d: %.2e + %.2ej" %(eig, eigs_ct[eig].real, eigs_ct[eig].imag))

    plt.scatter(eigs_ct[eig].real, eigs_ct[eig].imag, color=C, s=S, marker=m)
plt.grid()
plt.xlim([-2, 0.05])
plt.ylim([-0.5, 0.5])
# plt.show()
# plt.savefig('/home/ng213/sharpy_cases/08_SimpleGlider/rootlocus_test.pdf')

# Simulation
dlti = sc.signal.dlti(ss_coupled.A,
                      ss_coupled.B,
                      ss_coupled.C,
                      ss_coupled.D,
                      dt=ss_coupled.dt)

# Initial State
X0 = np.zeros(ss_coupled.states)

# Time
N = 100
T = N*ss_coupled.dt
t_dom = np.linspace(0, T, N)
U = np.zeros((N, ss_coupled.inputs))
U[:, -4] = 1

out = sc.signal.dlsim(dlti, U, t_dom)

# Get aerodynamic circulation
x_a = out[2][:, :uvlm.SS.states]

# This is very very dirty but its temporary
import sharpy.solvers.linearassembler as linearassembler
import sharpy.linear.assembler.linearuvlm as linearuvlm
assbly_settings = {'flow': 'LinearUVLM',
                   'linearisation_step': -1,
                   'LinearUVLM': lin_settings}
dummy_lin = linearassembler.LinearAssembler()
dummy_lin.initialise(data, assbly_settings)
out_dummy = dummy_lin.unpack_ss_vectors(None, x_a, None, tsaero0)




print('End')
