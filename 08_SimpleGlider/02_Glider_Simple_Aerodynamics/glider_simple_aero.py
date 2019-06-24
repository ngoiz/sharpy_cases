import numpy as np
import os
import sys
sys.path.append('/home/ng213/code/sharpy/')
import matplotlib.pyplot as plt
import sharpy.linear.src.lin_aeroelastic as linaeroelastic
import sharpy.linear.src.libss as libss

sys.path.append('/home/ng213/sharpy_cases/08_SimpleGlider')
import glider
import sharpy.sharpy_main
import simple_aero

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
# M = 4
# N = 12
# M_star_fact = 10
# #
M = 4
N = 5
M_star_fact = 10

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
ws.update_mass_stiffness(sigma=1, lumped_mass_node=4, lumped_mass_=48*0,
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
                               'Modal']
                               # 'SaveData']
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

# beam.Kstr[:flex_dof, :flex_dof] += aeroelastic_system.Kss

Crr_grav = aeroelastic_system.linearise_gravity_forces(tsstruct0)

beam.Cstr[-4:, :] = 0  # Erase quaternion equations - using euler parametrisation
beam.Cstr[-9:, -9:] += Crr_grav + aeroelastic_system.euler_propagation_equations(tsstruct0)

# Get rid of all dofs except the rolling motion
beam.update_modal()

# Create simple aerodynamics

aero = simple_aero.SimpleAero(tsaero0, tsstruct0)
aero.assemble()


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

beam.U = phi
beam.freq_natural = np.zeros(4)
# beam.freq_natural = np.zeros(5)
# beam.freq_natural = np.zeros(9)

# for i in range(len(beam.freq_natural)):
#     diag_factor = beam.U[:, i].T.dot(beam.Mstr.dot(beam.U[:, i]))
#     beam.U[:, i] = 1 / np.sqrt(diag_factor) * beam.U[:, i]

# beam_ct = beam
beam.dlti = False
# beam.newmark_damp = 5e-3
beam.modal = True
beam.proj_modes = 'undamped'
beam.Nmodes = len(beam.freq_natural)
# beam.discr_method = 'newmark'
beam.inout_coords = 'modes'

# beam_ct.dlti = False
# beam_ct.modal = True
# beam_ct.proj_modes = 'undamped'
# beam_ct.inout_coords = 'modes'
# beam_ct.Nmodes = len(beam.freq_natural)
# beam_ct.assemble()

beam.assemble()
beam_eigs = np.linalg.eig(beam.SScont.A)[0]


# Change aero output from G frame to A frame
theta = tsstruct0.euler[1]
rot_matrix = np.zeros((3, 3))
rot_matrix[0, :] = np.array([-np.cos(theta), -np.sin(theta), 0])
rot_matrix[1, :] = np.array([np.cos(theta), -np.sin(theta), 0])
rot_matrix[2, 2] = 1

aero.addGain(rot_matrix, where='out')

# Couple UVLM and BEAM
# AERO TO BEAM
Tsa = np.zeros((beam.SScont.inputs, aero.ss.outputs))
Tsa[0:3, :] = np.eye(3)

# BEAM to UVLM
Tas = np.zeros((aero.ss.inputs, beam.SScont.outputs))
Tas[:, -4:-1] = np.eye(3)

ss_coupled = libss.couple(uvlm.SS, beam.SScont, Tas, Tsa)

eigs_ct, fd_modes = np.linalg.eig(ss_coupled.A)
# eigs_ct = np.log(eigs_dt) / ws.dt

# Sort eigvals in real mag
order = np.argsort(eigs_ct.real)[::-1]
eigs_ct = eigs_ct[order]
fd_modes = fd_modes[:, order]

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
plt.xlim([-5, 1])
plt.ylim([-7, 7])
plt.show()

print('End')
