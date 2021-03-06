import numpy as np
import os
import sys
sys.path.append('/home/ng213/code/sharpy/')
import matplotlib.pyplot as plt
import sharpy.linear.src.libsparse as libsp
import scipy.linalg as sclalg
import scipy as sc
import sharpy.utils.h5utils as h5
import sharpy.linear.src.linuvlm as linuvlm
import sharpy.linear.src.lin_aeroelastic as linaeroelastic
import sharpy.rom.krylovreducedordermodel as krylovrom
import sharpy.linear.src.libss as libss
import sharpy.rom.frequencyresponseplot as freq_resp

import cases.templates.flying_wings as wings
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
ws.update_mass_stiffness(sigma=1, lumped_mass_node=0, lumped_mass_=108*0,
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

beam_settings = {'modal_projection': False,
                 'inout_coords': 'nodes',
                 'discrete_time': True,
                 'newmark_damp': 0.15,
                 'discr_method': 'newmark',
                 'dt': dt,
                 'proj_modes': 'undamped',
                 'use_euler': use_euler,
                 'num_modes': 13,
                 'print_info': True,
                 'gravity': False}
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

# beam.Cstr[-4:, :] = 0  # Erase quaternion equations - using euler parametrisation
beam.Cstr += C_rigid_effects
beam.linearise_gravity_forces(tsstruct0)
# Get rid of all dofs except the rolling motion

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



beam.assemble()

# Remove integrals from state
flex_nodes = data.structure.num_dof.value
psi = np.zeros((beam.SSdisc.states, beam.SSdisc.states-rigid_dof))
psi[:flex_nodes, :flex_nodes] = np.eye(flex_nodes)
psi[-(flex_nodes+rigid_dof):, -(flex_dof+rigid_dof):] = np.eye(flex_dof + rigid_dof)

beam.SSdisc.A = psi.T.dot(beam.SSdisc.A.dot(psi))
beam.SSdisc.B = psi.T.dot(beam.SSdisc.B)
beam.SSdisc.C = beam.SSdisc.C.dot(psi)

beam_eigs = np.log(np.linalg.eig(beam.SSdisc.A)[0])/beam.SSdisc.dt


# Add modal input/output to UVLM
out_mat = beam.U.T
out_mat[-1, :] = 0
mode_integr = beam.U.copy()
mode_integr[-2,3] = 0
in_mat = sclalg.block_diag(beam.U*0, mode_integr)
uvlm.SS.addGain(in_mat, where='in')
uvlm.SS.addGain(out_mat, where='out')

# Couple UVLM and BEAM
Tsa = np.eye(beam.SSdisc.inputs, uvlm.SS.outputs)
Tas = np.eye(uvlm.SS.inputs, beam.SSdisc.outputs)
# Remove the couple between orientation and aerodynamics
# Tas[-1, -1] = 0
# Tas[3, 3] = 0
# Tsa[-1, -1] = 0
ss_coupled = libss.couple(uvlm.SS, beam.SSdisc, Tas, Tsa)

eigs_dt, fd_modes = np.linalg.eig(ss_coupled.A)
eigs_ct = np.log(eigs_dt) / ws.dt

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
plt.xlim([-2, 0.05])
plt.ylim([-0.5, 0.5])
plt.show()
# plt.savefig('/home/ng213/sharpy_cases/08_SimpleGlider/rootlocus_test.pdf')

print('End')
