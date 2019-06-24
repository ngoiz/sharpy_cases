# Horten wing Analysis
# Norberto Goizueta March 2019

import numpy as np
import os
import sharpy.linear.src.libsparse as libsp
import scipy.linalg as sclalg
import time
import sharpy.linear.src.lin_aeroelastic as linaeroelastic
import sharpy.linear.src.libss as libss

import sharpy.sharpy_main

# Horten class that I should move at some point
import horten_wing

# Problem Set up
u_inf = 20.
alpha_deg = 0.
rho = 1.02
num_modes = 40
thrust = 3.2709231144166235
alpha_deg = 3.947691716349666
cs_deflection = -3.1179984586324663

# Lattice Discretisation
M = 6
N = 11
M_star_fact = 10
#
# M = 16  # 4
# N = 60
# M_star_fact = 18

# Linear UVLM settings
integration_order = 2
remove_predictor = False
use_sparse = True

# ROM Properties
algorithm = 'mimo_rational_arnoldi'
r = 2
frequency_continuous_k = np.array([0.])

# Case Admin - Create results folders
case_name = 'horten'
case_nlin_info = 'M%dN%dMs%d_nmodes%d' %(M, N, M_star_fact, num_modes)
case_rom_info = 'rom_MIMORA_r%d_sig%04d_%04dj' % (r, frequency_continuous_k[0].real*1000,
                                                  frequency_continuous_k[0].imag*1000)
fig_folder = './figures/'
os.system('mkdir -p %s' % fig_folder)

# SHARPy reference solution
ws = horten_wing.HortenWing(M=M,
                            N=N,
                            Mstarfactor=M_star_fact,
                            u_inf=u_inf,
                            thrust=thrust,
                            alpha_deg=alpha_deg,
                            cs_deflection_deg=cs_deflection,
                            case_name_format=1,
                            physical_time=3,
                            case_remarks='',
                            case_route='cases/')
ws.horseshoe = False
ws.gust_intensity = -0.01
ws.n_tstep = 1

ws.clean_test_files()
ws.update_mass_stiffness(sigma=1)
ws.update_aero_properties()
ws.update_fem_prop()
ws.generate_aero_file()
ws.generate_fem_file()
ws.set_default_config_dict()

ws.config['SHARPy']['flow'] = ['BeamLoader',
                               'AerogridLoader',
                               # 'StaticTrim',
                               'StaticCoupled',
                               'Modal',
                               # 'AeroForcesCalculator',
                               'AerogridPlot',
                               'BeamPlot',
                               # 'DynamicCoupled',
                               # 'AeroForcesCalculator',
                               # 'Modal',
                               # 'SaveData']
                               ]

ws.config['SHARPy']['write_screen'] = True
# ws.config['StaticTrim']['fz_tolerance'] = 1e-4,
# ws.config['StaticTrim']['fx_tolerance'] = 1e-4
# ws.config['StaticTrim']['m_tolerance'] = 1e-4
ws.config['DynamicCoupled']['n_time_steps'] = 1
ws.config['Modal'] = {'print_info': True,
                      'use_undamped_modes': True,
                      'NumLambda': 40,
                      'rigid_body_modes': False,
                      'write_modes_vtk': 'off',
                      'print_matrices': 'off',
                      'write_data': 'off',
                      'continuous_eigenvalues': 'off',
                      'dt': ws.dt,
                      'plot_eigenvalues': False}
ws.config.write()

data = sharpy.sharpy_main.main(['',ws.case_route+'/'+ws.case_name+'.solver.txt'])

# Linearise Reference Solution
tsaero0 = data.aero.timestep_info[-1]
tsstrct0 = data.aero.timestep_info[-1]
dt = ws.dt
tsaero0.rho = rho
flex_dof = data.structure.num_dof.value
rigid_dof = 0

# Scaling factors for the UVLM
scaling_factors = {'length': 0.5*ws.c_root,
                   'speed': u_inf,
                   'density': rho}

# Linear Aeroelastic System
lin_settings = {'dt': dt,
                'integr_order': integration_order,
                'density': rho,
                'remove_predictor': remove_predictor,
                'use_sparse': use_sparse,
                'ScalingDict': scaling_factors,
                'rigid_body_motion': False}

aeroelastic_system = linaeroelastic.LinAeroEla(data,
                                               custom_settings_linear=lin_settings)
beam = aeroelastic_system.lingebm_str
uvlm = aeroelastic_system.linuvlm

# Get BEAM <-> UVLM gain matrices
aeroelastic_system.get_gebm2uvlm_gains()

# Non-zero aerodynamic forces at the linearisation point addition to the beam matrices
# beam.Kstr[:flex_dof, :flex_dof] += aeroelastic_system.Kss
# C_non_zero = np.block([[np.zeros((flex_dof, flex_dof)), aeroelastic_system.Csr],
#                        [aeroelastic_system.Crs, aeroelastic_system.Crr]])
# beam.Cstr += C_non_zero

# UVLM lattice to beam nodes gain matrices
zero_matrix = np.zeros((3*uvlm.Kzeta, beam.num_dof))
gain_struct_2_aero = np.block([[aeroelastic_system.Kdisp[:,:-10], zero_matrix],
                     [aeroelastic_system.Kvel_disp[:, :-10], aeroelastic_system.Kvel_vel[:, :-10]]])

gain_aero_2_struct = aeroelastic_system.Kforces[:-10, :]

# Linear structural solver solution settings
beam.dlti = True
beam.newmark_damp = 5e-3
beam.modal = True
beam.proj_modes = 'undamped'
beam.Nmodes = num_modes
beam.discr_method = 'newmark'

# # Update modes
# eigs, eigvecs = sclalg.eig(sclalg.solve(beam.Mstr, beam.Kstr))
# omega = np.sqrt(eigs)
# order = np.argsort(eigs)[:beam.Nmodes]
# beam.freq_natural = omega[order]
# beam.U = eigvecs[:, order]

# Orthogonalise eigenvectos
# beam.U = sclalg.qr(beam.U)[0][:, :len(beam.freq_natural)]

# Update structural model
beam.assemble()

# Group modes into symmetric and anti-symmetric modes
modes_sym = np.zeros_like(beam.U)
total_modes = len(beam.freq_natural)
# rbm_modes = beam.U[:10]
free_nodes = data.structure.num_node - 1
ind_z = [6*i + 2 for i in range(free_nodes)]
ind_mx = [6*i + 3 for i in range(free_nodes)]

# for i in range(10,15):
#     plt.title('Mode %d, Frequency %.2f rad/s' % (i-9, beam.freq_natural[i]))
#     plt.plot(beam.U[ind_z[:free_nodes//2], i], color='k')
#     plt.plot(beam.U[ind_z[free_nodes//2:], i], color='k', ls='--')
#     plt.show()
# Modes come paired for each beam. Therefore they need to be grouped
# Mode 2i corresponds to beam 0 and Mode 2i+1 corresponds to beam 1
# After the transformation:
#   Mode 2*i will be the symmetric combination of the two
#   Mode 2*i+1 will be the antisymmetric combination of the two
# for i in range((total_modes-10)//2):
#     je = 2*i + 10
#     jo = 2*i+1 + 10
#     modes_sym[:, je] = 1/np.sqrt(2)*(beam.U[:, je] + beam.U[:, jo])
#     modes_sym[:, jo] = 1/np.sqrt(2)*(beam.U[:, je] - beam.U[:, jo])

# beam.U = modes_sym
# # Remove anti-symmetric modes based on the z index
# # 1) Obtain z index for each beam
# ind_w1 = [6*i + 2 for i in range(free_nodes // 2)]  # Wing 1 nodes are in the first half rows
# ind_w2 = [6*i + 2 for i in range(free_nodes // 2, free_nodes)]  # Wing 2 nodes are in the second half rows
# #for i in range(total_modes):
# #    plt.plot(beam.U[ind_w1, i])
# #    plt.plot(beam.U[ind_w2, i])
# #    plt.show()
#
#
#
# # Find symmetric modes and discard antisymmetric ones
# sym_mode_index = []
# for i in range(total_modes//2):
#     found_symmetric = False
#
#     for j in range(2):
#         ind = 2*i + j
#
#         # Maximum z displacement for wings 1 and 2
#         ind_max_w1 = np.argmax(np.abs(modes_sym[ind_w1, ind]))
#         ind_max_w2 = np.argmax(np.abs(modes_sym[ind_w2, ind]))
#         z_max_w1 = modes_sym[ind_w1, ind][ind_max_w1]
#         z_max_w2 = modes_sym[ind_w2, ind][ind_max_w2]
#
#         z_max_diff = np.abs(z_max_w1 - z_max_w2)
#         if z_max_diff < np.abs(z_max_w1 + z_max_w2):
#             sym_mode_index.append(ind)
#             if found_symmetric == True:
#                 raise NameError('Symmetric Mode previously found')
#             found_symmetric = True
#
# beam.U = modes_sym[:, sym_mode_index]
# beam.freq_natural = beam.freq_natural[sym_mode_index]
natural_frequency_ref = beam.freq_natural.copy()  # Save reference natural frequencies

# for i in range(len(natural_frequency_ref)):
#     plt.plot(beam.U[ind_w1, i])
#     plt.plot(beam.U[ind_w2, i])
#     plt.show()

# Change mode projection basis so that it doesn't change
# phiRR = np.eye(10)
# phiSR = beam.U[:-10, :10]
# phiRS = beam.U[-10:, 10:]
# phiSS = beam.U[:-10, 10:]
#
# phi = np.block([[phiSS, phiSR], [phiRS, phiRR]])
# phi = np.block([[np.eye(phiSS.shape[0], phiSS.shape[1]), np.zeros((total_modes, 10))],
#                 [-beam.U[-10:, :total_modes], np.eye(10)]])
# phi2 = beam.U.dot(phi)

# # Assemble UVLM
uvlm.assemble_ss()

# Scale UVLM
t_uvlm_scale0 = time.time()
print('Scaling UVLM...')
uvlm.nondimss()
length_ref = uvlm.ScalingFacts['length']
speed_ref = uvlm.ScalingFacts['speed']
force_ref = uvlm.ScalingFacts['force']
time_ref = uvlm.ScalingFacts['time']
t_uvlm_scale = time.time() - t_uvlm_scale0
print('...UVLM Scaling complete in %.2f s' % t_uvlm_scale)

# UVLM remove gust input - gusts not required in analysis
uvlm.SS.B = libsp.csc_matrix(aeroelastic_system.linuvlm.SS.B[:, :6*aeroelastic_system.linuvlm.Kzeta])
uvlm.SS.D = libsp.csc_matrix(aeroelastic_system.linuvlm.SS.D[:, :6*aeroelastic_system.linuvlm.Kzeta])
#
# UVLM projection onto modes
print('Projecting UVLM onto modes')
gain_input = libsp.dot(gain_struct_2_aero, sclalg.block_diag(beam.U[:, :beam.Nmodes], beam.U[:, :beam.Nmodes]))
gain_output = libsp.dot(beam.U[:, :beam.Nmodes].T, gain_aero_2_struct)
uvlm.SS.addGain(gain_input, where='in')
uvlm.SS.addGain(gain_output, where='out')
print('...complete')

# # Krylov MOR
# rom = krylovrom.KrylovReducedOrderModel()
# rom.initialise(data, aeroelastic_system.linuvlm.SS)
# frequency_continuous_k = np.array([0.1j])
# frequency_continuous_w = 2 * u_inf * frequency_continuous_k / ws.c_root
# interpolation_point = np.exp(frequency_continuous_w*aeroelastic_system.linuvlm.SS.dt)
#
# rom.run(algorithm, r, interpolation_point)
#
# Frequency analysis
kn = np.pi / aeroelastic_system.linuvlm.SS.dt
kv = np.logspace(-3, np.log10(kn), 50)
kv = np.logspace(-3, np.log10(1), 50)
wv = 2*u_inf/ws.c_root*kv
#
# frequency_response_fom = aeroelastic_system.linuvlm.SS.freqresp(wv)
# frequency_response_rom = rom.ssrom.freqresp(wv)
# rom_max_magnitude = np.max(np.abs(frequency_response_rom))
# error_frequency_response = np.max(np.abs(frequency_response_rom - frequency_response_fom))
#
# fplot = freq_resp.FrequencyResponseComparison()
# fsettings = {'frequency_type': 'k',
#              'plot_type': 'real_and_imaginary'}
# fplot.initialise(data, aeroelastic_system.linuvlm.SS, rom, fsettings)
# i = 0
# o = 0
# in_range = 2*i
# in_rangef = in_range + 2
# out_range = 2*0
# out_rangef = out_range+2
# fplot.plot_frequency_response(kv, frequency_response_fom[out_range:out_rangef, in_range:in_rangef, :],
#                               frequency_response_rom[out_range:out_rangef, in_range:in_rangef, :], interpolation_point)
# fplot.save_figure(fig_folder + '/ROM_Freq_response_' + case_name + case_nlin_info + case_rom_info + '.pdf')
#
# # Modal UVLM eigenvalues
# eigs_UVLM = sclalg.eigvals(rom.ssrom.A)
# eigs_UVLM_cont = np.log(eigs_UVLM)/rom.ssrom.dt
#
# # Couple with structural model
# u_inf_vec = np.linspace(30, 230, 201)
# num_uinf = len(u_inf_vec)
#
# # Initiliase variables to store eigenvalues for later plot
# real_part_plot = []
# imag_part_plot = []
# uinf_part_plot = []
#
# for i in range(num_uinf):
#
#     # Scaling of properties
u_ref = u_inf
q_ref = 0.5*rho*u_ref**2
t_ref = length_ref/u_ref
force_c = q_ref * length_ref ** 2
#
# Update structural model
beam.dt = aeroelastic_system.linuvlm.SS.dt
beam.freq_natural = natural_frequency_ref * t_ref
beam.inout_coords = 'modes'
beam.assemble()
beam_ss = beam.SSdisc
#
# Update BEAM -> UVLM gains due to scaling
Tas = np.eye(2*num_modes) / length_ref

# Update UVLM -> BEAM gains with scaling
Tsa = np.diag((force_c*t_ref**2) * np.ones(beam.Nmodes))
#
#     # Assemble new aeroelastic systems
ss_aeroelastic = libss.couple(ss01=uvlm.SS, ss02=beam_ss, K12=Tas, K21=Tsa)
#     ss_aeroelastic = libss.couple(ss01=rom.ssrom, ss02=beam_ss, K12=Tas, K21=Tsa)
#
dt_new = ws.c_root / M / u_ref
assert np.abs(dt_new - t_ref * aeroelastic_system.linuvlm.SS.dt) < 1e-14, 'dimensional time-scaling not correct!'
# #
#     # Asymptotic stability of the system
eigs, eigvecs = sclalg.eig(ss_aeroelastic.A)
eigs_mag = np.abs(eigs)
order = np.argsort(eigs_mag)[::-1]
eigs = eigs[order]
eigvecs = eigvecs[:, order]
eigs_mag = eigs_mag[order]
# #     frequency_dt_evals = 0.5*np.angle(eigs)/np.pi/dt_new
# #
# #     # Nunst = np.sum(eigs_mag>1.)
# #
eigs_cont = np.log(eigs) / dt_new
eigs_cont = eigs_cont[eigs_cont.imag >= 0]  # Remove lower plane symmetry
#     Nunst = np.sum(eigs_cont.real > 0)
#     eigmax = np.max(eigs_mag)
#     fn = np.abs(eigs_cont)
#
#     # Beam
#     eigs_beam = sclalg.eigvals(beam.SSdisc.A)
#     order = np.argsort(eigs_beam)[::-1]
#     eigs_beam = eigs_beam[order]
#     eigmax_beam = np.max(np.abs(eigs_beam))
#
#     print('DLTI\tu: %.2f m/2\tmax.eig.: %.6f\tmax.eig.gebm: %.6f' \
#           % (u_ref, eigmax, eigmax_beam))
#     # print('\tGEBM nat. freq. (Hz):'+len(fn_gebm)*'\t%.2f' %tuple(fn_gebm))
#     print('\tN unstab.: %.3d' % (Nunst,))
#     print('\tUnstable aeroelastic natural frequency CT(rad/s):' + Nunst * '\t%.2f' %tuple(fn[:Nunst]))
#     print('\tUnstable aeroelastic natural frequency DT(Hz):' + Nunst * '\t%.2f' %tuple(frequency_dt_evals[:Nunst]))
#
#     # Store eigenvalues for plot
#     real_part_plot.append(eigs_cont.real)
#     imag_part_plot.append(eigs_cont.imag)
#     uinf_part_plot.append(np.ones_like(eigs_cont.real)*u_ref)
#
# real_part_plot = np.hstack(real_part_plot)
# imag_part_plot = np.hstack(imag_part_plot)
# uinf_part_plot = np.hstack(uinf_part_plot)
#
# ax = plt.subplot(111)
#
# ax.scatter(eigs_UVLM_cont.real, eigs_UVLM_cont.imag,
#            color='grey',
#            marker='^')
#
# dataplot = ax.scatter(real_part_plot, imag_part_plot, c=uinf_part_plot,
#                       cmap=plt.cm.winter,
#                       marker='s',
#                       s=7)
#
# cb = plt.colorbar(dataplot)
# cb.set_label('Freestream Velocity, $U_{\infty}$ [m/s]')
#
# ax.set_xlim([-10,10])
# ax.set_ylim([-5, 500])
# ax.set_xlabel('Real $\lambda$ [rad/s]')
# ax.set_ylabel('Imag $\lambda$ [rad/s]')
# ax.grid()
# plt.savefig(fig_folder + '/Root_locus_' + case_name + case_nlin_info + case_rom_info + '.eps')
# plt.show()
