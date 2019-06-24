# HALE aircraft stability
# Horten wing Analysis
# Norberto Goizueta March 2019

import numpy as np
import matplotlib.pyplot as plt
import sharpy.linear.src.libsparse as libsp
import scipy.linalg as sclalg
import sharpy.linear.src.lin_aeroelastic as linaeroelastic
import sharpy.linear.src.libss as libss

import sharpy.sharpy_main

# Horten class that I should move at some point

# Problem Set up
u_inf = 10.
alpha_deg = 0.
rho = 1.225
num_modes = 10
num_flex_modes = num_modes - 9
thrust = 3.2709231144166235
alpha_deg = 3.947691716349666
cs_deflection = -3.1179984586324663

# Linear UVLM settings
integration_order = 2
remove_predictor = False
use_sparse = True

data = sharpy.sharpy_main.main(['','hale.solver.txt'])

# Linearise Reference Solution
tsaero0 = data.aero.timestep_info[-1]
tsstrct0 = data.aero.timestep_info[-1]
dt = data.settings['Modal']['dt'].value
tsaero0.rho = rho
flex_dof = data.structure.num_dof.value
rigid_dof = 9

# Scaling factors for the UVLM
scaling_factors = {'length': 1, #0.5*ws.c_root,
                   'speed': 1, #u_inf,
                   'density': 1}  #rho}

# Linear Aeroelastic System
lin_settings = {'dt': dt,
                'integr_order': integration_order,
                'density': rho,
                'remove_predictor': remove_predictor,
                'use_sparse': use_sparse,
                'ScalingDict': scaling_factors,
                'rigid_body_motion': True,
                'use_euler': True}

aeroelastic_system = linaeroelastic.LinAeroEla(data,
                                               custom_settings_linear=lin_settings)
beam = aeroelastic_system.lingebm_str
uvlm = aeroelastic_system.linuvlm

# Trim matrices to reduce state dimensionality by 1 (Euler instead of quaternion)
beam.Mstr = beam.Mstr[:-1, :-1]
beam.Cstr = beam.Cstr[:-1, :-1]
beam.Kstr = beam.Kstr[:-1, :-1]

# Get BEAM <-> UVLM gain matrices
aeroelastic_system.get_gebm2uvlm_gains()
# Non-zero aerodynamic forces at the linearisation point addition to the beam matrices
beam.Kstr[:flex_dof, :flex_dof] += aeroelastic_system.Kss

C_rigid_effects = np.block([[np.zeros((flex_dof, flex_dof)), aeroelastic_system.Csr],
                       [aeroelastic_system.Crs, aeroelastic_system.Crr]])
beam.Cstr[-rigid_dof:, :] = 0
beam.Cstr += C_rigid_effects

beam.update_modal()

# UVLM lattice to beam nodes gain matrices
zero_matrix = np.zeros((3*uvlm.Kzeta, beam.num_dof))
gain_struct_2_aero = np.block([[aeroelastic_system.Kdisp, zero_matrix],
                     [aeroelastic_system.Kvel_disp, aeroelastic_system.Kvel_vel]])

gain_aero_2_struct = aeroelastic_system.Kforces

# Linear structural solver solution settings
beam.dlti = True
beam.newmark_damp = 5e-3
beam.modal = True
beam.proj_modes = 'undamped'
beam.Nmodes = num_modes
beam.discr_method = 'newmark'

# Reorder modes
phiRR = np.eye(9)
phiSR = beam.U[:-9, :9]
phiRS = beam.U[-9:, 9:]
phiSS = beam.U[:-9, 9:]

beam.freq_natural = np.concatenate((beam.freq_natural[9:9+num_flex_modes], beam.freq_natural[:9]))
phi = np.block([[phiSS[:, :num_flex_modes], phiSR], [phiRS[:, :num_flex_modes], phiRR]])
beam.U = phi

for i in range(len(beam.freq_natural)):
    diag_factor = beam.U[:, i].T.dot(beam.Mstr.dot(beam.U[:, i]))
    beam.U[:, i] = 1 / np.sqrt(diag_factor) * beam.U[:, i]

# Update structural model
beam.assemble()

# Group modes into symmetric and anti-symmetric modes
modes_sym = np.zeros_like(beam.U)
total_modes = len(beam.freq_natural)
# rbm_modes = beam.U[:10]
free_nodes = data.structure.num_node - 1
ind_z = [6*i + 2 for i in range(free_nodes)]
ind_mx = [6*i + 3 for i in range(free_nodes)]
#
# for i in range(total_modes):
#     plt.title('Mode %d, Frequency %.2f rad/s' % (i+1, beam.freq_natural[i]))
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
ind_w1 = [6*i + 2 for i in range(free_nodes // 2)]  # Wing 1 nodes are in the first half rows
ind_w2 = [6*i + 2 for i in range(free_nodes // 2, free_nodes)]  # Wing 2 nodes are in the second half rows
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
#
#     plt.plot(beam.U[ind_w1, i])
#     plt.plot(beam.U[ind_w2, i])
#     plt.scatter(beam.U[-10:, i], ls='--')
#     plt.show()

# phi = np.block([[np.eye(phiSS.shape[0], phiSS.shape[1]), np.zeros((total_modes, 10))],
#                 [-beam.U[-10:, :total_modes], np.eye(10)]])
# phi2 = beam.U.dot(phi)

# # Assemble UVLM
uvlm.assemble_ss()

# # Scale UVLM
# t_uvlm_scale0 = time.time()
# print('Scaling UVLM...')
# uvlm.nondimss()
# length_ref = uvlm.ScalingFacts['length']
# speed_ref = uvlm.ScalingFacts['speed']
# force_ref = uvlm.ScalingFacts['force']
# time_ref = uvlm.ScalingFacts['time']
# t_uvlm_scale = time.time() - t_uvlm_scale0
# print('...UVLM Scaling complete in %.2f s' % t_uvlm_scale)

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
# wv = 2*u_inf/ws.c_root*kv
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
# q_ref = 0.5*rho*u_ref**2
# t_ref = length_ref/u_ref
# force_c = q_ref * length_ref ** 2
#
# Update structural model with scaled parameters
beam.dt = aeroelastic_system.linuvlm.SS.dt

# beam.freq_natural = natural_frequency_ref * t_ref ** 2
# beam.Cstr = beam.Cstr * t_ref
beam.inout_coords = 'modes'
beam.assemble()
beam_ss = beam.SSdisc
#
# Update BEAM -> UVLM gains due to scaling
# Tas = np.eye(2*num_modes) / length_ref
Tas = np.eye(2*beam.Nmodes)

# Update UVLM -> BEAM gains with scaling
# Tsa = np.diag((force_c*t_ref**2) * np.ones(beam.Nmodes))
Tsa = np.diag(1. * np.ones(beam.Nmodes))
# Tsa = np.ones(beam.Nmodes)
#
#     # Assemble new aeroelastic systems
ss_aeroelastic = libss.couple(ss01=uvlm.SS, ss02=beam_ss, K12=Tas, K21=Tsa)
#     ss_aeroelastic = libss.couple(ss01=rom.ssrom, ss02=beam_ss, K12=Tas, K21=Tsa)
#
dt_new = dt # ws.c_root / M / u_ref
# assert np.abs(dt_new - t_ref * aeroelastic_system.linuvlm.SS.dt) < 1e-14, 'dimensional time-scaling not correct!'
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

# Remove zero eigenvalues (9 integrals + 3 euler)
non_zero_evals = np.abs(eigs_cont)!=0
eigs_cont = eigs_cont[non_zero_evals]
eigvecs = eigvecs[:, non_zero_evals]
# eigs_cont = eigs_cont[eigs_cont.imag >= 0]  # Remove lower plane symmetry

for i in range(15):
    print('mu = %.4f + %.4fj' % (eigs_cont[i].real, eigs_cont[i].imag))

struct_modes = eigvecs[uvlm.SS.states:, :num_modes]

nodal_coords = sclalg.block_diag(beam.U, beam.U).dot(struct_modes)

# Y_freq = ss_aeroelastic.freqresp(wv)
print('Routine Completed')
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


def plot_aeroelastic_mode(mode):
    evec = eigvecs[uvlm.SS.states:-36, mode]
    a = beam.U.dot(evec)

    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(a[ind_w1].real, color='k', ls='-')
    ax[0].plot(a[ind_w1].imag, color='k', ls='--')
    ax[0].plot(a[ind_w2].real, color='b', ls='-')
    ax[0].plot(a[ind_w2].imag, color='b', ls='--')
    ax[0].set_title('Mode Frequency %.2f rad/s, %.2f Hz' % (eigs_cont[mode].imag, eigs_cont[mode].imag/2/np.pi))
    ax[1].plot(a[-6:-3].real, color='k', ls='-')
    ax[1].plot(a[-6:-3].imag, color='k', ls='--')
    ax[1].plot(a[-3:].real, color='b', ls='-')
    ax[1].plot(a[-3:].imag, color='b', ls='--')
    plt.show()