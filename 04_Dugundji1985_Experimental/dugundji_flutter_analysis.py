import numpy as np
import os
import matplotlib.pyplot as plt
import sharpy.linear.src.libsparse as libsp
import scipy.linalg as sclalg
import sharpy.linear.src.lin_aeroelastic as linaeroelastic
import sharpy.rom.krylovreducedordermodel as krylovrom
import sharpy.linear.src.libss as libss
import sharpy.rom.frequencyresponseplot as freq_resp

import flying_wings as wings
import sharpy.sharpy_main

# Problem Set up
u_inf = 1.
alpha_deg = 0.
rho = 1.02
num_modes = 5

# Lattice Discretisation
M = 16
N = 60
M_star_fact = 18
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
r = 1
frequency_continuous_k = np.array([0.0])

# Case Admin - Create results folders
case_name = 'dugundji_0_2_90_s'
case_nlin_info = 'M%dN%dMs%d_nmodes%d' %(M, N, M_star_fact, num_modes)
case_rom_info = 'rom_MIMORA_r%d_sig%04d_%04dj' % (r, frequency_continuous_k[0].real*1000, frequency_continuous_k[0].imag*1000)
fig_folder = './figures/'
os.system('mkdir -p %s' % fig_folder)


# SHARPy nonlinear reference solution
ws = wings.FlyingWing(M=M,
                  N=N,
                  Mstar_fact=M_star_fact,
                  u_inf=u_inf,
                  alpha=alpha_deg,
                  rho=rho,
                  b_ref=0.305*2,
                  main_chord=0.076,
                  aspect_ratio=8,
                  sweep=0,
                  physical_time=2,
                  n_surfaces=2,
                  route='cases',
                  case_name=case_name)

ws.gust_intensity = 0.01
ws.sigma = 1.

# Mass and stiffness properties
rho_ply = 1520  # [kg/m3]
width = 76e-3  # [m]
length = 305e-3  # [m]
thickness = 0.134e-3  # [m] per ply
cross_section_area = width * thickness * 6  # [m2] whole laminate (6 plies)
mu = rho_ply * cross_section_area
j_x = rho * thickness * width ** 3 / 12
j_y = rho * thickness * length ** 3 / 12
j_z = j_x + j_y

# ws.mass = np.zeros((1, 6, 6))
# ws.mass[0, : , :] = np.diag([mu, mu, mu, mu/3, mu/3, mu/3])

ea = 5.49e4*76
eiy = 4.89e-1*width
eiz = 4.12*width
ga = 4.502e3*76
gj = 2.42e-1*width

ws.clean_test_files()
ws.update_derived_params(ea, ga, gj, eiy, eiz, mu, np.array([j_x, j_y, j_z]))
ws.update_aero_prop()
ws.n_tstep = 1
ws.update_fem_prop()
ws.set_default_config_dict()

ws.generate_aero_file()
ws.generate_fem_file()

ws.config['SHARPy']['flow'] = ['BeamLoader', 'AerogridLoader',
                        #'StaticUvlm',
                        'StaticCoupled',
                        'AerogridPlot', 'BeamPlot',
                        'DynamicCoupled',
                        'Modal',
                        'SaveData']
ws.config['SHARPy']['write_screen'] = 'on'
ws.config['Modal']['NumLambda'] = 40
ws.config['Modal']['rigid_body_modes'] = False
ws.config['DynamicCoupled']['aero_solver_settings']['velocity_field_input']['gust_length'] = 5
ws.config.write()

data = sharpy.sharpy_main.main(['',ws.route+ws.case_name+'.solver.txt'])


# Linearise - reduce UVLM and project onto modes

# Linearisation
# Original data point
tsaero0 = data.aero.timestep_info[-1]
tsstruct0 = data.structure.timestep_info[-1]

dt = data.settings['DynamicCoupled']['dt']
tsaero0.rho = rho

scaling_factors = {'length': 0.5*ws.c_ref,
                   'speed': u_inf,
                   'density': rho}

# Create aeroelastic system
aeroelastic_system = linaeroelastic.LinAeroEla(data)

beam = aeroelastic_system.lingebm_str  # Beam model

# structure to UVLM gains
aeroelastic_system.get_gebm2uvlm_gains()

# Clamped structure, no rigid body modes
zero_matrix = np.zeros((3*aeroelastic_system.linuvlm.Kzeta, beam.num_dof))
gain_struct_2_aero = np.block([[aeroelastic_system.Kdisp[:, :-10], zero_matrix],
                     [aeroelastic_system.Kvel_disp[:, :-10], aeroelastic_system.Kvel_vel[:, :-10]]])

gain_aero_2_struct = aeroelastic_system.Kforces[:-10, :]

# Linear structural solver solution settings
beam.dlti = True
beam.newmark_damp = 5e-3
beam.modal = True
beam.proj_modes = 'undamped'
beam.Nmodes = num_modes
beam.discr_method = 'newmark'

# Update structural model
beam.assemble()

# Group modes into symmetric and anti-symmetric modes
modes_sym = np.zeros_like(beam.U)
total_modes = len(beam.freq_natural)

for i in range(total_modes//2):
    je = 2*i
    jo = 2*i + 1
    modes_sym[:, je] = 1./np.sqrt(2)*(beam.U[:, je] + beam.U[:, jo])
    modes_sym[:, jo] = 1./np.sqrt(2)*(beam.U[:, je] - beam.U[:, jo])

beam.U = modes_sym

# Remove anti-symmetric modes
# Wing 1 and 2 nodes
# z-displacement index
ind_w1 = [6*i + 2 for i in range(N // 2)]  # Wing 1 nodes are in the first half rows
ind_w1_m = [6*i + 4 for i in range(N // 2)]  # Wing 1 nodes are in the first half rows
ind_w2 = [6*i + 2 for i in range(N // 2, N)]  # Wing 2 nodes are in the second half rows

# for i in range(total_modes//2):
#     plt.plot(beam.U[ind_w1, 2*i], color='r')
#     plt.plot(beam.U[ind_w2, 2*i], color='b', ls=':')
#     plt.plot(beam.U[ind_w1, 2*i+1], ls='-.')
#     plt.plot(beam.U[ind_w2, 2*i+1], ls='--')
#     plt.show()
#
# plt.close('all')
#
# for i in range(total_modes//2):
#     plt.plot(modes_sym[ind_w1, 2*i], color='r')
#     plt.plot(modes_sym[ind_w2, 2*i], color='b', ls=':')
#     plt.plot(modes_sym[ind_w1, 2*i+1], ls='-.')
#     plt.plot(modes_sym[ind_w2, 2*i+1], ls='--')
#     plt.show()
# plt.close('all')

sym_mode_index = []
for i in range(total_modes//2):
    found_symmetric = False

    for j in range(2):
        ind = 2*i + j

        # Maximum z displacement for wings 1 and 2
        ind_max_w1 = np.argmax(np.abs(modes_sym[ind_w1, ind]))
        ind_max_w2 = np.argmax(np.abs(modes_sym[ind_w2, ind]))
        z_max_w1 = modes_sym[ind_w1, ind][ind_max_w1]
        z_max_w2 = modes_sym[ind_w2, ind][ind_max_w2]

        z_max_diff = np.abs(z_max_w1 - z_max_w2)
        if z_max_diff < np.abs(z_max_w1 + z_max_w2):
            sym_mode_index.append(ind)
            if found_symmetric == True:
                raise NameError('Symmetric Mode previously found')
            found_symmetric = True

beam.U = modes_sym[:, sym_mode_index]
beam.freq_natural = beam.freq_natural[sym_mode_index]
natural_frequency_ref = beam.freq_natural.copy()  # Save reference natural frequencies

for i in range(5):
#     fig, ax = plt.subplots()
    print('Mode %d - Frequency %.2f rad/s  %.2f Hz' % (i, beam.freq_natural[i], beam.freq_natural[i]/2/np.pi))
#     ax.plot(beam.U[ind_w1, i])
#     ax2 = plt.twinx(ax)
#     ax2.plot(beam.U[ind_w1_m, i], ls='-.')
#     plt.show()

# UVLM Assembly
aeroelastic_system.linuvlm.assemble_ss()

# UVLM Scaling (UVLM is independent of the freestream velocity)
aeroelastic_system.linuvlm.nondimss()
length_ref = aeroelastic_system.linuvlm.ScalingFacts['length']
speed_ref = aeroelastic_system.linuvlm.ScalingFacts['speed']
force_ref = aeroelastic_system.linuvlm.ScalingFacts['force']
time_ref = aeroelastic_system.linuvlm.ScalingFacts['time']

# UVLM remove gust input - gusts not required in analysis
aeroelastic_system.linuvlm.SS.B = libsp.csc_matrix(
    aeroelastic_system.linuvlm.SS.B[:, :6*aeroelastic_system.linuvlm.Kzeta])
aeroelastic_system.linuvlm.SS.D = libsp.csc_matrix(
    aeroelastic_system.linuvlm.SS.D[:, :6*aeroelastic_system.linuvlm.Kzeta])

# UVLM projection onto modes
gain_input = libsp.dot(gain_struct_2_aero, sclalg.block_diag(beam.U[:, :beam.Nmodes], beam.U[:, :beam.Nmodes]))
gain_output = libsp.dot(beam.U[:, :beam.Nmodes].T, gain_aero_2_struct)
aeroelastic_system.linuvlm.SS.addGain(gain_input, where='in')
aeroelastic_system.linuvlm.SS.addGain(gain_output, where='out')

# Krylov MOR
rom = krylovrom.KrylovReducedOrderModel()
rom.initialise(data, aeroelastic_system.linuvlm.SS)

frequency_continuous_w = 2 * u_inf * frequency_continuous_k / ws.c_ref
interpolation_point = np.exp(frequency_continuous_w*aeroelastic_system.linuvlm.SS.dt)

rom.run(algorithm, r, interpolation_point)

# Frequency analysis
kn = np.pi / aeroelastic_system.linuvlm.SS.dt
kv = np.logspace(-3, np.log10(kn), 50)
kv = np.logspace(-3, np.log10(1), 50)
wv = 2*u_inf/ws.c_ref*kv

frequency_response_fom = aeroelastic_system.linuvlm.SS.freqresp(wv)
frequency_response_rom = rom.ssrom.freqresp(wv)
rom_max_magnitude = np.max(np.abs(frequency_response_rom))
error_frequency_response = np.max(np.abs(frequency_response_rom - frequency_response_fom))

fplot = freq_resp.FrequencyResponseComparison()
fsettings = {'frequency_type': 'k',
             'plot_type': 'real_and_imaginary'}
fplot.initialise(data, aeroelastic_system.linuvlm.SS, rom, fsettings)
i = 0
o = 0
in_range = 2*i
in_rangef = in_range + 2
out_range = 2*0
out_rangef = out_range+2
fplot.plot_frequency_response(kv, frequency_response_fom[out_range:out_rangef, in_range:in_rangef, :],
                              frequency_response_rom[out_range:out_rangef, in_range:in_rangef, :], interpolation_point)
fplot.save_figure(fig_folder + '/ROM_Freq_response_' + case_name + case_nlin_info + case_rom_info + '.pdf')

# Modal UVLM eigenvalues
eigs_UVLM = sclalg.eigvals(rom.ssrom.A)
eigs_UVLM_cont = np.log(eigs_UVLM)/rom.ssrom.dt

# Couple with structural model
u_inf_vec = np.linspace(1, 41, 21)
num_uinf = len(u_inf_vec)

# Initiliase variables to store eigenvalues for later plot
real_part_plot = []
imag_part_plot = []
uinf_part_plot = []

for i in range(num_uinf):

    # Scaling of properties
    u_ref = u_inf_vec[i]
    q_ref = 0.5*rho*u_ref**2
    t_ref = length_ref/u_ref
    force_c = q_ref * length_ref ** 2

    # Update structural model
    beam.dt = aeroelastic_system.linuvlm.SS.dt
    beam.freq_natural = natural_frequency_ref * t_ref
    beam.inout_coords = 'modes'
    beam.assemble()
    beam_ss = beam.SSdisc

    # Update BEAM -> UVLM gains due to scaling
    Tas = np.eye(2*num_modes) / length_ref

    # Update UVLM -> BEAM gains with scaling
    Tsa = np.diag((force_c*t_ref**2) * np.ones(beam.Nmodes))

    # Assemble new aeroelastic systems
    ss_aeroelastic = libss.couple(ss01=rom.ssrom, ss02=beam_ss, K12=Tas, K21=Tsa)

    dt_new = ws.c_ref / M / u_ref
    assert np.abs(dt_new - t_ref * aeroelastic_system.linuvlm.SS.dt) < 1e-14, 'dimensional time-scaling not correct!'

    # Asymptotic stability of the system
    eigs, eigvecs = sclalg.eig(ss_aeroelastic.A)
    eigs_mag = np.abs(eigs)
    order = np.argsort(eigs_mag)[::-1]
    eigs = eigs[order]
    eigvecs = eigvecs[:, order]
    eigs_mag = eigs_mag[order]
    frequency_dt_evals = 0.5*np.angle(eigs)/np.pi/dt_new

    # Nunst = np.sum(eigs_mag>1.)

    eigs_cont = np.log(eigs) / dt_new
    eigs_cont = eigs_cont[eigs_cont.imag >= 0]  # Remove lower plane symmetry
    Nunst = np.sum(eigs_cont.real > 0)
    eigmax = np.max(eigs_mag)
    fn = np.abs(eigs_cont)

    # Beam
    eigs_beam = sclalg.eigvals(beam.SSdisc.A)
    order = np.argsort(eigs_beam)[::-1]
    eigs_beam = eigs_beam[order]
    eigmax_beam = np.max(np.abs(eigs_beam))

    print('DLTI\tu: %.2f m/2\tmax.eig.: %.6f\tmax.eig.gebm: %.6f' \
          % (u_ref, eigmax, eigmax_beam))
    # print('\tGEBM nat. freq. (Hz):'+len(fn_gebm)*'\t%.2f' %tuple(fn_gebm))
    if Nunst == 0:
        print('\tStable Aeroelastic System')
    else:
        print('\tN unstab.: %.3d' % (Nunst,))
        print('\tUnstable aeroelastic natural frequency CT(rad/s):' + Nunst * '\t%.2f' %tuple(fn[:Nunst]))
        print('\tUnstable aeroelastic natural frequency DT(Hz):' + Nunst * '\t%.2f' %tuple(frequency_dt_evals[:Nunst]))

    # Store eigenvalues for plot
    real_part_plot.append(eigs_cont.real)
    imag_part_plot.append(eigs_cont.imag)
    uinf_part_plot.append(np.ones_like(eigs_cont.real)*u_ref)

real_part_plot = np.hstack(real_part_plot)
imag_part_plot = np.hstack(imag_part_plot)
uinf_part_plot = np.hstack(uinf_part_plot)

ax = plt.subplot(111)

ax.scatter(eigs_UVLM_cont.real, eigs_UVLM_cont.imag,
            color='grey',
            marker='^')

dataplot = ax.scatter(real_part_plot, imag_part_plot, c=uinf_part_plot,
                      cmap=plt.cm.winter,
                      marker='s',
                      s=7)

cb = plt.colorbar(dataplot)
cb.set_label('Freestream Velocity, $U_{\infty}$ [m/s]')

ax.set_xlim([-10,10])
ax.set_ylim([-5, 500])
ax.set_xlabel('Real $\lambda$ [rad/s]')
ax.set_ylabel('Imag $\lambda$ [rad/s]')
ax.grid()
plt.savefig(fig_folder + '/Root_locus_' + case_name + case_nlin_info + case_rom_info + '.eps')
plt.show()

# # First 3 modes plot
# ax = plt.subplot(111)
# for i in range(3)
#     ax.plot(eig)
