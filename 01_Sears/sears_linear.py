# Sears Gust Response verification
# N Goizueta
# March 19

import os
import sys
sys.path.append('~/code/sharpy/')

import numpy as np
import scipy as sc

import cases.templates.flying_wings as wings
import sharpy.sharpy_main
import sharpy.linear.src.lin_aeroelastic as linaeroela
import sharpy.linear.src.libsparse as libsp
import sharpy.linear.src.libss as libss
import sharpy.rom.krylovreducedordermodel as krylovrom
import sharpy.utils.analytical as analytical

def save_variables(file_name, vars, var_title):
    fid = open(file_name, 'w')

    title_line = len(var_title)*'%s,' % var_title

    fid.write(title_line+'\n')

    for elem in range(vars[0].shape[0]):
        # var_line = len(var_title)*'%8f\t' % tuple(var_title[elem, :])
        vars_in_line = []
        vars_in_line.append([vars[i][elem] for i in range(len(var_title))])
        # print(vars_in_line[0])
        var_line = ''.join('%f,' % item for item in vars_in_line[0])
        # print(var_line)
        fid.write(var_line+'\n')

    fid.close()

# Case Setup
# Discretisation
M = 16
N = 8 #80
MstarFact = 100
nsurf = 1
rho = 1.225

# Flight Conditions
u_inf = 50
alpha_deg = 0
main_ea = 0.0
AR = 4000

# Linear settings
remove_predictor = False
use_sparse = False
integration_order = 2

# ROM Settings
algorithm = 'dual_rational_arnoldi'
frequency_continuous_k = np.array([0.0])
krylov_r = 15

# Case Admin
case_route = os.path.abspath('.')
results_folder = case_route + '/res/'
fig_folder = case_route + '/figs/'
os.system('mkdir -p %s' % results_folder)
os.system('mkdir -p %s' % fig_folder)
case_name = 'sears_uinf%04d_AR%02d_M%dN%dMs%d_KR%d_sp%i' % (u_inf, AR, M, N, MstarFact, krylov_r, use_sparse)

# Wing model
ws = wings.Goland(M=M,
                         N=N,
                         Mstar_fact=MstarFact,
                         n_surfaces=nsurf,
                         u_inf=u_inf,
                         rho = rho,
                         alpha=alpha_deg,
                         aspect_ratio=AR,
                         route=results_folder,
                         case_name=case_name)

ws.main_ea = main_ea
ws.clean_test_files()
ws.update_derived_params()
ws.generate_fem_file()
ws.generate_aero_file()

# Solution settings

ws.set_default_config_dict()
ws.config['SHARPy']['flow'] = ['BeamLoader', 'AerogridLoader', 'Modal', 'StaticUvlm', 'BeamPlot','AerogridPlot']

ws.config['LinearUvlm'] = {'dt': ws.dt,
                           'integr_order': integration_order,
                           'density': ws.rho,
                           'remove_predictor': remove_predictor,
                           'use_sparse': use_sparse,
                           'ScalingDict': {'length': 1.,
                                           'speed': 1.,
                                           'density': 1.}}
ws.config['Modal']['NumLambda'] = 40
ws.config['Modal']['keep_linear_matrices'] = 'on'
ws.config['Modal']['use_undamped_modes'] = True
ws.config.write()

# Solve nonlinear solution
data = sharpy.sharpy_main.main(['...', results_folder + case_name + '.solver.txt'])
tsaero = data.aero.timestep_info[-1]
tsstruct = data.structure.timestep_info[-1]

# Linearisation parameters
dt = ws.dt
tsaero.rho = ws.rho
scaling_factors = {'length': 1,#0.5*ws.c_ref,
                   'speed': 1,#u_inf,
                   'density': 1}#ws.rho}

# Linearise UVLM
aeroelastic_system = linaeroela.LinAeroEla(data)
uvlm = aeroelastic_system.linuvlm
uvlm.assemble_ss()
aeroelastic_system.get_gebm2uvlm_gains()

# Remove lattice coordinates and velocities from the inputs to the system
uvlm.SS.B = libsp.csc_matrix(uvlm.SS.B[:, -uvlm.Kzeta:])
uvlm.SS.D = libsp.csc_matrix(uvlm.SS.D[:, -uvlm.Kzeta:])

# Create system to transmit a vertical gust across the chord in time
A_gust = np.zeros((M+1, M+1))
A_gust[1:, :-1] = np.eye(M)
B_gust = np.zeros((M+1, 1))
B_gust[0] = 1
C_gust = np.eye(M+1)
D_gust = np.zeros((C_gust.shape[0],1))
print(D_gust.shape)
if use_sparse:
    A_gust = libsp.csc_matrix(A_gust)
    B_gust = libsp.csc_matrix(B_gust)
    C_gust = libsp.csc_matrix(C_gust)
    D_gust = libsp.csc_matrix(D_gust)
print(D_gust.shape)
ss_gust = libss.ss(A_gust, B_gust, C_gust, D_gust, dt=ws.dt)
# print(ss_gust_airfoil.A.shape)
# print(ss_gust_airfoil.B.shape)
# print(ss_gust_airfoil.C.shape)
# print(ss_gust_airfoil.D.shape)
# B_temp = B_gust.copy()
# B_temp.shape = (M+1, 1)
# D_temp = D_gust.copy()
# D_temp.shape = (M+1, 1)
# sc_gust_airfoil = sc.signal.dlti(A_gust, B_temp, C_gust, D_temp, dt=dt)

# Gain to get uz at single chordwise position across entire span
K_lattice_gust = np.zeros((uvlm.SS.inputs, ss_gust.outputs))
for i in range(M+1):
    K_lattice_gust[i*(N+1):(i+1)*(N+1), i] = np.ones((N+1,))

# Add gain to gust generator
ss_gust.addGain(K_lattice_gust, where='out')

# UVLM - output: obtain vertical force
uvlm.SS.addGain(aeroelastic_system.Kforces, where='out')

K_Fz = np.zeros((1,aeroelastic_system.Kforces.shape[0]))
# Output - Vertical force coefficient

qS = 0.5 * ws.rho * u_inf ** 2 * ws.wing_span * ws.c_ref

wdof = 0
for node in range(data.structure.num_node):

    node_bc = data.structure.boundary_conditions[node]
    if node_bc != 1:
        node_ndof = 6
        vertical_force_index = np.array([0, 0, 1, 0, 0, 0]) / qS
        K_Fz[:, wdof: wdof + node_ndof] = vertical_force_index
    else:
        node_ndof = 0

    wdof += node_ndof

uvlm.SS.addGain(K_Fz, where='out')

# Join systems
sears_ss = libss.series(ss_gust, uvlm.SS)
# wt = np.linspace(0.01, 14, 100)
# Y_test = sears_ss.freqresp(wt)
#
# plt.plot(wt, Y_test[0, 0, :].real)
# plt.plot(wt, Y_test[0, 0, :].imag)
# plt.savefig(fig_folder + 'Test_MxG.eps')

# ROM
rom = krylovrom.KrylovReducedOrderModel()
rom.initialise(data, sears_ss)
frequency_continuous_w = 2 * u_inf * frequency_continuous_k / ws.c_ref
frequency_dt = np.exp(frequency_continuous_k*dt)
rom.run(algorithm, krylov_r, frequency_dt)

# Time Domain Simulation
# Full model
sc_ss = sc.signal.dlti(sears_ss.A, sears_ss.B, sears_ss.C, sears_ss.D, dt=dt)
sc_ss_rom = sc.signal.dlti(rom.ssrom.A, rom.ssrom.B, rom.ssrom.C, rom.ssrom.D, dt=dt)

# Gust inputs
Nsteps = 1000
t_dom = np.linspace(0, dt*Nsteps, Nsteps+1)
k_gust = 0.2
omega_g = 2 * u_inf * k_gust / ws.c_ref
u_g = np.sin(omega_g*t_dom)

# # Simulate
out_fom = sc.signal.dlsim(sc_ss, u_g)
out_rom = sc.signal.dlsim(sc_ss_rom, u_g)

# Sears analytical time response
S_analytical = analytical.sears_fun(k_gust)
S_gain = np.abs(S_analytical)
S_phase = np.angle(S_analytical)
CL_gust = 2 * np.pi / u_inf * S_gain * np.sin(omega_g * t_dom + S_phase)

# # Plot time domain outputs
# plt.figure()
# plt.plot(t_dom, out_fom[1], label='FOM')
# plt.plot(t_dom, out_rom[1], label='ROM')
# plt.plot(t_dom, CL_gust, label='Analytical')
# plt.xlabel('Time, t [s]')
# plt.ylabel('Coefficient of Lift, $C_L$ [-]')
# plt.legend()
# plt.grid()
# plt.savefig(fig_folder + 't_dom_' + case_name + '.eps')
#
save_variables(results_folder + 't_dom_' + case_name + '.csv', [t_dom, out_fom[1][:], out_rom[1][:], CL_gust],
               ('t', 'CL_FOM', 'CL_ROM', 'CL_sears'))


# # Gust profile over airfoil
# out_gust = sc.signal.dlsim(sc_gust_airfoil, u_g)
#
# # Plot single time instance
# plt.figure()
# for n in range(16,21):
#     lab = 't%.4f s' % t_dom[n]
#     plt.plot(np.arange(M+1)*ws.c_ref/M, out_gust[1][n, :], label=lab)
# # plt.plot(np.arange(M+1)*ws.c_ref/M, out_gust[1][20, :], label='t20')
# plt.xlabel('Chordwise position, x [m]')
# plt.ylabel('Gust Vertical Velocity')
# plt.legend()
# plt.savefig(fig_folder + 'gust_airfoil' + case_name + '.eps')

# Frequency analysis
ds = 2. / M
fs = 1. / ds
fn = fs / 2.
ks = 2. * np.pi * fs
kn = 2. * np.pi * fn
Nk = 151
# kv = np.logspace(-3, np.log10(1), Nk)
kv = np.linspace(0.01, 3, Nk)
wv = 2. * u_inf / ws.c_ref * kv

# # Analytical answer
Y_analytical = analytical.sears_fun(kv)

Y_freq_resp = libss.freqresp(sears_ss, wv) * u_inf / 2 / np.pi

Y_freq_resp_rom = libss.freqresp(rom.ssrom, wv)

save_variables(results_folder + 'freq_data_' + case_name + '.csv', [wv, kv, Y_analytical.real, Y_analytical.imag, Y_freq_resp_rom[0,0,:].real, Y_freq_resp_rom[0,0,:].imag,
                                                                    Y_freq_resp[0, 0, :].real, Y_freq_resp[0, 0, :].imag],
('wv', 'kv', 'Y_sears_r', 'Y_sears_i', 'Y_ROM_r', 'Y_ROM_i','Y_FOM_r', 'Y_FOM_i'))


sc_kussner_A = sears_ss.A
sc_kussner_B = sears_ss.B.copy()
sc_kussner_B.shape = (sc_kussner_A.shape[0], 1)
sc_kussner_C = sears_ss.C
sc_kussner_D = sears_ss.D.copy()
sc_kussner_D.shape = (sc_kussner_C.shape[0], 1)

dlti_kussner = sc.signal.dlti(sc_kussner_A, sc_kussner_B, sc_kussner_C, sc_kussner_D, dt=dt)

Nsteps = 1000
t_dom = np.linspace(0, dt*Nsteps, Nsteps+1)

out = sc.signal.dstep(dlti_kussner, t=t_dom)

save_variables(results_folder + 'kussner_data_' + case_name + '.csv', [t_dom, out[1][0]],
('t', 'Fz'))

# Y_freq_resp_rom *= u_inf / np.pi / 2 * np.exp(1j*kv*(1.75*ws.c_ref/M)/(0.5*ws.c_ref))
#
# # plt.plot(kv, Y_freq_resp_rom[0,0,:].real,
# #          color='r',
# #          lw=2, label='Linear UVLM - Real')
# # plt.plot(kv, Y_freq_resp_rom[0,0,:].imag,
# #          color='r',
# #          ls='-', label='Linear UVLM - Imag')
#
# plt.figure()
# plt.plot(kv, Y_freq_resp_rom[0,0,:].real,
#          color='k',
#          lw=2, label='ROM UVLM - Real')
# plt.plot(kv, Y_freq_resp_rom[0,0,:].imag,
#          color='b',
#          ls='-', label='ROM UVLM - Imag')
# plt.plot(kv, Y_analytical.real,
#          color='k',
#          ls='--',
#          lw=2,
#          label='Analytical - Real')
# plt.plot(kv, Y_analytical.imag,
#          color='b',
#          ls='--',
#          lw=2, label='Analytical - Imag')
#
# plt.legend()
# plt.grid()
# plt.ylabel("Sears' Function Coefficient, $\mathcal{S}_{0}(ik)$")
# plt.title('%d Panels - %d UVLM States - %d ROM States' % (M*N, uvlm.SS.states, rom.ssrom.states))
# plt.xlabel('Reduced Frequency, k [-]')
# plt.savefig(fig_folder + 'Sears_Freq_' + case_name + '.eps')
# # plt.show()
# #
# mag = np.abs(Y_freq_resp_rom[0,0,:])
# phase= np.angle(Y_freq_resp_rom[0,0,:])
#
# fig, ax = plt.subplots(nrows=2, sharex=True)
# ax[0].plot(kv, mag, color='k')
# ax[0].plot(kv, np.abs(Y_analytical), color='b', ls='-', alpha=0.6)
# ax[1].plot(kv, phase, color='k', label='ROM')
# ax[1].plot(kv, np.angle(Y_analytical), color='b', ls='-', alpha=0.6, label='Analytical')
# ax[1].legend()
# ax[1].set_xlabel('Reduced Frequency, k [-]')
# ax[0].set_ylabel('Magnitude, abs [-]')
# ax[1].set_ylabel('Phase, $\Phi$ [rad]')
# plt.savefig(fig_folder + 'Bode_' + case_name + '.eps')
# plt.show()
#
# plot_settings = {'frequency_type': 'k', 'plot_type':'bode'}
# fp = freqplot.FrequencyResponseComparison()
# fp.initialise(data, sears_ss, rom, plot_settings)
# fp.plot_frequency_response(kv, Y_freq_resp, Y_freq_resp_rom, np.array([0]))
# fp.savefig(fig_folder + 'Sears_rom_' + case_name + '.pdf')
# # Nsteps = 100
# t_dom = np.linspace(0, Nsteps*dt, Nsteps + 1)
# uz = np.sin(t_dom)
# ss_gust.B.shape = (M+1, 1)
# ss_sc = sc.signal.dlti(ss_gust.A, ss_gust.B, ss_gust.C, None, dt=ss_gust.dt)
# out = sc.signal.dlsim(ss_sc, uz)

print('End of Routine')
