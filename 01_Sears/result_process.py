# Result Processing
import os
import sys
import csv
import pandas as pd
sys.path.append('~/code/sharpy/')

import numpy as np
import matplotlib.pyplot as plt

rcParams.update({'figure.autolayout': True})
plt.rc('font', family='serif', serif='Times')


def read_variables(file_name):
    with open(file_name, 'r') as csvfile:
        line = csv.reader(csvfile, delimiter=',')


# Discretisation
M = 16
N = 80
MstarFact = 30
nsurf = 1
rho = 1.225
c_ref = 1.8288

# Flight Conditions
u_inf = 50
alpha_deg = 0
main_ea = 0.0
AR = 100

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
fig_folder = case_route + '/figs_esa/'
os.system('mkdir -p %s' % results_folder)
os.system('mkdir -p %s' % fig_folder)
case_name = 'sears_uinf%04d_AR%02d_M%dN%dMs%d_KR%d_sp%i' % (u_inf, AR, M, N, MstarFact, krylov_r, use_sparse)
#
data = pd.read_csv(results_folder + 'freq_data_' + case_name+'.csv')

Y_rom = np.zeros((len(data['kv'])))
Y_rom += data['Y_ROM_i']*1j + data['Y_ROM_r']
Y_rom *= u_inf / np.pi / 2 * np.exp(-1j*data['kv']*(-1.75/M)/0.5)
# data['Y_ROM_r'] *= np.real(u_inf / np.pi / 2 * np.exp(1j*data['kv']*(0.25/M)/0.5))
# data['Y_ROM_i'] *= np.imag(u_inf / np.pi / 2 * np.exp(1j*data['kv']*(0.25/M)/0.5))

plt.figure()
plt.plot(data['kv'], Y_rom.real,
         color='k',
         lw=2,
         label='ROM UVLM - Real')
plt.plot(data['kv'], Y_rom.imag,
         color='k',
         lw=2,
         ls='-.', label='ROM UVLM - Imag')
plt.plot(data['kv'], data['Y_sears_r'],
         color='b',
         ls='-',
         lw=4,
         alpha=0.5,
         label='Analytical - Real')
plt.plot(data['kv'], data['Y_sears_i'],
         color='b',
         ls='-',
         alpha=0.5,
         lw=4, label='Analytical - Imag')

plt.legend()
plt.grid()
plt.ylabel("Sears' Function Coefficient, $\mathcal{S}_{0}(ik)$")
plt.title('%d Panels - %d ROM States' % (M*N, krylov_r))
plt.xlabel(r'Reduced Frequency, $\frac{\omega U}{c/2}$ [-]')
plt.savefig(fig_folder + 'Sears_Freq_' + case_name + '.eps')
plt.savefig(fig_folder + 'Sears_Freq_' + case_name + '.png')
# plt.show()
#
# mag = np.sqrt(data['Y_ROM_r']**2 + data['Y_ROM_i']**2)
# phase= np.arctan(data['Y_ROM_i']/data['Y_ROM_r'])
mag = np.abs(Y_rom)
phase = np.angle(Y_rom)
sears_mag = np.sqrt(data['Y_sears_r']**2 + data['Y_sears_i']**2)
sears_phase= np.arctan(data['Y_sears_i']/data['Y_sears_r'])

fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].set_title("Sears' function $\mathcal{S}_0(ik)$ - %d UVLM Panels - %d ROM States" % (M*N, krylov_r))
ax[0].plot(data['kv'], mag, color='k', lw=2)
ax[0].plot(data['kv'], sears_mag, color='b', ls='-', alpha=0.5, lw=4)
ax[1].plot(data['kv'], phase * 180/np.pi, color='k', lw=2, label='ROM')
ax[1].plot(data['kv'], sears_phase * 180/np.pi, color='b', ls='-', alpha=0.5, label='Analytical', lw=4)
ax[1].legend()
ax[1].set_xlabel(r'Reduced Frequency, $\frac{\omega U}{c/2}$ [-]')
ax[0].set_ylabel('Magnitude, abs [-]')
ax[1].set_ylabel('Phase, $\Phi$ [deg]')
# ax[1].set_yticklabels([0, -15, -30, -45][::-1])
ax[1].set_ylim([-50, 5])
ax[1].grid()
ax[0].grid()
plt.show()
# plt.savefig(fig_folder + 'Bode_' + case_name + '.png')
# plt.savefig(fig_folder + 'Bode_' + case_name + '.pdf')


# # Kussner
# data = pd.read_csv(results_folder + 'kussner_data_' + case_name+'.csv')
# S_ref = c_ref ** 2 / AR
# Fz = 0.5 * 1.225 * u_inf ** 2 * S_ref
# phi = data['Fz'] / 2 / np.pi * u_inf
# t_bar = u_inf * data['t'] / c_ref * 2
# phi_analytical = 1 - 0.5 * np.exp(-0.13*t_bar)-0.5*np.exp(-1*t_bar)
#
# plt.figure()
# plt.title('Kussner Gust Response - %d UVLM Panels - %d ROM States' % (M*N, krylov_r))
# plt.plot(t_bar, phi, lw=2, color='k', label='ROM')
# plt.plot(t_bar, phi_analytical,lw=4, alpha=0.5, color='b', label='Kussner Function')
# plt.xlabel(r'Reduced Time, $\frac{Ut}{c/2}$ [s]')
# plt.ylabel(r'Kussner Function Coefficient, $\psi$ [-]')
# plt.grid()
# plt.legend()
# plt.xlim([0, 20])
# # plt.show()
# plt.savefig(fig_folder + 'Kussner_' + case_name + '.png')
# plt.savefig(fig_folder + 'Kussner_' + case_name + '.pdf')

# plt.figure()
# plt.plot(t_bar, data['Fz']*np.sin(0.2))
