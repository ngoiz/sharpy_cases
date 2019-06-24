import sys
sys.path.append('/home/ng213/code/sharpy/')
import numpy as np
import os
import matplotlib.pyplot as plt
import sharpy.linear.src.libss as libss
import sharpy.linear.src.libsparse as libsp
import sharpy.linear.src.lingebm as lingebm
import sharpy.rom.krylovreducedordermodel as ROM
import sharpy.rom.frequencyresponseplot as freq_plots
import scipy.io as scio
import scipy as sc
import scipy.signal as sig

# Load matrices
A = scio.loadmat('A.mat')
B = scio.loadmat('B.mat')
C = scio.loadmat('C.mat')
A = A['A']
B = B['B']
C = C['C']
D = np.zeros((B.shape[1], C.shape[0]))

# Convert A to dense
As = sc.sparse.csc_matrix(A)
Ad = As.todense()

A = libsp.csc_matrix(As)

# Assemble continuous time system
fom_ss = libss.ss(A, B, C, D)
fom_sc = sig.lti(Ad, B, C, D)

# Compare frequency response
wv = np.logspace(-1, 3, 1000)
wvsc, mag_fom_sc, ph_fom_sc = fom_sc.bode(wv)
Y_fom_ss = fom_ss.freqresp(wv)[0, 0, :]
mag_fom_ss = 20*np.log10(np.abs(Y_fom_ss))
ph_fom_ss = np.angle(Y_fom_ss)*180/np.pi

print(np.max(np.abs(mag_fom_sc-mag_fom_ss)))

# Build rom
rom = ROM.KrylovReducedOrderModel()
rom.initialise(data=None, ss=fom_ss)
algorithm = "dual_rational_arnoldi"
r = 2
interpolation_points = np.array([0.0j, 1.0j, 10.0j, 100.j])

rom.run(algorithm, r, interpolation_points)

Y_rom = rom.ssrom.freqresp(wv)[0, 0, :]
mag_rom = 20*np.log10(np.abs(Y_rom))
ph_rom = np.angle(Y_rom)*180/np.pi

# Eigenvalues
evals_rom = np.linalg.eigvals(rom.ssrom.A)
evals_fom = sc.linalg.eigvals(Ad)

# fig, ax = plt.subplots(nrows=2, figsize=(16,9), sharex=True)
#
# ax[0].semilogx(wvsc, mag_fom_ss)
# ax[0].semilogx(wv, mag_rom, '--')
#
# ax[1].semilogx(wvsc, ph_fom_ss)
# ax[1].semilogx(wv, ph_rom, '--')
#
# ax[1].legend()
# plt.show()

case_name = 'cases/' + algorithm + '_r'+ str(r) +'_K' + str(len(interpolation_points)) + '/'

try:
    os.makedirs(case_name)
except FileExistsError:
    pass
np.save(case_name + 'wv', wv)
np.save(case_name + 'FOM', Y_fom_ss)
np.save(case_name + 'ROM', Y_rom)
np.save(case_name + 'interpolation_points', interpolation_points)
np.save(case_name + 'evals_rom', evals_rom)
np.save(case_name + 'evals_fom', evals_fom)

print(rom.ssrom.states)
# fig, ax = plt.subplots(nrows=2, figsize=(16,9), sharex=True)
#
# ax[0].semilogx(wvsc, mag_fom_sc)
# ax[0].semilogx(wv, mag_fom_ss, '--')
#
# ax[1].semilogx(wvsc, ph_fom_sc)
# ax[1].semilogx(wv, ph_fom_ss, '--')
#
# ax[1].legend()
# plt.show()

print('End of Generation routine')