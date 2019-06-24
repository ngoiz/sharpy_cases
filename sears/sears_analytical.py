import numpy as np
import matplotlib.pyplot as plt
from scipy import special, signal

C_l_alpha = 2 * np.pi

def S_12(k):
    S_12 = 2 / (np.pi * k * (special.hankel1(0, k) + 1j * special.hankel1(1, k)))
    return S_12

def S_0(k):
    S_0 = np.e ** (-1j * k) * np.conj(S_12(k))
    return S_0

k_dom = np.arange(0, 3, 0.05)
sears = S_0(k_dom)
sears[0] = 1

# sears_mag = np.abs(sears)
#
# sears_phase = np.angle(sears, True)
#
# # RFA Approximation - Second Order Pade Approximation (from Palacios Ch3 pg 70)
# p = np.array([0.1405, 0.7931])
# z = np.array([4.034, 0.2286])
# k = 0.1159
#
# rfa = signal.ZerosPolesGain(z, p, k)
#
# w, H = signal.freqresp(rfa, k_dom)
#
#
# fig, ax = plt.subplots(nrows=2)
#
# ax[0].plot(k_dom, sears_mag)
# ax[0].plot(k_dom, np.abs(H))
# ax[1].plot(k_dom, sears_phase)
# ax[1].plot(k_dom, -np.angle(H, True))
# plt.show()
#
# # Reverse
# mag_diff = 0.5635
# red_freq = np.interp(mag_diff, k_dom, sears_mag)
