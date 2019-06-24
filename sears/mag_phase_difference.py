import numpy as np
from scipy import signal

def gain_and_phase(sig1, sig2, t, period, t_end_of_transient=0):
    """
    Args:
        sig1:
        sig2:
        t:
        t_end_of_transient:
    Returns:
    """
    # Cut signal to get rid of transients
    t_mod = t.copy()[t_end_of_transient:]
    sig1 = sig1.copy()[t_end_of_transient:]
    sig2 = sig2.copy()[t_end_of_transient:]

    # Shift to zero time
    t_mod -= t_mod[0]

    # Number of timesteps
    n_tsteps = len(t_mod)

    # Time shift for correlation
    dt_corr = np.linspace(-t_mod[-1], t_mod[-1], 2 * n_tsteps - 1)

    # Correlate functions
    f_corr = signal.correlate(sig1, sig2)

    # Time Shift
    delta_t = dt_corr[np.argmax(f_corr)]

    # plt.plot(dt_corr, f_corr)
    # plt.show()

    # Phase shift
    phase_shift = -2 * np.pi * (((0.5 + delta_t / period)%1) - 0.5)

    # Gain margin
    gain_margin = np.max(sig2) / np.max(sig1)

    return gain_margin, phase_shift


if __name__=='__main__':
    import matplotlib.pyplot as plt
    t = np.linspace(0,1,1000)
    omega = 6 * np.pi
    period = 2 * np.pi / omega
    print(period)
    phase = 15 * np.pi / 180

    ref = np.sin(omega * t)
    gain = 0.8
    pert = gain * np.sin(omega * t - phase)

    # plt.plot(t, ref)
    # plt.plot(t, pert)
    # plt.show()

    gain_c, phase_c = gain_and_phase(ref, pert, t, period, 25)
    print(gain_c)
    print(gain_c / gain)
    print(phase_c)
    print(phase_c / phase)
