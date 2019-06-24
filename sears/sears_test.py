"""
Sears Function Comparison
"""

import cases.templates.flying_wings as flying_wings
import sharpy.sharpy_main
import numpy as np
from scipy import signal
from tests.coupled.prescribed.sears.sears_analytical import S_0
from tests.coupled.prescribed.sears.mag_phase_difference import gain_and_phase

import matplotlib.pyplot as plt

make_plots = True
# Create a reference system
reduced_freq_domain = np.linspace(0.05, 1, 9)
# reduced_freq_domain = np.linspace(1.75, 2, 2)

# Properties
i = -1
sears_gain = np.zeros_like(reduced_freq_domain)
sears_phase = np.zeros_like(sears_gain)
mag_diff = np.zeros_like(reduced_freq_domain)
phase_shift = np.zeros_like(reduced_freq_domain)
mag_diff_nl = np.zeros_like(reduced_freq_domain)
phase_shift_nl = np.zeros_like(reduced_freq_domain)

for reduced_frequency in reduced_freq_domain:
    i += 1
    # reduced_frequency = 0.25
    # M, N, Mstarfact = 8, 4, 50
    M, N, Mstarfact = 12, 16, 50
    u_inf = 20
    alpha = 0
    AR = 30
    rho = 1.225


    ws = flying_wings.Goland(M=M,
                                    N=N,
                                    Mstar_fact=Mstarfact,
                                    u_inf=u_inf,
                                    alpha=alpha,
                                    rho=rho,
                                    aspect_ratio=AR,
                                    case_name='sears',
                                    route='./cases')

    gust_length = np.pi * ws.c_ref / reduced_frequency
    gust_intensity = 0.01

    N_periods = 4
    omega = reduced_frequency * 2 * u_inf / ws.c_ref
    period = 2 * np.pi / omega
    ws.physical_time = N_periods * period


    for sim_type in ['nlin', 'lin']:

        ws.case_name = 'sears' + sim_type + 'k_%04.0f_uns' %(100*reduced_frequency)
        ws.gust_length = gust_length
        ws.horseshoe = False
        # ws.dt_factor = 0.5
        ws.gust_intensity = gust_intensity
        ws.gravity_on = False

        ws.clean_test_files()
        ws.update_derived_params()
        ws.update_aero_prop()
        ws.update_fem_prop()
        ws.generate_fem_file()
        ws.generate_aero_file()
        ws.set_default_config_dict()

        ws.config['SHARPy']['flow'] = ['BeamLoader', 'AerogridLoader',
                                       'StaticUvlm',
                                       'DynamicUVLM']
                                       # 'SaveData']


        if sim_type == 'lin':
            ws.config['DynamicUVLM']['aero_solver'] = 'StepLinearUVLM'
            ws.config['DynamicUVLM']['aero_solver_settings'] = {'dt': ws.dt,
                                                                'integr_order': 1,
                                                                'remove_predictor': True,
                                                                # 'velocity_field_generator': 'SteadyVelocityField',
                                                                # 'velocity_field_input': {'u_inf': u_inf,
                                                                #                          'u_inf_direction': [1., 0, 0]}
                                                                'velocity_field_generator': 'GustVelocityField',
                                                                'velocity_field_input': {'u_inf': u_inf,
                                                                                                      'u_inf_direction': [1., 0, 0],
                                                                                                      'gust_shape': 'continuous_sin',
                                                                                                      'gust_length': gust_length,
                                                                                                      'gust_intensity': gust_intensity * u_inf,
                                                                                                      'offset': 2.,
                                                                                                      'span': ws.wing_span}
                                                                }
        else:
            ws.config['DynamicUVLM']['aero_solver'] = 'StepUvlm'
            ws.config['DynamicUVLM']['aero_solver_settings'] = {
                'print_info': 'off',
                'horseshoe': True,
                'num_cores': 4,
                'n_rollup': 100,
                'convection_scheme': 0,
                'rollup_dt': ws.dt,
                'rollup_aic_refresh': 1,
                'rollup_tolerance': 1e-4,
                'velocity_field_generator': 'GustVelocityField',
                'velocity_field_input': {'u_inf': ws.u_inf,
                                         'u_inf_direction': [1., 0, 0],
                                         'gust_shape': 'continuous_sin',
                                         'gust_length': ws.gust_length,
                                         'gust_intensity': ws.gust_intensity * ws.u_inf,
                                         'offset': 2.0,
                                         'span': ws.main_chord * ws.aspect_ratio},
                'rho': ws.rho,
                'n_time_steps': ws.n_tstep,
                'dt': ws.dt,
                'gamma_dot_filtering': 0,
                'part_of_fsi': False}
            ws.config['DynamicUVLM']['include_unsteady_force_contribution'] = 'on'

        ws.config['SHARPy']['write_screen'] = True
        ws.config.write()

        data = sharpy.sharpy_main.main(['',ws.route + '/' + ws.case_name +'.solver.txt'])


        #Total forces
        # data.aero.linear['System'].get_total_forces_gain()

        n_tsteps = len(data.aero.timestep_info)
        forces = np.zeros((n_tsteps, 3))
        forces_z = np.zeros((n_tsteps, 1))
        ux_gust = np.zeros((n_tsteps, 1))
        uz_gust = np.zeros((n_tsteps, 1))

        for N in range(n_tsteps):
            aero_tstep = data.aero.timestep_info[N]

            y_vec = np.concatenate([aero_tstep.forces[isurf][0:3].reshape(-1, order='C')
                                    for isurf in range(len(aero_tstep.forces))])

            # forces[N :] = data.aero.linear['System'].Kftot.dot(y_vec)
            forces_z[N] = aero_tstep.forces[0][2].sum() + aero_tstep.dynamic_forces[0][2].sum()
            ux_gust[N] = data.aero.timestep_info[N].u_ext[0][0, 0, 0]
            uz_gust[N] = data.aero.timestep_info[N].u_ext[0][2, 0, 0]

        S = ws.wing_span * ws.c_ref
        rho = ws.rho

        q = 0.5 * rho * u_inf ** 2

        qS = q * S
        if sim_type == 'lin':
            CL = forces_z / qS
        else:
            Clnlin = forces_z / qS

    # Instantaneous angle of attack
    alpha = np.arctan(uz_gust/ux_gust)
    CLqs = 2 * np.pi * alpha



    # Gust in space at an arbitrary time
    # uz_space = data.aero.timestep_info[9].u_ext[0][2, :,0]
    # zeta_space = data.aero.timestep_info[9].zeta[0][0, :, 0]
    # plt.plot(zeta_space, uz_space)
    # plt.show()

    # Phase difference
    period = gust_length / u_inf
    omega = 2 * np.pi / period
    reduced_frequency = omega * ws.c_ref / 2 / u_inf
    
    tmax = (n_tsteps - 1) * ws.dt
    t = np.linspace(ws.dt, tmax, n_tsteps)
    dta = np.linspace(-t[-1], t[-1], 2 * n_tsteps - 1)
    def freq_difference(CL):
        xcorr = signal.correlate(CL, CLqs)
        time_shift = dta[xcorr.argmax()]
        phase_shift = 2 * np.pi * (((0.5 + time_shift / period)%1)-0.5)
        # phase_shift_deg = phase_shift[i] * 180 / np.pi
        mag_diff = CL.max() / CLqs.max()
        return mag_diff, phase_shift

    # mag_diff[i], phase_shift[i] = freq_difference(CL)
    # mag_diff_nl[i], phase_shift_nl[i] = freq_difference(Clnlin)
    mag_diff[i], phase_shift[i] = gain_and_phase(CLqs, CL, t, period, n_tsteps//5)
    mag_diff_nl[i], phase_shift_nl[i] = gain_and_phase(CLqs, Clnlin, t, period, n_tsteps//5)


    # Sears function
    sears_gain[i] = np.abs(S_0(reduced_frequency))
    sears_phase[i] = np.angle(S_0(reduced_frequency))

    # LE x coordinate
    x_LE = data.aero.timestep_info[0].zeta[0][0, 0, 0]
    x_0_gust_offset = 2
    gust_offset_LE = x_0_gust_offset + x_LE
    sears_Cl = - sears_gain[i] * np.sin(omega*(t-gust_offset_LE/u_inf)+sears_phase[i]) * 2 * np.pi * gust_intensity / 2

    if make_plots == True:
        fig, ax = plt.subplots()
        ax.plot(t, CLqs, '-', color='k', label='Instantaneous')

        ax.plot(t, CL, label='Linear')
        ax.plot(t, Clnlin, ls='--', label='Nonlinear')
        ax.plot(t, sears_Cl, label='Sears')
        ax.set_xlabel('Time, t [s]')
        ax.set_ylabel('Lift Coefficient, $C_L$ [-]')
        ax.set_title('Reduced frequency k = %f' %reduced_frequency)
        plt.legend()
        plt.show()
        print('End')

    # plt.plot(t, uz_gust)
    # plt.show()

fig, ax = plt.subplots(nrows=2)
ax[0].plot(reduced_freq_domain, mag_diff, label='Linear')
ax[0].plot(reduced_freq_domain, mag_diff_nl, label='Nonlinear')
ax[0].plot(reduced_freq_domain, sears_gain, label='Sears')
ax[0].set_ylabel('Absolute Gain, M [-]')

ax[1].plot(reduced_freq_domain, -phase_shift, label='Linear')
ax[1].plot(reduced_freq_domain, -phase_shift_nl, label='Nonlinear')
ax[1].plot(reduced_freq_domain, sears_phase, label='Sears')
ax[1].set_ylabel('Phase Difference, $\Phi$, [rad]')
ax[1].set_xlabel('Reduced Frequency, k')
plt.legend()
plt.show()
# fig.savefig('FrequencyResponse_uns_c4.eps')

fig, ax = plt.subplots(nrows=2)
ax[0].plot(reduced_freq_domain, mag_diff, label='Linear')
ax[0].plot(reduced_freq_domain, mag_diff_nl, label='Nonlinear')
ax[0].plot(reduced_freq_domain, sears_gain, label='Sears')
ax[0].set_ylabel('Absolute Gain, M [-]')

ax[1].plot(reduced_freq_domain, -phase_shift * 180 / np.pi, label='Linear')
ax[1].plot(reduced_freq_domain, -phase_shift_nl * 180 / np.pi, label='Nonlinear')
ax[1].plot(reduced_freq_domain, sears_phase * 180 / np.pi, label='Sears')
ax[1].set_ylabel('Phase Difference, $\Phi$, [deg]')
ax[1].set_xlabel('Reduced Frequency, k')
plt.legend()
plt.show()
fig.savefig('FrequencyResponse_uns_c4_deg_m16.eps')

# Embedded frequency response
# linsys = data.aero.linear['SS']
# H = libss.freqresp(linsys, np.linspace(0,3,5))

# fourier = fftpack.fft(CL)
# freqs = fftpack.fftfreq(len(CL), ws.dt)
#
# plt.plot(freqs, np.abs(fourier))
# plt.show()

# 7/1/19
# PLOT NOT SHOWING
# VERIFY SIN GUST RESPONSE



# linearise the system
# data.aero.timestep_info[-1].rho = ws.rho
#
# linsys = linuvlm.Dynamic(data.aero.timestep_info[0],
#                          dt=ws.dt,
#                          integr_order=2,
#                          RemovePredictor=True,
#                          ScalingDict=None,
#                          UseSparse=False)
#
# linsys.assemble_ss()
#
#
# # Sinusoidal gust input
# # Gust wavelength
# gust_wlength = 10
# gust_intensity = 0.1
# u_inf_direction = [1., 0., 0.]
#
# dt = 0.01
# NT = 100
#
#
# def gust_field(x, N):
#     v_gust = gust_intensity * np.sin(2 * np.pi * (u_inf * dt * N - x))
#
#     return v_gust
#
# # Generate input
# zeta = data.aero.timestep_info[0].zeta
# zeta_star = data.aero.timestep_info[0].zeta_star
#
# N = 0
#
#
# def u_aero_gust(N):
#
#     u_ext = data.aero.timestep_info[0].u_ext
#
#     for i_surf in range(len(zeta)):
#         for i in range(zeta[i_surf].shape[1]):
#             for j in range(zeta[i_surf].shape[2]):
#                 u_ext[i_surf][:, i, j] = np.dot(u_inf, u_inf_direction)
#                 u_ext[i_surf][2, i, j] = gust_field(zeta[i_surf][0, i, j], N)
#
#     # Reorder velocity into column vector
#     u_ext_vec = np.concatenate([u_ext[i_surf].reshape(-1, order='C')
#                                 for i_surf in range(len(u_ext))])
#     zeta_vec = np.concatenate([zeta[i_surf].reshape(-1, order='C')
#                                for i_surf in range(len(zeta))])
#     zeta_star_vec = np.concatenate([zeta_star[i_surf].reshape(-1, order='C')
#                                     for i_surf in range(len(zeta_star))])
#     u_aero = np.concatenate((zeta_vec, zeta_star_vec, u_ext_vec))
#
#     return u_aero
#
# # Initial Conditions

