import sys
import os
sys.path.append('/home/ng213/code/sharpy/')
import cases.templates.flying_wings as wings
import sharpy.sharpy_main
import h5py as h5
import numpy as np

# Problem Set up
u_inf = 1.
alpha_deg = 00.
roll_deg = 00.
yaw_deg = 00.
rho = 1.225
num_modes = 4

# Lattice Discretisation
M = 6
N = 12
M_star_fact = 10
#
# M = 16  # 4
# N = 60
# M_star_fact = 18
# M = 30
# N = 60
# M_star_fact = 18

# Linear UVLM settings
integration_order = 2
remove_predictor = False
use_sparse = True

# Case Admin - Create results folders
case_name = 'flat_wing'
case_nlin_info = 'M%dN%dMs%d_a%04d' %(M, N, M_star_fact, alpha_deg*100)
# os.system('mkdir -p %s' % fig_folder)


# SHARPy nonlinear reference solution
ws = wings.FlyingWing(M=M,
                  N=N,
                  Mstar_fact=M_star_fact,
                  u_inf=u_inf,
                  alpha=alpha_deg,
                  yaw=yaw_deg,
                  roll=roll_deg,
                  rho=rho,
                    b_ref=10,
                      main_chord=1,
                      aspect_ratio=10,
                  sweep=0,
                  physical_time=2,
                  n_surfaces=2,
                  route='cases',
                  case_name=case_name)

ws.gust_intensity = 0.01
ws.sigma = 0.1
ws.gravity_on = False

ws.clean_test_files()
ws.update_derived_params()
ws.update_aero_prop()
ws.n_tstep = 1
ntsteps = 500
ws.main_ea = 0.4
ws.update_fem_prop()
ws.set_default_config_dict()

ws.generate_aero_file()
ws.generate_fem_file()

ws.config['SHARPy']['flow'] = ['BeamLoader', 'AerogridLoader',
                        #'StaticUvlm',
                        'StaticCoupled',
                        'AerogridPlot', 'BeamPlot',
#                        'DynamicCoupled',
                        'Modal',
#                         'SaveData']
                        'LinearAssembler',
                               'AsymptoticStability',
                               'LinDynamicSim',
]
ws.config['SHARPy']['write_screen'] = 'on'
ws.config['Modal']['NumLambda'] = 40
ws.config['Modal']['rigid_body_modes'] = True
ws.config['DynamicCoupled']['aero_solver_settings']['velocity_field_input']['gust_length'] = 5

# Linear Model
aero_settings = dict()
aero_settings['remove_inputs'] = ['u_gust']
aero_settings['dt'] = ws.dt
aero_settings['density'] = rho

beam_settings = dict()
beam_settings['modal_projection'] = False
beam_settings['discrete_time'] = True
beam_settings['dt'] = ws.dt
beam_settings['use_euler'] = False
beam_settings['gravity'] = False
beam_settings['remove_dofs'] = []
beam_settings['print_info'] = True

#
# beam_settings = {'modal_projection': True,
#                  'inout_coords': 'modes',
#                  'discrete_time': True,
#                  'newmark_damp': 0.5e-3,
#                  'discr_method': 'newmark',
#                  'dt': ws.dt,
#                  'proj_modes': 'undamped',
#                  'use_euler': False,
#                  'num_modes': num_modes}
# lin_settings = {'integr_order': integration_order,
#                  'remove_predictor': remove_predictor,
#                  'use_sparse': use_sparse,
#                 'density': rho,
#                 'dt': ws.dt,
#                  'ScalingDict': scaling_factors,
#                  'beam_settings': beam_settings}
ws.config['LinearAssembler'] = {'flow': ['LinearBeam', 'LinearUVLM', 'LinearCustom'],
                                'join_series': False,
                                'LinearCustom':{'solver_name': 'flex_solver',
                                                'solver_path': './cases'},
                                'LinearBeam': beam_settings,
                                'LinearUVLM': aero_settings,
                                'LinearAeroelastic': {
                                'aero_settings': aero_settings,
                                'beam_settings': beam_settings,
                                'rigid_body_motion': True}
                                }
ws.config['LinDynamicSim'] = {'dt': ws.dt,
                              'n_tsteps': ntsteps,
                              'sys_id': 'LinearCustom',
                              'postprocessors': ['BeamPlot', 'AerogridPlot'],
                              'postprocessors_settings': {'AerogridPlot': {
                                                              'u_inf': ws.u_inf,
                                                              'folder': ws.route + '/output/',
                                                              'include_rbm': 'on',
                                                              'include_applied_forces': 'on',
                                                              'minus_m_star': 0},
                              'BeamPlot': {'folder': ws.route + '/output/',
                                'include_rbm': 'on',
                                'include_applied_forces': 'on'}}}
                              # }
ws.config['AsymptoticStability'] = {'sys_id': 'LinearCustom',
                                    'print_info': 'on',
                                    'frequency_cutoff':100}
ws.config.write()

input_vec = np.zeros((ntsteps, 21))
input_vec = np.zeros((ntsteps, 237))
t_dom = np.linspace(0, ntsteps*ws.dt)
# input_vec[10:110, 6] = 0.996
# input_vec[10:110, 8] = 1*0.08715
input_vec[10:110, -6] = 1

def generate_linear_files(ws, input_vec):
    with h5.File(ws.route + '/' + ws.case_name + '.lininput.h5', 'a') as h5file:
        x0 = h5file.create_dataset(
            # 'x0', data=np.zeros((138)))
            # 'x0', data=np.zeros((950)))
            'x0', data=np.zeros((1094)))
        u = h5file.create_dataset(
            'u', data=input_vec)


generate_linear_files(ws, input_vec)

data = sharpy.sharpy_main.main(['',ws.route + ws.case_name + '.solver.txt'])

# Number of inputs

# # Linear input files

