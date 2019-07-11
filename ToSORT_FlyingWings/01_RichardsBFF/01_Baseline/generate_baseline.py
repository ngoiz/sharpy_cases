import sys
sys.path.append('/home/ng213/code/sharpy/')
# sys.path.append('/Users/Norberto/code/sharpy/')
from cases.hangar.richards_wing import Baseline
import sharpy.sharpy_main

ws = Baseline(M=4,
              N=11,
              Mstarfactor=5,
              u_inf=28,
              rho=1.02,
              alpha_deg=7.7563783342984385-4,
              cs_deflection_deg=-6.733360628875144*0,
              thrust=10.140622253017584)

ws.set_properties()
ws.initialise()
# ws.sweep_LE = 0
ws.clean_test_files()
ws.update_mass_stiffness(sigma=0.5, sigma_mass=1.5)
ws.update_fem_prop()
ws.generate_fem_file()
ws.update_aero_properties()
ws.generate_aero_file()
ws.set_default_config_dict()

ws.config['SHARPy']['flow'] = ['BeamLoader',
                               'AerogridLoader',
                               'StaticCoupled',
                               # 'StaticTrim',
                               'Modal',
                               'BeamPlot',
                               'AerogridPlot',
                               'LinearAssembler',
                               # 'SaveData',
                               'AsymptoticStability',
                               ]

ws.config['Modal']['rigid_body_modes'] = True


ws.config['LinearAssembler'] = {'flow': ['LinearAeroelastic'],
                               'LinearAeroelastic': {
                                   'beam_settings': {'modal_projection': False,
                                                     'inout_coords': 'nodes',
                                                     'discrete_time': True,
                                                     'newmark_damp': 0.5,
                                                     'discr_method': 'newmark',
                                                     'dt': ws.dt,
                                                     'proj_modes': 'undamped',
                                                     'use_euler': 'off',
                                                     'num_modes': 40,
                                                     'print_info': 'on',
                                                     'gravity': 'on',
                                                     'remove_dofs': []},
                                   'aero_settings': {'dt': ws.dt,
                                                     'integr_order': 2,
                                                     'density': ws.rho*1,
                                                     'remove_predictor': False,
                                                     'use_sparse': True,
                                                     'rigid_body_motion': True,
                                                     'use_euler': False,
                                                     'remove_inputs': ['u_gust']},
                                   'rigid_body_motion': True}}

ws.config['AsymptoticStability'] = {'sys_id': 'LinearAeroelastic',
                                    'print_info': 'on',
                                    'modes_to_plot': [0, 13, 15, 17, 19, 21, 23, 33],
                                    'display_root_locus': 'on',
                                    'frequency_cutoff': 0,
                                    'export_eigenvalues': 'off',
                                    'num_evals': 40,
                                    'folder': ws.case_route}

ws.config.write()

data = sharpy.sharpy_main.main(['', ws.case_route + '/' + ws.case_name + '.solver.txt'])
