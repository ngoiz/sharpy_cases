import sys
sys.path.append('/home/ng213/code/')
import cases.hangar.horten_wing as horten
import sharpy.sharpy_main
import numpy as np

M = 4
N = 7
Mstarfactor = 5

# Trim condition at u_inf=30
sigma = 1
u_inf = 30
cs_deflection_deg = 1.018791341281045
thrust = 1.018791341281045
alpha_deg = 1.018791341281045
rho = 1.02

flow = ['BeamLoader',
        'AerogridLoader',
        'StaticCoupled',
        # 'StaticTrim',
        'BeamPlot',
        'AerogridPlot',
        'DynamicCoupled',
        'AeroForcesCalculator',
        # 'Modal',
        # 'LinearAssembler',
        # 'AsymptoticStability']
]

ws = horten.HortenWing(M=M,
                 N=N,
                 Mstarfactor=Mstarfactor,
                 u_inf=u_inf,
                 rho=rho,
                 alpha_deg=alpha_deg,
                 beta_deg=0.,
                 cs_deflection_deg=cs_deflection_deg,
                 thrust=thrust,
                physical_time=5,
                case_name_format=10,
                       case_remarks='_dynamic')

# ws.sweep_LE = 20

ws.clean_test_files()
ws.update_mass_stiffness(sigma=sigma)
ws.update_fem_prop()
ws.update_aero_properties()
ws.generate_fem_file()
ws.generate_aero_file()
ws.set_default_config_dict()

ws.config['SHARPy']['flow'] = flow




ws.config.write()

data = sharpy.sharpy_main.main(['', ws.case_route + '/' + ws.case_name + '.solver.txt'])




