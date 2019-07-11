import sys
sys.path.append('/home/ng213/code/')
import cases.hangar.horten_wing as horten
from cases.hangar.richards_wing import Baseline
import sharpy.sharpy_main
import numpy as np

M = 4
N = 11
Mstarfactor = 5

# Trim condition at u_inf=30
# sigma = 1
# sigma_mass = 1
# u_inf = 30
# cs_deflection_deg = 1.018791341281045
# thrust = 1.018791341281045
# alpha_deg = 1.018791341281045
# rho = 1.02

# Trim condition at u_inf=28
sigma = 0.5
sigma_mass = 1.5
# u_inf = 30
cs_deflection_deg = 0.21145532020938737
thrust =5.685822951655951
alpha_deg = 4.48151503886813
rho = 1.02

flow = ['BeamLoader',
        'AerogridLoader',
        'StaticCoupled',
        # 'StaticTrim',
        'BeamPlot',
        'AerogridPlot',
        'AeroForcesCalculator',
        'Modal',
        'LinearAssembler',
        'AsymptoticStability']
# ]

for u_inf in np.linspace(35, 45, 11):
    ws = Baseline(M=M,
                     N=N,
                     Mstarfactor=Mstarfactor,
                     u_inf=u_inf,
                     rho=rho,
                     alpha_deg=alpha_deg,
                     beta_deg=0.,
                     cs_deflection_deg=cs_deflection_deg,
                     thrust=thrust,
                    case_name_format=1)

    # ws.sweep_LE = 20
    ws.set_properties()
    ws.initialise()
    ws.clean_test_files()
    ws.update_mass_stiffness(sigma=sigma, sigma_mass=sigma_mass)
    ws.update_fem_prop()
    ws.update_aero_properties()
    ws.generate_fem_file()
    ws.generate_aero_file()
    ws.set_default_config_dict()

    ws.config['SHARPy']['flow'] = flow




    ws.config.write()

    data = sharpy.sharpy_main.main(['', ws.case_route + '/' + ws.case_name + '.solver.txt'])




