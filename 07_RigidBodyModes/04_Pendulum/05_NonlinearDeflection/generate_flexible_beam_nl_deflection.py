######################################################################
##################  PYTHON PACKAGES  #################################
######################################################################
# Usual SHARPy
import numpy as np
import os
import sys
sys.path.append('/home/ng213/code/sharpy/')
import sharpy.utils.generate_cases as gc
import sharpy.utils.algebra as algebra
# Generate errors during execution
import sys

deg2rad = np.pi/180

######################################################################
##################  DEFINE CASE  #####################################
######################################################################
case_name = 'flexible_beam_40deg'
route = os.path.dirname(os.path.realpath(__file__)) + '/cases/'

gravity = True
# Aerodynamic information

# Dynamic simulation
n_tstep=4000
dt=0.001

# Strucutral properties
nnodes = 11
length = 1.0
mass_per_unit_length = 0.15
mass_iner = 1e-4
EA = 1e9
GJ = 1e9
GA = 1e9
EI = 0.15
tip_force = 0.0*np.array([0.0,0.0,1.0,0.0,0.0,0.0])
tip_mass = 10.
theta_ini = 45.0*deg2rad
euler = np.array([0, theta_ini, 0])


# Useless aero information
m = 1
mstar = 1
m_distribution = 'uniform'
airfoil = np.zeros((1,20,2),)
airfoil[0,:,0] = np.linspace(0.,1.,20)

# Create the structure
beam1 = gc.AeroelasticInformation()
node_pos = np.zeros((nnodes,3),)
# node_pos[:, 0] = np.linspace(0.0, length, nnodes)
r = np.linspace(0.0, length, nnodes)
node_pos[:, 0] = -r*np.sin(theta_ini)
node_pos[:, 2] = -r*np.cos(theta_ini)
beam1.StructuralInformation.generate_uniform_sym_beam(node_pos, mass_per_unit_length, mass_iner, EA, GA, GJ, EI, num_node_elem = 3, y_BFoR = 'y_AFoR', num_lumped_mass=1)
beam1.StructuralInformation.body_number = np.zeros((beam1.StructuralInformation.num_elem,), dtype = int)
beam1.StructuralInformation.boundary_conditions[0] = 1
beam1.StructuralInformation.boundary_conditions[-1] = -1
beam1.StructuralInformation.lumped_mass_nodes = np.array([nnodes-1], dtype = int)
beam1.StructuralInformation.lumped_mass = np.ones((1,))*tip_mass
beam1.StructuralInformation.lumped_mass_inertia = np.zeros((1,3,3))
beam1.StructuralInformation.lumped_mass_position = np.zeros((1,3))
beam1.AerodynamicInformation.create_one_uniform_aerodynamics(
                                    beam1.StructuralInformation,
                                    chord = 1.,
                                    twist = 10.,
                                    sweep = 0.,
                                    num_chord_panels = m,
                                    m_distribution = 'uniform',
                                    elastic_axis = 0.25,
                                    num_points_camber = 20,
                                    airfoil = airfoil)

# Simulation details
SimInfo = gc.SimulationInformation()
SimInfo.set_default_values()

SimInfo.solvers['SHARPy']['flow'] = ['BeamLoader',
                        'AerogridLoader',
                        'InitializeMultibody',
                                     # 'Modal',
                        'DynamicCoupled',
                        #              'InitializeMultibody',
                            'Modal',
                                     'SaveData']
SimInfo.solvers['SHARPy']['case'] = case_name
SimInfo.solvers['SHARPy']['route'] = route
SimInfo.set_variable_all_dicts('dt', dt)
SimInfo.define_num_steps(n_tstep)
SimInfo.set_variable_all_dicts('rho', 0.0)
SimInfo.solvers['SteadyVelocityField']['u_inf'] = 1.
SimInfo.solvers['SteadyVelocityField']['u_inf_direction'] = np.array([0., 1., 0.])
SimInfo.set_variable_all_dicts('velocity_field_input', SimInfo.solvers['SteadyVelocityField'])

SimInfo.solvers['BeamLoader']['unsteady'] = 'on'

SimInfo.solvers['AerogridLoader']['unsteady'] = 'on'
SimInfo.solvers['AerogridLoader']['mstar'] = 2

SimInfo.solvers['WriteVariablesTime']['FoR_number'] = np.array([0, 1], dtype = int)
SimInfo.solvers['WriteVariablesTime']['FoR_variables'] = ['mb_quat']
SimInfo.solvers['WriteVariablesTime']['structure_nodes'] = np.array([nnodes-1], dtype = int)
SimInfo.solvers['WriteVariablesTime']['structure_variables'] = ['pos']

SimInfo.solvers['NonLinearDynamicMultibody']['gravity_on'] = gravity
SimInfo.solvers['NonLinearDynamicMultibody']['newmark_damp'] = 0.15

SimInfo.solvers['StepUvlm']['gamma_dot_filtering'] = 0

SimInfo.solvers['DynamicCoupled']['structural_solver'] = 'NonLinearDynamicMultibody'
SimInfo.solvers['DynamicCoupled']['structural_solver_settings'] = SimInfo.solvers['NonLinearDynamicMultibody']
SimInfo.solvers['DynamicCoupled']['aero_solver'] = 'StepUvlm'
SimInfo.solvers['DynamicCoupled']['aero_solver_settings'] = SimInfo.solvers['StepUvlm']
SimInfo.solvers['DynamicCoupled']['postprocessors'] = ['BeamPlot', 'AerogridPlot']
SimInfo.solvers['DynamicCoupled']['postprocessors_settings'] = {
                                                                'BeamPlot': SimInfo.solvers['BeamPlot'],
                                                                'AerogridPlot': SimInfo.solvers['AerogridPlot']}

SimInfo.solvers['Modal'] = {'print_info': True,
                     'use_undamped_modes': True,
                     'NumLambda': 30,
                     'rigid_body_modes': True,
                     'write_modes_vtk': 'on',
                     'print_matrices': 'on',
                     'write_data': 'on',
                     'continuous_eigenvalues': 'off',
                     'dt': dt,
                     'plot_eigenvalues': False}


SimInfo.solvers['StaticCoupled'] = {'print_info': 'on',
                             'structural_solver': 'NonLinearStatic',
                             'structural_solver_settings': {'print_info': 'off',
                                                            'max_iterations': 200,
                                                            'num_load_steps': 1,
                                                            'delta_curved': 1e-5,
                                                            'min_delta': 1e-12,
                                                            'gravity_on': 'on',
                                                            'gravity': 9.81},
                             # 'aero_solver': 'StaticUvlm',
                             # 'aero_solver_settings': {'print_info': 'on',
                             #                          'horseshoe': self.horseshoe,
                             #                          'num_cores': 4,
                             #                          'n_rollup': int(1),
                             #                          'rollup_dt': self.c_root / self.M / self.u_inf,
                             #                          'n_rollup': int(1),
                             #                          'rollup_dt': dt, #self.c_root / self.M / self.u_inf,
                             #                          'rollup_aic_refresh': 1,
                             #                          'rollup_tolerance': 1e-4,
                             #                          'velocity_field_generator': 'SteadyVelocityField',
                             #                          'velocity_field_input': {'u_inf': u_inf,
                             #                                                   'u_inf_direction': [1., 0, 0]},
                             #                          '0rho': rho},
                             'max_iter': 200,
                             'n_load_steps': 1,
                             'tolerance': 1e-12,
                             'relaxation_factor': 0.2}

SimInfo.with_forced_vel = False
SimInfo.with_dynamic_forces = False

# Create the BC file
LC1 = gc.LagrangeConstraint()
LC1.behaviour = 'hinge_FoR'
LC1.body_FoR = 0
LC1.rot_axis_AFoR = np.array([0.0,1.0,0.0])
LC = []
LC.append(LC1)

MB1 = gc.BodyInformation()
MB1.body_number = 0
MB1.FoR_position = np.zeros((6,),)
MB1.FoR_velocity = np.zeros((6,),)
MB1.FoR_acceleration = np.zeros((6,),)
MB1.FoR_movement = 'free'
# MB1.quat = algebra.euler2quat_ag(euler)
MB1.quat = np.array([1.0,0.0,0.0,0.0])
MB = []
MB.append(MB1)

# Write files
gc.clean_test_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
SimInfo.generate_solver_file()
SimInfo.generate_dyn_file(n_tstep)
beam1.generate_h5_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
gc.generate_multibody_file(LC, MB,SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])

print("DONE")
