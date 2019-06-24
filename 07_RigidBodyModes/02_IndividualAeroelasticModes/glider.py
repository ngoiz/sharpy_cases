# Simple Glider
# N Goizueta April 19


import numpy as np
import os
import sys
sys.path.append('/home/ng213/code/sharpy')
import h5py as h5
import sharpy.utils.geo_utils as geo_utils
import sharpy.utils.algebra as algebra
import configobj
import sharpy.sharpy_main


class Glider(object):

    def __init__(self,
                 M,
                 N,
                 Mstarfact,
                 u_inf,
                 rho=1.225,
                 alpha_deg=5,
                 beta_deg=0,
                 cs_deflection_deg=0,
                 physical_time=10,
                 case_route='./',
                 case_name='glider'):

        # Numerical discretisation
        self.M = M
        self.N = N
        self.Mstarfactor = Mstarfact

        self.n_node_elem = 3
        self.n_elem_wing = 5
        self.n_elem_fuselage = 2
        self.n_elem_tailplane = 2
        self.n_surfaces = 4

        self.case_name = case_name + '_u_inf%.4d' % int(u_inf)
        self.case_route = os.path.abspath(case_route + self.case_name + '/')
        self.config = None

        # Flight conditions
        self.u_inf = u_inf
        self.rho = rho
        self.alpha = alpha_deg * np.pi / 180
        self.beta = beta_deg * np.pi / 180
        self.cs_deflection = cs_deflection_deg * np.pi / 180
        self.thrust = 0

        # Compute number of nodes
        n_node = 0
        self.n_node_wing = self.n_elem_wing * (self.n_node_elem - 1)
        self.n_node_fuselage = self.n_elem_fuselage * self.n_node_elem - 1
        self.n_node_tailplane = self.n_elem_tailplane * (self.n_node_elem - 1)
        n_node += self.n_node_fuselage + 2 * self.n_node_wing + 2 * self.n_node_tailplane
        self.n_node = n_node

        # Compute number of elements
        self.n_elem = 2 * (self.n_elem_wing + self.n_elem_tailplane) + self.n_elem_fuselage

        # Wing geometry
        self.span = 15  # [m]
        self.sweep_LE = 0 * np.pi / 180  # [rad] Leading Edge Sweep
        self.S_ref = 6.11
        self.c_root = self.S_ref / self.span  # [m] Root chord
        self.taper_ratio = 1.0
        self.loc_cg = 0.45  # CG position wrt to LE (from sectional analysis)
        self.ea_offset_root = 0.33
        self.fuselage_length = 4.93
        # Tailplane incidence -5 deg
        self.tailplane_incidence = -5 * np.pi / 180
        self.S_w = 1.14
        AR = self.span/self.c_root
        self.tailplane_span = np.sqrt(AR/self.S_w)
        self.c_w = self.S_w/self.tailplane_span


        # Horseshoe wake
        self.horseshoe = False
        self.wake_type = 2
        self.dt_factor = 1
        self.dt = 1 / self.M / self.u_inf

        # Dynamics
        self.n_tstep = int(physical_time/self.dt)
        self.gust_intensity = 0.1

        # Numerics
        self.tolerance = 1e-12
        self.fsi_tolerance = 1e-10
        self.relaxation_factor = 0.2

        # H5 Variables initialisation as class attributes
        # coordinates
        self.x = np.zeros((n_node,))
        self.y = np.zeros((n_node,))
        self.z = np.zeros((n_node,))
        # beam number
        self.beam_number = np.zeros(self.n_elem, dtype=int)
        # frame of reference delta
        self.frame_of_reference_delta = np.zeros((self.n_elem, self.n_node_elem, 3))
        # connectivity of beams
        self.connectivities = np.zeros((self.n_elem, self.n_node_elem), dtype=int)
        self.app_forces = np.zeros((n_node, 6))

        #stiffness and mass constant along span/fuselage.
        # stiffness
        self.n_stiffness = 1
        self.base_stiffness = np.zeros((self.n_stiffness, 6, 6))
        self.elem_stiffness = np.zeros((self.n_elem,), dtype=int)
        # mass
        self.n_mass = 3
        self.base_mass = np.zeros((self.n_mass, 6, 6))
        self.elem_mass = np.zeros(self.n_elem, dtype=int)

        # structural twist
        self.structural_twist = np.zeros((self.n_elem, self.n_node_elem))
        # boundary conditions
        self.boundary_conditions = np.zeros((n_node,), dtype=int)

        # Lumped mass to tune CG/Inertia
        self.n_lumped_mass = 1

        self.lumped_mass_nodes = np.zeros((self.n_lumped_mass), dtype=int)
        self.lumped_mass = np.zeros(self.n_lumped_mass)
        self.lumped_mass_inertia = np.zeros((self.n_lumped_mass, 3, 3))
        self.lumped_mass_position = np.zeros((self.n_lumped_mass, 3))

        # Aerodynamic properties
        # H5 AERO FILE VARIABLES INITIALISATION
        # airfoil distribution
        self.airfoil_distribution = np.zeros((self.n_elem, self.n_node_elem), dtype=int)
        # surface distribution
        self.surface_distribution = np.zeros((self.n_elem,), dtype=int) - 1
        self.surface_m = np.zeros((self.n_surfaces,), dtype=int)
        self.m_distribution = 'uniform'
        # aerodynamic nodes boolean
        self.aero_nodes = np.zeros((self.n_node,), dtype=bool)
        # aero twist
        self.twist = np.zeros((self.n_elem, self.n_node_elem))
        # chord
        self.chord = np.zeros((self.n_elem, self.n_node_elem))
        # elastic axis
        self.elastic_axis = np.zeros((self.n_elem, self.n_node_elem))

        # control surfaces attributes initialisation
        self.n_control_surfaces = 1
        self.control_surface = np.zeros((self.n_elem, self.n_node_elem), dtype=int) - 1
        self.control_surface_type = np.zeros((self.n_control_surfaces,), dtype=int)
        self.control_surface_deflection = np.zeros((self.n_control_surfaces,))
        self.control_surface_chord = np.zeros((self.n_control_surfaces,), dtype=int)
        self.control_surface_hinge_coord = np.zeros((self.n_control_surfaces,), dtype=float)


    def update_mass_stiffness(self, sigma, lumped_mass_node, lumped_mass_, lumped_mass_inertia_):

        n_elem_fuselage = self.n_elem_fuselage
        n_elem_wing = self.n_elem_wing
        n_node_wing = self.n_node_wing
        n_node_fuselage = self.n_node_fuselage
        c_root = self.c_root
        taper_ratio = self.taper_ratio

        # Local chord to root chord initialisation
        c_bar_temp = np.linspace(c_root, taper_ratio * c_root, n_elem_wing)

        # Structural properties at the wing root section from Richards 2016
        ea = 1e6
        ga = 1e6
        gj = 4.24e5
        eiy = 3.84e5
        eiz = 2.46e7

        mass_fuse_fact = 0.4
        mass_tail_fact = 0.07
        mass_target = 318 - lumped_mass_
        # Sectional mass from Richards 2016
        mu_0 = mass_target / (self.span + mass_fuse_fact*self.fuselage_length + mass_tail_fact*self.tailplane_span)
        # Bending inertia properties from Mardanpour 2013
        j_root = 0.01
        # j_tip = 0.2e-2 * 4.448 / 9.81

        self.cg = mu_0 * (mass_fuse_fact * self.fuselage_length / 2 + mass_tail_fact * self.fuselage_length) / mass_target
        # Number of stiffnesses used
        n_stiffness = self.n_stiffness

        # Initialise the stiffness database
        base_stiffness = self.base_stiffness

        # Wing root section stiffness properties
        base_stiffness[0, :, :] = sigma * np.diag([ea, ga, ga, gj, eiy, eiz])

        # Mass variation along the span
        n_mass = self.n_mass

        # Initialise database
        base_mass = self.base_mass

        # Quadratic variation in the mass per unit span
        # mu_temp = mu_0 * c_bar_temp ** 2
        # j_temp = np.linspace(j_root, j_tip, n_elem_wing)

        base_mass[0, :, :] = np.diag([mu_0, mu_0, mu_0, j_root, j_root, j_root])
        base_mass[1, :, :] = mass_fuse_fact * np.diag([mu_0, mu_0, mu_0, j_root, j_root, j_root])
        base_mass[2, :, :] = mass_tail_fact * np.diag([mu_0, mu_0, mu_0, j_root, j_root, j_root])

        lumped_mass_nodes = self.lumped_mass_nodes
        lumped_mass = self.lumped_mass
        lumped_mass_inertia = self.lumped_mass_inertia
        lumped_mass_position = self.lumped_mass_position

        # Lumped masses nodal position
        lumped_mass_nodes[0] = lumped_mass_node

        # Lumped mass value from Richards 2013
        lumped_mass[0] = lumped_mass_

        # Lumped mass inertia
        lumped_mass_inertia[0, :, :] = lumped_mass_inertia_

        # Define class attributes
        self.lumped_mass = lumped_mass
        self.lumped_mass_nodes = lumped_mass_nodes
        self.lumped_mass_inertia = lumped_mass_inertia
        self.lumped_mass_position = lumped_mass_position

        self.base_stiffness = base_stiffness
        self.base_mass = base_mass

    def update_fem_prop(self):
        """
        Computes the FEM properties prior to analysis such as the connectivity matrix, coordinates, etc

        Returns:

        """
        # Obtain class attributes
        n_node_elem = self.n_node_elem
        n_elem = self.n_elem
        n_elem_wing = self.n_elem_wing
        n_elem_fuselage = self.n_elem_fuselage
        n_elem_tailplane = self.n_elem_tailplane
        n_node = self.n_node
        n_node_wing = self.n_node_wing
        n_node_fuselage = self.n_node_fuselage
        n_node_tailplane = self.n_node_tailplane
        fuselage_length = self.fuselage_length
        span = self.span
        tailplane_span = self.tailplane_span
        sweep_LE = self.sweep_LE

        # mass and stiffness matrices
        stiffness = self.base_stiffness
        mass = self.base_mass
        n_stiffness = stiffness.shape[0]
        n_mass = mass.shape[0]

        # H5 FEM FILE VARIABLES INITIALISATION
        # coordinates
        x = np.zeros((n_node,))
        y = np.zeros((n_node,))
        z = np.zeros((n_node,))
        # twist
        structural_twist = np.zeros((n_elem, n_node_elem))
        # beam number
        beam_number = np.zeros(n_elem, dtype=int)
        # frame of reference delta
        frame_of_reference_delta = np.zeros((n_elem, n_node_elem, 3))
        # connectivity of beams
        conn = np.zeros((n_elem, n_node_elem), dtype=int)
        # stiffness
        stiffness = np.zeros((n_stiffness, 6, 6))
        elem_stiffness = np.zeros((n_elem,), dtype=int)
        # mass
        mass = np.zeros((n_mass, 6, 6))
        elem_mass = np.zeros(n_elem, dtype=int)
        # boundary conditions
        boundary_conditions = np.zeros((n_node,), dtype=int)
        # applied forces
        app_forces = np.zeros((n_node, 6))

        # Elastic axis @TODO move to aerodynamics
        main_ea_root = self.ea_offset_root

        # assemble connectivites
        # worked elements
        we = 0
        # worked nodes
        wn = 0

        # FUSELAGE
        beam_number[we: we+1] = 0

        # Coordinates
        x[wn: wn + n_node_fuselage] = np.linspace(0, fuselage_length, n_node_fuselage)
        y[wn: wn + n_node_fuselage] = 0

        # connectivities
        for ielem in range(n_elem_wing):
            conn[we + ielem, :] = (np.ones(n_node_elem) * (we + ielem) * (n_node_elem - 1) +
                                   [0, 2, 1])
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [0.0, 0.0, 0.0]

        elem_stiffness[we: we+n_elem_fuselage] = 0
        elem_mass[we: we+n_elem_fuselage] = 1

        boundary_conditions[0] = 1
        we += n_elem_fuselage
        wn += n_node_fuselage

        # RIGHT WING
        beam_number[we:we + 1] = 1
        # coordinates
        x[wn:wn + n_node_wing] = 0
        y[wn:wn + n_node_wing] = np.linspace(0, span / 2, n_node_wing + 1)[1:]

         # connectivities
        for ielem in range(n_elem_wing):
            conn[we + ielem, :] = (np.ones(n_node_elem) * (we + ielem) * (n_node_elem - 1) +
                                   [0, 2, 1])
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]

        conn[we, 0] = 0
        # element stiffness
        elem_stiffness[we: we+ielem] = 0
        elem_mass[we: we+ielem] = 0

        # boundary conditions
        boundary_conditions[wn + n_node_wing - 1] = -1

        # updated worked nodes and elements
        we += n_elem_wing
        wn += n_node_wing

        # LEFT WING
        beam_number[we:we + n_elem_wing] = 2
        # y coordinate (positive right)
        y[wn:wn + n_node_wing] = np.linspace(0,
                                             -span / 2,
                                             n_node_wing+1)[1:]
        x[wn:wn + n_node_wing] = 0

        # connectivities
        for ielem in range(n_elem_wing):
            conn[we + ielem, :] = (np.ones(n_node_elem) * (we + ielem) * (n_node_elem - 1) +
                                   [0, 2, 1])
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [+1.0, 0.0, 0.0]

            # elem_mass[we + ielem] = 0
            # elem_stiffness[we + ielem] = 0
        conn[we, 0] = 0
        # element stiffness and mass
        elem_stiffness[we:we+n_elem_wing] = 0
        elem_mass[we:we+n_elem_wing] = 0

        # boundary conditions of free end
        boundary_conditions[wn + n_node_wing - 1] = -1

        # update worked elements and nodes
        we += n_elem_wing
        wn += n_node_wing

        # RIGHT TAILPLANE
        beam_number[we:we + n_elem_tailplane] = 3
        # coordinates
        y[wn:wn + n_node_tailplane] = np.linspace(0,
                                                     tailplane_span / 2,
                                                     n_node_tailplane + 1)[1:]
        x[wn:wn + n_node_tailplane] = fuselage_length

        # connectivities
        for ielem in range(n_elem_tailplane):
            conn[we + ielem, :] = (np.ones(n_node_elem) * (we + ielem) * (n_node_elem - 1) +
                                   [0, 2, 1])
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [-1.0, 0.0, 0.0]

        conn[we, 0] = n_node_fuselage - 1

        # element stiffness and mass
        elem_stiffness[we:we + n_elem_tailplane] = 0
        elem_mass[we:we + n_elem_tailplane] = 2

        boundary_conditions[wn + n_node_tailplane - 1] = -1
        # update worked elements and nodes
        we += n_elem_tailplane
        wn += n_node_tailplane

        # LEFT TAILPLANE
        beam_number[we:we + n_elem_tailplane] = 4
        # coordinates
        y[wn:wn + n_node_tailplane] = np.linspace(0,
                                                     -tailplane_span / 2,
                                                     n_node_tailplane + 1)[1:]
        x[wn:wn + n_node_tailplane] = fuselage_length

        # connectivities
        for ielem in range(n_elem_tailplane):
            conn[we + ielem, :] = (np.ones(n_node_elem) * (we + ielem) * (n_node_elem - 1) +
                                   [0, 2, 1])
            for inode in range(n_node_elem):
                frame_of_reference_delta[we + ielem, inode, :] = [+1.0, 0.0, 0.0]

        conn[we, 0] = n_node_fuselage - 1

        # element stiffness and mass
        elem_stiffness[we:we + n_elem_tailplane] = 0
        elem_mass[we:we + n_elem_tailplane] = 2

        # boundary conditions at the free end
        boundary_conditions[wn + n_node_tailplane - 1] = -1

        # update worked elements and nodes
        we += n_elem_wing
        wn += n_node_wing

        # set attributes
        self.x = x
        self.y = y
        self.z = z
        self.connectivities = conn
        self.elem_stiffness = elem_stiffness
        self.elem_mass = elem_mass
        self.frame_of_reference_delta = frame_of_reference_delta
        self.boundary_conditions = boundary_conditions
        self.beam_number = beam_number

    def generate_fem_file(self):
        """
        Generates the ``.fem.h5`` folder containing the structural information of the problem

        The file is written to ``self.case_route / self.case_name .fem.h5``

        """

        if not os.path.isdir(self.case_route):
            os.makedirs(self.case_route)

        with h5.File(self.case_route + '/' + self.case_name + '.fem.h5', 'a') as h5file:
            coordinates = h5file.create_dataset('coordinates',
                                                data=np.column_stack((self.x, self.y, self.z)))
            connectivities = h5file.create_dataset('connectivities', data=self.connectivities)
            num_nodes_elem_handle = h5file.create_dataset(
                'num_node_elem', data=self.n_node_elem)
            num_nodes_handle = h5file.create_dataset(
                'num_node', data=self.n_node)
            num_elem_handle = h5file.create_dataset(
                'num_elem', data=self.n_elem)
            stiffness_db_handle = h5file.create_dataset(
                'stiffness_db', data=self.base_stiffness)
            stiffness_handle = h5file.create_dataset(
                'elem_stiffness', data=self.elem_stiffness)
            mass_db_handle = h5file.create_dataset(
                'mass_db', data=self.base_mass)
            mass_handle = h5file.create_dataset(
                'elem_mass', data=self.elem_mass)
            frame_of_reference_delta_handle = h5file.create_dataset(
                'frame_of_reference_delta', data=self.frame_of_reference_delta)
            structural_twist_handle = h5file.create_dataset(
                'structural_twist', data=self.structural_twist)
            bocos_handle = h5file.create_dataset(
                'boundary_conditions', data=self.boundary_conditions)
            beam_handle = h5file.create_dataset(
                'beam_number', data=self.beam_number)
            app_forces_handle = h5file.create_dataset(
                'app_forces', data=self.app_forces)
            lumped_mass_nodes_handle = h5file.create_dataset(
                'lumped_mass_nodes', data=self.lumped_mass_nodes)
            lumped_mass_handle = h5file.create_dataset(
                'lumped_mass', data=self.lumped_mass)
            lumped_mass_inertia_handle = h5file.create_dataset(
                'lumped_mass_inertia', data=self.lumped_mass_inertia)
            lumped_mass_position_handle = h5file.create_dataset(
                'lumped_mass_position', data=self.lumped_mass_position)


    def update_aero_properties(self):
        """
        Updates the aerodynamic properties of the horten wing

        """

        # Retrieve attributes
        n_elem = self.n_elem
        n_node_elem = self.n_node_elem
        n_node_wing = self.n_node_wing
        n_node_fuselage = self.n_node_fuselage
        n_elem_fuselage = self.n_elem_fuselage
        n_elem_wing = self.n_elem_wing
        n_elem_tailplane = self.n_elem_tailplane
        n_node_tailplane = self.n_node_tailplane
        c_root = self.c_root
        c_w = self.c_w
        tailplane_incidence = self.tailplane_incidence
        taper_ratio = self.taper_ratio
        # washout_root = self.washout_root
        # washout_tip = self.washout_tip
        n_control_surfaces = self.n_control_surfaces
        cs_deflection = self.cs_deflection
        m = self.M
        # main_ea_root = self.main_ea_root
        # main_ea_tip = self.main_ea_tip
        main_ea = self.ea_offset_root
        airfoil_distribution = self.airfoil_distribution
        chord = self.chord
        surface_distribution = self.surface_distribution
        surface_m = self.surface_m
        aero_nodes = self.aero_nodes
        elastic_axis = self.elastic_axis
        twist = self.twist

        control_surface = self.control_surface
        control_surface_type = self.control_surface_type
        control_surface_deflection = self.control_surface_deflection
        control_surface_chord = self.control_surface_chord
        control_surface_hinge_coord = self.control_surface_hinge_coord

        self.dt = 1 / self.M / self.u_inf

        # control surface type: 0 = static
        # control surface type: 1 = dynamic
        control_surface_type[0] = 0
        control_surface_deflection[0] = cs_deflection
        control_surface_chord[0] = 2  # m
        control_surface_hinge_coord[0] = 0.25

        # FUSELAGE (Surface 0, Beam 0)
        we = 0
        wn = 0

        i_surf = -1
        airfoil_distribution[we:we + n_elem_fuselage] = 0
        surface_distribution[we:we + n_elem_fuselage] = i_surf
        surface_m[i_surf] = m

        aero_nodes[wn:wn + n_node_fuselage] = True

        temp_chord = 0
        temp_washout = 0

        # # apply chord and elastic axis at each node
        node_counter = 0
        for ielem in range(we, we + n_elem_fuselage):
            for i_local_node in range(n_node_elem):
                if not i_local_node == 0:
                    node_counter += 1
                chord[ielem, i_local_node] = temp_chord
        #         elastic_axis[ielem, i_local_node] = main_ea_root
        #         twist[ielem, i_local_node] = -temp_washout

        we += n_elem_fuselage
        wn += n_node_fuselage

        # RIGHT WING
        # surface_id
        i_surf = 0
        airfoil_distribution[we:we + n_elem_wing, :] = 0
        surface_distribution[we:we + n_elem_wing] = i_surf
        surface_m[i_surf] = m

        # specify aerodynamic characteristics of wing nodes
        aero_nodes[wn:wn + n_node_wing] = True

        # linear taper initialisation
        # temp_chord = np.linspace(c_root, taper_ratio * c_root, n_node_wing + 1)

        # linear wash out initialisation
        # temp_washout = np.linspace(washout_root, washout_tip, n_node_wing + 1)

        # elastic axis variation
        # temp_ea = np.linspace(main_ea_root, main_ea_tip, n_node_wing + 1)

        # apply chord and elastic axis at each node
        node_counter = 0
        for ielem in range(we, we + n_elem_wing):
            for i_local_node in range(n_node_elem):
                if not i_local_node == 0:
                    node_counter += 1
                chord[ielem, i_local_node] = c_root
                elastic_axis[ielem, i_local_node] = main_ea
                twist[ielem, i_local_node] = 0
                # if ielem >= round(((we + n_elem_wing) / 2)):
                #     control_surface[ielem, i_local_node] = 0

        # update working element and node
        we += n_elem_wing
        wn += n_node_wing

        # # LEFT FUSELAGE (Surface 2, Beam 2)
        # i_surf = 2
        # airfoil_distribution[we:we + n_elem_fuselage] = 0
        # surface_distribution[we:we + n_elem_fuselage] = i_surf
        # surface_m[i_surf] = m
        #
        # aero_nodes[wn:wn + n_node_fuselage] = True
        #
        # temp_chord = c_root
        # temp_washout = 0
        #
        # # apply chord and elastic axis at each node
        # node_counter = 0
        # for ielem in range(we, we + n_elem_fuselage):
        #     for i_local_node in range(n_node_elem):
        #         if not i_local_node == 0:
        #             node_counter += 1
        #         chord[ielem, i_local_node] = temp_chord
        #         elastic_axis[ielem, i_local_node] = main_ea_root
        #         twist[ielem, i_local_node] = -temp_washout
        #
        # we += n_elem_fuselage
        # wn += n_node_fuselage

        # LEFT WING (Surface 3, Beam 3)
        i_surf = 1
        airfoil_distribution[we:we + n_elem_wing, :] = 0
        surface_distribution[we: we + n_elem_wing] = i_surf
        surface_m[i_surf] = m

        # linear taper initialisation
        # temp_chord = np.linspace(c_root, taper_ratio * c_root, n_node_wing + 1)

        # linear wash out initialisation
        # temp_washout = np.linspace(washout_root, washout_tip, n_node_wing + 1)

        # specify aerodynamic characterisics of wing nodes
        aero_nodes[wn:wn + n_node_wing] = True

        # linear taper initialisation
        # apply chord and elastic axis at each node
        node_counter = 0
        for ielem in range(we, we + n_elem_wing):
            for i_local_node in range(n_node_elem):
                if not i_local_node == 0:
                    node_counter += 1
                chord[ielem, i_local_node] = c_root
                elastic_axis[ielem, i_local_node] = main_ea
                twist[ielem, i_local_node] = 0
                # if ielem >= round((we + n_elem_wing / 2)):
                #     control_surface[ielem, i_local_node] = 0

        # update working element and node
        we += n_elem_wing
        wn += n_node_wing

        # RIGHT TAILPLANE (Surface 3, Beam 3)
        i_surf = 2
        airfoil_distribution[we:we + n_elem_tailplane, :] = 0
        surface_distribution[we: we + n_elem_tailplane] = i_surf
        surface_m[i_surf] = m

        # linear taper initialisation
        # temp_chord = np.linspace(c_root, taper_ratio * c_root, n_node_wing + 1)

        # linear wash out initialisation
        # temp_washout = np.linspace(washout_root, washout_tip, n_node_wing + 1)

        # specify aerodynamic characterisics of wing nodes
        aero_nodes[wn:wn + n_node_wing] = True

        # linear taper initialisation
        # apply chord and elastic axis at each node
        node_counter = 0
        for ielem in range(we, we + n_elem_tailplane):
            for i_local_node in range(n_node_elem):
                if not i_local_node == 0:
                    node_counter += 1
                chord[ielem, i_local_node] = c_w
                elastic_axis[ielem, i_local_node] = main_ea
                twist[ielem, i_local_node] = -tailplane_incidence
                # if ielem >= round((we + n_elem_tailplane / 2)):
                control_surface[ielem, i_local_node] = 0

        # update working element and node
        we += n_elem_tailplane
        wn += n_node_tailplane

        # LEFT TAILPLANE (Surface 3, Beam 3)
        i_surf = 3
        airfoil_distribution[we:we + n_elem_tailplane, :] = 0
        surface_distribution[we: we + n_elem_tailplane] = i_surf
        surface_m[i_surf] = m

        # linear taper initialisation
        # temp_chord = np.linspace(c_root, taper_ratio * c_root, n_node_wing + 1)

        # linear wash out initialisation
        # temp_washout = np.linspace(washout_root, washout_tip, n_node_wing + 1)

        # specify aerodynamic characterisics of wing nodes
        aero_nodes[wn:wn + n_node_wing] = True

        # linear taper initialisation
        # apply chord and elastic axis at each node
        node_counter = 0
        for ielem in range(we, we + n_elem_tailplane):
            for i_local_node in range(n_node_elem):
                if not i_local_node == 0:
                    node_counter += 1
                chord[ielem, i_local_node] = c_w
                elastic_axis[ielem, i_local_node] = main_ea
                twist[ielem, i_local_node] = -tailplane_incidence
                # if ielem >= round((we + n_elem_tailplane / 2)):
                control_surface[ielem, i_local_node] = 0

        # update working element and node
        we += n_elem_tailplane
        wn += n_node_tailplane

        # end node is the middle node
        mid_chord = np.array(chord[:, 1], copy=True)
        chord[:, 1] = chord[:, 2]
        chord[:, 2] = mid_chord

        mid_ea = np.array(elastic_axis[:, 1], copy=True)
        elastic_axis[:, 1] = elastic_axis[:, 2]
        elastic_axis[:, 2] = mid_ea

        # Update aerodynamic attributes of class
        self.chord = chord
        self.twist = twist
        self.aero_nodes = aero_nodes
        self.elastic_axis = elastic_axis
        self.control_surface = control_surface

    def generate_aero_file(self, route=None, case_name=None):
        """
        Generates the ``.aero.h5`` file with the aerodynamic properties of the wing

        Args:
            route (str): route to write case file. If None is specified the default will be used
            case_name (str): name of file. If None is specified the default will be used

        """

        if not route:
            route = self.case_route

        if not case_name:
            case_name = self.case_name

        if not os.path.isdir(self.case_route):
            os.makedirs(self.case_route)

        chord = self.chord
        twist = self.twist
        airfoil_distribution = self.airfoil_distribution
        surface_distribution = self.surface_distribution
        surface_m = self.surface_m
        m_distribution = self.m_distribution
        aero_nodes = self.aero_nodes
        elastic_axis = self.elastic_axis
        control_surface = self.control_surface
        control_surface_deflection = self.control_surface_deflection
        control_surface_chord = self.control_surface_chord
        control_surface_hinge_coord = self.control_surface_hinge_coord
        control_surface_type = self.control_surface_type

        control_surface_deflection[0] = self.cs_deflection

        with h5.File(route + '/' + case_name + '.aero.h5', 'a') as h5file:
            airfoils_group = h5file.create_group('airfoils')
            # add one airfoil
            naca_airfoil_main = airfoils_group.create_dataset('0', data=np.column_stack(
                geo_utils.generate_naca_camber(P=0, M=0)))
            naca_airfoil_tail = airfoils_group.create_dataset('1', data=np.column_stack(
                geo_utils.generate_naca_camber(P=0, M=0)))
            naca_airfoil_fin = airfoils_group.create_dataset('2', data=np.column_stack(
                geo_utils.generate_naca_camber(P=0, M=0)))

            # chord
            chord_input = h5file.create_dataset('chord', data=chord)
            dim_attr = chord_input.attrs['units'] = 'm'

            # twist
            twist_input = h5file.create_dataset('twist', data=twist)
            dim_attr = twist_input.attrs['units'] = 'rad'

            # airfoil distribution
            airfoil_distribution_input = h5file.create_dataset('airfoil_distribution', data=airfoil_distribution)

            surface_distribution_input = h5file.create_dataset('surface_distribution', data=surface_distribution)
            surface_m_input = h5file.create_dataset('surface_m', data=surface_m)
            m_distribution_input = h5file.create_dataset('m_distribution',
                                                         data=m_distribution.encode('ascii', 'ignore'))

            aero_node_input = h5file.create_dataset('aero_node', data=aero_nodes)
            elastic_axis_input = h5file.create_dataset('elastic_axis', data=elastic_axis)

            control_surface_input = h5file.create_dataset('control_surface', data=control_surface)
            control_surface_deflection_input = h5file.create_dataset('control_surface_deflection',
                                                                     data=control_surface_deflection)
            control_surface_chord_input = h5file.create_dataset('control_surface_chord', data=control_surface_chord)
            control_surface_hinge_coord_input = h5file.create_dataset('control_surface_hinge_coord',
                                                                      data=control_surface_hinge_coord)
            control_surface_types_input = h5file.create_dataset('control_surface_type', data=control_surface_type)

    def clean_test_files(self):
        """
        Clears previously generated files
        """

        case_name = self.case_name
        route = self.case_route

        # FEM
        fem_file_name = route + '/' + case_name + '.fem.h5'
        if os.path.isfile(fem_file_name):
            os.remove(fem_file_name)

        # Dynamics File
        dyn_file_name = route + '/' + case_name + '.dyn.h5'
        if os.path.isfile(dyn_file_name):
            os.remove(dyn_file_name)

        # Aerodynamics File
        aero_file_name = route + '/' + case_name + '.aero.h5'
        if os.path.isfile(aero_file_name):
            os.remove(aero_file_name)

        # Solver file
        solver_file_name = route + '/' + case_name + '.solver.txt'
        if os.path.isfile(solver_file_name):
            os.remove(solver_file_name)

        # Flight conditions file
        flightcon_file_name = route + '/' + case_name + '.flightcon.txt'
        if os.path.isfile(flightcon_file_name):
            os.remove(flightcon_file_name)

        if os.path.isdir(route):
            os.system('rm -r %s' %route)


    def set_default_config_dict(self, route=None, case_name=None):
        """
        Generates default solver configuration file
        Returns:

        """

        if not route:
            route = self.case_route

        if not case_name:
            case_name = self.case_name

        u_inf = self.u_inf
        rho = self.rho
        dt = self.dt
        tolerance = self.tolerance
        alpha = self.alpha
        beta = self.beta
        thrust = self.thrust
        # thrust_nodes = self.thrust_nodes
        cs_deflection = self.cs_deflection
        fsi_tolerance = self.fsi_tolerance
        n_tstep = self.n_tstep
        gust_intensity = self.gust_intensity
        relaxation_factor = self.relaxation_factor

        file_name = route + '/' + case_name + '.solver.txt'

        settings = dict()
        settings['SHARPy'] = {'case': case_name,
                              'route': route,
                              'flow': ['BeamLoader',
                                       'AerogridLoader',
                                       # 'NonLinearStatic',
                                       'StaticCoupled',
                                       # 'StaticTrim',
                                       'AerogridPlot',
                                       'BeamPlot',
                                       # 'DynamicCoupled',
                                       'Modal'],
                                       # 'SaveData'],
                              'write_screen': 'on',
                              'write_log': 'on',
                              'log_folder': route + '/output/',
                              'log_file': case_name + '.log'}

        settings['BeamLoader'] = {'unsteady': 'off',
                                  'orientation': algebra.euler2quat(np.array([0.0,
                                                                              self.alpha,
                                                                              self.beta]))}

        settings['StaticUvlm'] = {'print_info': 'on',
                                  'horseshoe': self.horseshoe,
                                  'num_cores': 4,
                                  'n_rollup': 1,
                                  'rollup_dt': dt,
                                  'rollup_aic_refresh': 1,
                                  'rollup_tolerance': 1e-4,
                                  'velocity_field_generator': 'SteadyVelocityField',
                                  'velocity_field_input': {'u_inf': u_inf,
                                                           'u_inf_direction': [1., 0, 0]},
                                  'rho': rho}

        settings['StaticCoupled'] = {'print_info': 'on',
                                     'structural_solver': 'NonLinearStatic',
                                     'structural_solver_settings': {'print_info': 'off',
                                                                    'max_iterations': 200,
                                                                    'num_load_steps': 1,
                                                                    'delta_curved': 1e-5,
                                                                    'min_delta': tolerance,
                                                                    'gravity_on': 'on',
                                                                    'gravity': 9.81},
                                     'aero_solver': 'StaticUvlm',
                                     'aero_solver_settings': {'print_info': 'on',
                                                              'horseshoe': self.horseshoe,
                                                              'num_cores': 4,
                                                              'n_rollup': int(1),
                                                              'rollup_dt': self.c_root / self.M / self.u_inf,
                                                              'n_rollup': int(1),
                                                              'rollup_dt': dt, #self.c_root / self.M / self.u_inf,
                                                              'rollup_aic_refresh': 1,
                                                              'rollup_tolerance': 1e-4,
                                                              'velocity_field_generator': 'SteadyVelocityField',
                                                              'velocity_field_input': {'u_inf': u_inf,
                                                                                       'u_inf_direction': [1., 0, 0]},
                                                              'rho': rho},
                                     'max_iter': 200,
                                     'n_load_steps': 1,
                                     'tolerance': tolerance,
                                     'relaxation_factor': 0.2}

        if self.horseshoe is True:
            settings['AerogridLoader'] = {'unsteady': 'off',
                                          'aligned_grid': 'on',
                                          'mstar': 1,
                                          'freestream_dir': ['1', '0', '0'],
                                          'control_surface_deflection': ['']}
        else:
            settings['AerogridLoader'] = {'unsteady': 'off',
                                          'aligned_grid': 'on',
                                          'mstar': int(self.M * self.Mstarfactor),
                                          'freestream_dir': ['1', '0', '0'],
                                          'control_surface_deflection': ['']}

        settings['NonLinearStatic'] = {'print_info': 'off',
                                       'max_iterations': 150,
                                       'num_load_steps': 1,
                                       'delta_curved': 1e-8,
                                       'min_delta': tolerance,
                                       'gravity_on': True,
                                       'gravity': 9.81}

        settings['StaticTrim'] = {'solver': 'StaticCoupled',
                                  'solver_settings': settings['StaticCoupled'],
                                  # 'thrust_nodes': thrust_nodes,
                                  'initial_alpha': alpha,
                                  'initial_deflection': cs_deflection,
                                  'initial_thrust': 0.,
                                  'max_iter': 200,
                                  'fz_tolerance': 1e-2,
                                  'fx_tolerance': 1e-2,
                                  'm_tolerance': 1e-2}

        settings['Trim'] = {'solver': 'StaticCoupled',
                            'solver_settings': settings['StaticCoupled'],
                            'initial_alpha': alpha,
                            'initial_beta': beta,
                            'cs_indices': [0],
                            'initial_cs_deflection': [cs_deflection],
                            # 'thrust_nodes': thrust_nodes,
                            'initial_thrust': [thrust, thrust]}

        settings['NonLinearDynamicCoupledStep'] = {'print_info': 'off',
                                                   'initial_velocity_direction': [-1., 0., 0.],
                                                   'max_iterations': 950,
                                                   'delta_curved': 1e-6,
                                                   'min_delta': tolerance,
                                                   'newmark_damp': 5e-3,
                                                   'gravity_on': True,
                                                   'gravity': 9.81,
                                                   'num_steps': n_tstep,
                                                   'dt': dt,
                                                   'initial_velocity': u_inf * 0}

        settings['NonLinearDynamicPrescribedStep'] = {'print_info': 'off',
                                                      'initial_velocity_direction': [-1., 0., 0.],
                                                      'max_iterations': 950,
                                                      'delta_curved': 1e-6,
                                                      'min_delta': self.tolerance,
                                                      'newmark_damp': 5e-3,
                                                      'gravity_on': True,
                                                      'gravity': 9.81,
                                                      'num_steps': self.n_tstep,
                                                      'dt': self.dt}

        settings['StepLinearUVLM'] = {'dt': self.dt,
                                      'integr_order': 1,
                                      'remove_predictor': True,
                                      'use_sparse': True,
                                      'velocity_field_generator': 'GustVelocityField',
                                      'velocity_field_input': {'u_inf': u_inf,
                                                               'u_inf_direction': [1., 0., 0.],
                                                               'gust_shape': '1-cos',
                                                               'gust_length': 1.,
                                                               'gust_intensity': self.gust_intensity * u_inf,
                                                               'offset': 30.,
                                                               'span': self.span}}

        settings['StepUvlm'] = {'print_info': 'off',
                                'horseshoe': self.horseshoe,
                                'num_cores': 4,
                                'n_rollup': 100,
                                'convection_scheme': self.wake_type,
                                'rollup_dt': dt,
                                'rollup_aic_refresh': 1,
                                'rollup_tolerance': 1e-4,
                                'velocity_field_generator': 'GustVelocityField',
                                'velocity_field_input': {'u_inf': u_inf * 1,
                                                         'u_inf_direction': [1., 0, 0],
                                                         'gust_shape': '1-cos',
                                                         'gust_length': 1.,
                                                         'gust_intensity': gust_intensity * u_inf,
                                                         'offset': 30.0,
                                                         'span': self.span},
                                # 'velocity_field_generator': 'SteadyVelocityField',
                                # 'velocity_field_input': {'u_inf': u_inf*1,
                                #                             'u_inf_direction': [1., 0., 0.]},
                                'rho': rho,
                                'n_time_steps': n_tstep,
                                'dt': dt,
                                'gamma_dot_filtering': 3}

        settings['DynamicCoupled'] = {'print_info': 'on',
                                      'structural_substeps': 1,
                                      'dynamic_relaxation': 'on',
                                      'clean_up_previous_solution': 'on',
                                      'structural_solver': 'NonLinearDynamicCoupledStep',
                                      'structural_solver_settings': settings['NonLinearDynamicCoupledStep'],
                                      'aero_solver': 'StepUvlm',
                                      'aero_solver_settings': settings['StepUvlm'],
                                      'fsi_substeps': 200,
                                      'fsi_tolerance': fsi_tolerance,
                                      'relaxation_factor': relaxation_factor,
                                      'minimum_steps': 1,
                                      'relaxation_steps': 150,
                                      'final_relaxation_factor': 0.0,
                                      'n_time_steps': n_tstep,
                                      'dt': dt,
                                      'include_unsteady_force_contribution': 'off',
                                      'postprocessors': ['BeamLoads', 'StallCheck', 'BeamPlot', 'AerogridPlot'],
                                      'postprocessors_settings': {'BeamLoads': {'folder': route + '/output/',
                                                                                'csv_output': 'off'},
                                                                  'StallCheck': {'output_degrees': True,
                                                                                 'stall_angles': {
                                                                                     '0': [-12 * np.pi / 180,
                                                                                           6 * np.pi / 180],
                                                                                     '1': [-12 * np.pi / 180,
                                                                                           6 * np.pi / 180],
                                                                                     '2': [-12 * np.pi / 180,
                                                                                           6 * np.pi / 180]}},
                                                                  'BeamPlot': {'folder': route + '/output/',
                                                                               'include_rbm': 'on',
                                                                               'include_applied_forces': 'on'},
                                                                  'AerogridPlot': {
                                                                      'u_inf': u_inf,
                                                                      'folder': route + '/output/',
                                                                      'include_rbm': 'on',
                                                                      'include_applied_forces': 'on',
                                                                      'minus_m_star': 0},
                                                                  #              'WriteVariablesTime': {
                                                                  #              #     'delimeter': ',',
                                                                  #              #     'structure_nodes': [0],
                                                                  #              #     'structure_variables': ['Z']
                                                                  #                 # settings['WriteVariablesTime'] = {'delimiter': ' ',
                                                                  #     'FoR_variables': ['GFoR_pos', 'GFoR_vel', 'GFoR_acc'],
                                                                  # 'FoR_number': [],
                                                                  # 'structure_variables': ['AFoR_steady_forces', 'AFoR_unsteady_forces','AFoR_position'],
                                                                  # 'structure_nodes': [0,-1],
                                                                  # 'aero_panels_variables': ['gamma', 'gamma_dot'],
                                                                  # 'aero_panels_isurf': [0,1,2],
                                                                  # 'aero_panels_im': [1,1,1],
                                                                  # 'aero_panels_in': [-2,-2,-2],
                                                                  # 'aero_nodes_variables': ['GFoR_steady_force', 'GFoR_unsteady_force'],
                                                                  # 'aero_nodes_isurf': [0,1,2],
                                                                  # 'aero_nodes_im': [1,1,1],
                                                                  # 'aero_nodes_in': [-2,-2,-2]
                                                                  #              }}}
                                                                  }}

        settings['Modal'] = {'print_info': True,
                             'use_undamped_modes': True,
                             'NumLambda': 30,
                             'rigid_body_modes': True,
                             'write_modes_vtk': 'on',
                             'print_matrices': 'on',
                             'write_data': 'on',
                             'continuous_eigenvalues': 'off',
                             'dt': dt,
                             'plot_eigenvalues': False}

        settings['AerogridPlot'] = {'folder': route + '/output/',
                                    'include_rbm': 'off',
                                    'include_applied_forces': 'on',
                                    'minus_m_star': 0,
                                    'u_inf': u_inf
                                    }
        settings['AeroForcesCalculator'] = {'folder': route + '/output/',
                                            'write_text_file': 'off',
                                            'text_file_name': case_name + '_aeroforces.csv',
                                            'screen_output': 'on',
                                            'unsteady': 'off'
                                            }
        settings['BeamPlot'] = {'folder': route + '/output/',
                                'include_rbm': 'on',
                                'include_applied_forces': 'on'}

        settings['BeamLoads'] = {'folder': route + '/output/',
                                 'csv_output': 'off'}

        settings['SaveData'] = {'folder': route + '/output'}

        config = configobj.ConfigObj()
        config.filename = file_name
        for k, v in settings.items():
            config[k] = v
        config.write()
        self.settings = settings

        self.config = config

if __name__=='__main__':

    glider = Glider(4,
                    5,
                    10,
                    u_inf=28,
                    rho=1.02,
                    alpha_deg=0,
                    beta_deg=0,
                    physical_time=200,
                    cs_deflection_deg=0)

    glider.clean_test_files()
    glider.update_mass_stiffness(sigma=1, lumped_mass_node=4, lumped_mass_=48*0,
                                 lumped_mass_inertia_= 0*np.diag([1,1,1]))
    glider.update_fem_prop()
    glider.generate_fem_file()
    glider.update_aero_properties()
    glider.generate_aero_file()
    glider.set_default_config_dict()

    #Run
    data = sharpy.sharpy_main.main(['', glider.case_route + '/' + glider.case_name + '.solver.txt'])
    M = data.structure.timestep_info[-1].modal['M']
    m = M[-10, -10]
    Iyy = M[-6, -6]
    cg = glider.cg
    print('Done')
