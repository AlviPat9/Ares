"""
DHC2 - Beaver
================

Flight Simulator environment for DHC2 - beaver aircraft.

Author: Alvaro Marcos Canedo
"""

from Ares.simulators.aircraft import Aircraft

import numpy as np


class DHC_beaver(Aircraft):
    """

    Definition of the model for the DHC2 - Beaver.

    """

    def __init__(self):
        """

        Constructor method

        """
        # Call superclass
        super().__init__()

        # Load aerodynamic data of the aircraft
        self.aero = {}

        # Definition of geometry of the aircraft
        self.c = None
        self.b = None

    def sensors(self, *args):
        pass

    def actuators(self, *args):
        pass

    def aerodynamic_model(self, power_plant, angles, angular_speed, airspeed, delta, accelerations):
        """
        Aerodynamic model of the DHC2 - Beaver.

        @param power_plant:
        @type
        @param angles: Necessary angles for determining the movement of the aircraft.
        @type angles: dict
        @param angular_speed: Angular speed of the aircraft.
        @type angular_speed: dict
        @param airspeed: Airspeed of the aircraft.
        @type airspeed: float
        @param delta: Control surface deflection.
        @type delta: float
        @param accelerations: Accelerations needed for the aerodynamic model of the DHC.
        @type accelerations: dict

        @return: Aerodynamic model of the aircraft. For the DHC2, the aerodynamic model includes the power plant.
        """
        Cx = self.aero['CX_0'] + self.aero['CX_apt'] * power_plant + self.aero['CX_apt2_a'] * angles['alpha'] * power_plant ** 2 + \
            self.aero['CX_a'] * angles['alpha'] + self.aero['CX_a2'] * angles['alpha'] ** 2 + \
            self.aero['CX_a3'] * angles['alpha'] ** 3 + self.aero['CX_q'] * angular_speed['q'] * self.c / airspeed + \
            self.aero['CX_dr'] * delta['r'] + self.aero['CX_df'] * delta['f'] + self.aero['CX_df_a'] * delta['f'] * angles['alpha']

        Cy = self.aero['CY_0'] + self.aero['CY_b'] * angles['beta'] + self.aero['CY_p'] * angular_speed['p'] * self.b / (2 * airspeed) + \
            self.aero['CY_r'] * angular_speed['r'] * self.b / (2 * airspeed) + self.aero['CY_da'] * delta['a'] + \
            self.aero['CY_dr'] * delta['r'] + self.aero['CY_dr_a'] * delta['r'] * angles['alpha'] + \
            self.aero['CY_bp'] * accelerations * self.b / (airspeed * 2)

        Cz = self.aero['CZ_0'] + self.aero['CZ_apt'] * power_plant + self.aero['CZ_a'] * angles['alpha'] + \
            self.aero['CZ_a3'] * angles['alpha'] ** 3 + self.aero['CZ_q'] + angular_speed['q'] * self.c / airspeed + \
            self.aero['CZ_de'] * delta['e'] + self.aero['CZ_de_b2'] * delta['e'] * angles['beta'] ** 2 + \
            self.aero['CZ_df'] * delta['f'] + self.aero['CZ_df_a'] * delta['f'] * angles['alpha']

        Cl = self.aero['Cl_0'] + self.aero['Cl_b'] * angles['beta'] + self.aero['Cl_p'] * angular_speed['p'] * self.b / (2 * airspeed) + \
            self.aero['Cl_r'] * self.b / (2 * airspeed) + self.aero['Cl_da'] * delta['a'] + self.aero['Cl_dr'] * delta['r'] + \
            self.aero['Cl_a2_apt'] * power_plant * angles['alpha'] ** 2 + self.aero['Cl_da_a'] * angles['alpha'] * delta['a']
        Cm = None
        Cn = None

        return np.array([[Cx, Cy, Cz], [Cl, Cm, Cn]])

    def power_plant_model(self, *args):
        pass

    def equations(self, *args):
        pass

    def pid(self, *args):
        pass

    def controller(self, *args):
        pass

    def mass_model(self, *args):
        pass

    def landing_gear(self, *args):
        pass
