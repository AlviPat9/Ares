"""
DHC2 - Beaver
================

Flight Simulator environment for DHC2 - beaver aircraft.

Author: Alvaro Marcos Canedo
"""

from Ares.simulators.aircraft import Aircraft
from Ares.aicraft_simulations.atmospheric_model import AtmosphericModel
from Ares.integrators.rk4 import RK4

import numpy as np
from scipy.signal import dlti


class DHC_beaver(Aircraft):
    """

    Definition of the model for the DHC2 - Beaver.

    """

    def __init__(self, initial_state: np.ndarray, step_size=0.3, final_time=1000):
        """

        Constructor method

        @param initial_state: Initial state of the model.
        @type initial_state: np.ndarray
        @param step_size: Step size for the calculation.
        @type step_size: float
        @param final_time: Final time for the simulation.

        """
        # Call superclass
        super().__init__()

        # Call integrator
        self.integration = RK4(system=self.equations(), step_size=step_size, final_time=final_time,
                               initial_state=initial_state)

        # Load aerodynamic data of the aircraft
        self.aero = {}

        # Instantiate atmospheric model
        self.atmosphere = AtmosphericModel()

        # Definition of the mass model
        self.mass = self.mass_model()

        # Definition of sensors for the aircraft
        self.sensors = self._sensors(step_size)

        # Definition of actuators for the aircraft
        self.actuators = self._actuators(step_size)

        # Definition of geometry of the aircraft
        self.c = 1.5875  # Mean aerodynamic Chord (MAC) [m]
        self.b = 14.63  # Wing span [m]
        self.S = 23.23  # Wing area [m^2]

    def _sensors(self, step_size):
        """

        Definition of the transfer functions of the actuators.

        @param step_size: Step size for the integration model.
        @type step_size: float

        @return: Transfer functions associated to the sensors of the aircraft.
        @rtype: dict

        """

        # 2nd order Pade approximation

        # Time delay of 0.1s
        anemometer = dlti([0.1 ** 2 / 12.0, -0.05, 1], [0.1 ** 2 / 12.0, 0.05, 1], dt=step_size)

        # Time delay of 0.06s
        inertial = dlti([0.0003, -0.03, 1], [0.0003, 0.03, 1], dt=step_size)

        return {'anemometer': anemometer, 'inertial': inertial}

    def _actuators(self, step_size):
        """

        Definition of the actuators model for the aircraft.

        @param step_size: Step size for the integration model.
        @type step_size: float

        @return: Transfer functions associated to the different actuators of the aircraft.
        @rtype: dict

        """

        # Low-pass filter model

        # Time constant of 0.1s
        aerodynamic = dlti([1], [0.1, 1], dt=step_size)

        # Time constant of 5s
        throttle_lever = dlti([1], [5, 1], dt=step_size)

        return {'aerodynamic': aerodynamic, 'Throttle_lever': throttle_lever}

    def aerodynamic_model(self, n, angles, angular_speed, airspeed, delta, accelerations):
        """
        Aerodynamic model of the DHC2 - Beaver.

        @param n: RPM of the engine.
        @type n: float
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
        @rtype: np.ndarray

        """

        power_plant = self.power_plant_model(n, airspeed)

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

        Cm = self.aero['Cm_0'] + self.aero['Cm_apt'] * power_plant + self.aero['Cm_a'] * angles['alpha'] + \
            self.aero['Cm_a2'] * angles['alpha'] ** 2 + self.aero['Cm_q'] * angular_speed['q'] * self.c / airspeed + \
            self.aero['Cm_de'] * delta['e'] + self.aero['Cm_b2'] * angles['beta'] ** 2 + \
            self.aero['Cm_r'] * angular_speed['r'] * self.b / (2 * airspeed) + self.aero['Cm_df'] * delta['f']

        Cn = self.aero['Cn_0'] + self.aero['Cn_b'] * angles['beta'] + + self.aero['Cn_b3'] * angles['beta'] ** 3 + \
            self.aero['Cn_p'] * angular_speed['p'] * self.b / (2 * airspeed) + self.aero['Cn_da'] * delta['a'] + \
            self.aero['Cn_dr'] * delta['r'] + self.aero['Cn_apt3'] * power_plant ** 3 + self.aero['Cn_q'] * self.c / airspeed

        return np.array([[Cx, Cy, Cz], [Cl, Cm, Cn]])

    def power_plant_model(self, n, airspeed):
        """

        Definition of the engine model for the DHC2 - Beaver.

        @param n: rpm of the engine.
        @type n: float
        @param airspeed: Airspeed of the aircraft.
        @type airspeed: float

        @return: Engine value modeled to be included in aerodynamic model.
        @rtype: float

        """

        # Definition of constants
        a = 0.08696
        b = 191.18
        Pz = 20  # Manifold pressure -> At the moment defines as constant.

        # Calculation of the engine power -> Last number is the conversion between horsepower and wats
        P = (-326.5 + (0.00412 * (Pz + 7.4) * (n + 2010) + (408.0 - 0.0965 * n) *
                       (1.0 - self.atmosphere.rho / self.atmosphere.rho_sl))) * 0.74570

        # Normalization to be included in aerodynamic model
        apt = a + b * P / (0.5 * self.atmosphere.rho * airspeed ** 3)

        return apt

    def equations(self, *args):
        pass

    def pid(self, *args):
        pass

    def controller(self, *args):
        pass

    def mass_model(self):
        """

        Mass model for the DHC2 - Beaver

        @return: Mass model of the DHC2 - Beaver.
        @rtype: dict

        """

        # Definition of the mass model for the aircraft
        return {"Ix": 5368.39, "Iy": 6928.93, "Iz": 11158.75, "Ixz": 117.64, "Ixy": 0.0, "Iyz": 0.0, "m": 2288.231}

    def landing_gear(self, *args):
        pass

    def forces(self, *args):
        pass


__all__ = ["DHC_beaver"]
