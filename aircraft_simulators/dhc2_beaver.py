"""
DHC2 - Beaver
================

Flight Simulator environment for DHC2 - beaver aircraft.

Author: Alvaro Marcos Canedo
"""

from Ares.simulators.aircraft import Aircraft
from Ares.aircraft_simulators.atmospheric_model import AtmosphericModel
from Ares.integrators.forward_euler import ForwardEuler

import numpy as np
from scipy.signal import dlti
import json


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
        self.integration = ForwardEuler(system=self.equations(), step_size=step_size, final_time=final_time,
                                        initial_state=initial_state)

        # Path to the aerodynamic coefficients
        path = r'C:\ProgramData\Calculos\python\Ares\aircraft_simulators\dhc2_beaver_aero.json'

        # Load aerodynamic data of the aircraft
        with open(path, 'r') as f:
            self.aero = json.load(f)

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

    def _sensors(self, step_size: float) -> dict:
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

    def _actuators(self, step_size: float) -> dict:
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

    def aerodynamic_model(self, n: float, angles: dict, angular_speed: dict, airspeed: float, delta: dict,
                          accelerations: dict) -> np.ndarray:
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
        @type delta: dict
        @param accelerations: Accelerations needed for the aerodynamic model of the DHC.
        @type accelerations: dict

        @return: Aerodynamic model of the aircraft. For the DHC2, the aerodynamic model includes the power plant.
        @rtype: np.ndarray

        """

        power_plant = self.power_plant_model(n, airspeed)

        Cx = self.aero['CX_0'] + self.aero['CX_apt'] * power_plant + self.aero['CX_apt2_a'] * angles[
            'alpha'] * power_plant ** 2 + \
             self.aero['CX_a'] * angles['alpha'] + self.aero['CX_a2'] * angles['alpha'] ** 2 + \
             self.aero['CX_a3'] * angles['alpha'] ** 3 + self.aero['CX_q'] * angular_speed['q'] * self.c / airspeed + \
             self.aero['CX_dr'] * delta['r'] + self.aero['CX_df'] * delta['f'] + self.aero['CX_df_a'] * delta['f'] * \
             angles['alpha']

        Cy = self.aero['CY_0'] + self.aero['CY_b'] * angles['beta'] + self.aero['CY_p'] * angular_speed[
            'p'] * self.b / (2 * airspeed) + \
             self.aero['CY_r'] * angular_speed['r'] * self.b / (2 * airspeed) + self.aero['CY_da'] * delta['a'] + \
             self.aero['CY_dr'] * delta['r'] + self.aero['CY_dr_a'] * delta['r'] * angles['alpha'] + \
             self.aero['CY_bp'] * accelerations['bp'] * self.b / (airspeed * 2)

        Cz = self.aero['CZ_0'] + self.aero['CZ_apt'] * power_plant + self.aero['CZ_a'] * angles['alpha'] + \
             self.aero['CZ_a3'] * angles['alpha'] ** 3 + self.aero['CZ_q'] + angular_speed['q'] * self.c / airspeed + \
             self.aero['CZ_de'] * delta['e'] + self.aero['CZ_de_b2'] * delta['e'] * angles['beta'] ** 2 + \
             self.aero['CZ_df'] * delta['f'] + self.aero['CZ_df_a'] * delta['f'] * angles['alpha']

        Cl = self.aero['Cl_0'] + self.aero['Cl_b'] * angles['beta'] + \
             self.aero['Cl_p'] * angular_speed['p'] * self.b / (2 * airspeed) + \
             self.aero['Cl_r'] * self.b / (2 * airspeed) + self.aero['Cl_da'] * delta['a'] + \
             self.aero['Cl_dr'] * delta['r'] + self.aero['Cl_a2_apt'] * power_plant * angles['alpha'] ** 2 + self.aero[
                 'Cl_da_a'] * angles['alpha'] * \
             delta['a']

        Cm = self.aero['Cm_0'] + self.aero['Cm_apt'] * power_plant + self.aero['Cm_a'] * angles['alpha'] + \
             self.aero['Cm_a2'] * angles['alpha'] ** 2 + self.aero['Cm_q'] * angular_speed['q'] * self.c / airspeed + \
             self.aero['Cm_de'] * delta['e'] + self.aero['Cm_b2'] * angles['beta'] ** 2 + \
             self.aero['Cm_r'] * angular_speed['r'] * self.b / (2 * airspeed) + self.aero['Cm_df'] * delta['f']

        Cn = self.aero['Cn_0'] + self.aero['Cn_b'] * angles['beta'] + + self.aero['Cn_b3'] * angles['beta'] ** 3 + \
             self.aero['Cn_p'] * angular_speed['p'] * self.b / (2 * airspeed) + self.aero['Cn_da'] * delta['a'] + \
             self.aero['Cn_dr'] * delta['r'] + self.aero['Cn_apt3'] * power_plant ** 3 + self.aero[
                 'Cn_q'] * self.c / airspeed

        return np.array([[Cx, Cy, Cz], [Cl, Cm, Cn]])

    def power_plant_model(self, n: float, airspeed: float) -> float:
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

    def equations(self, *args) -> np.ndarray:
        """

        Equations of the aircraft model developed for the DHC2 - Beaver.

        @param args:

        @return:
        """

        # Angular speeds
        p = args[1][0]
        q = args[1][1]
        r = args[1][2]

        # Euler angles
        phi = args[1][3]
        theta = args[1][4]
        psi = args[1][5]

        # Position
        x = args[1][6]
        y = args[1][7]
        z = args[1][8]

        # Speed
        u = args[1][9]
        v = args[1][10]
        w = args[1][11]

        # Delta
        delta = args[2]

        # Accelerations
        # TODO -> Maybe the acceleration should be calculated here
        accelerations = args[3]

        # Engine
        n = args[4]

        # Calculate angle of attack and angle of sideslip
        angles = {'alpha': 1, 'beta': 1}

        # Forces wrapper
        forces, torques = self.forces(n=n, angles=angles, angular_speed={'p': p, 'q': q, 'r': r},
                                      airspeed=np.linalg.norm([u, v, w]), delta=delta,
                                      euler_angles={'phi': phi, 'theta': theta, 'psi': psi}, accelerations=accelerations,
                                      mass=self.mass['m'])

        pp = (self.mass['Iz'] * torques[0] + self.mass['Ixz'] * torques[2] - q * r * self.mass['Ixz'] ** 2 - q * r *
              self.mass[
                  'Iz'] ** 2
              + self.mass['Ixz'] * self.mass['Iy'] * p * q + self.mass['Ixz'] * self.mass['Iz'] * p * q +
              self.mass['Iz'] * self.mass['Iy'] * q * r) / (self.mass['Ix'] * self.mass['Iz'] - self.mass['Ixz'] ** 2)
        qp = (torques[1] - self.mass['Ixz'] * p ** 2 + self.mass['Ixz'] * r ** 2 - self.mass['Ix'] * p * r +
              self.mass['Iz'] * p * r) / self.mass['Iy']

        rp = (self.mass['Ixz'] * torques[0] + self.mass['Ix'] * torques[2] + p * q * self.mass['Ix'] ** 2 -
              self.mass['Ix'] * self.mass['Iy'] * p * q - p * q * self.mass['Ixz'] ** 2 -
              self.mass['Ix'] * self.mass['Ixz'] * q * r + self.mass['Ixz'] * self.mass['Iy'] * q * r - self.mass[
                  'Ixz'] * self.mass['Iz'] * q * r) / (self.mass['Ix'] * self.mass['Iz'] * self.mass['Ixz'] ** 2)

        phip = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)
        thetap = q * np.cos(phi) - r * np.sin(phi)
        psip = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)

        up = forces[0] / self.mass['m'] - q * w + r * v
        vp = forces[1] / self.mass['m'] - r * u + p * w
        wp = forces[2] / self.mass['m'] - p * v + q * u

        rotation = self.rotation_matrix(np.array([phi, theta, psi]))

        xp, yp, zp = rotation.dot(np.array([x, y, z]))

        return np.array([pp, qp, rp, phip, thetap, psip, up, vp, wp, xp, yp, zp])

    def calculate(self):
        """

        Method to launch the calculation of the equations for the DHC2 - Beaver.

        @return:
        """

        # TODO -> Correct implementation of control surfaces deflection and engine variables
        delta = {'de': 0.0, 'dr': 0.0, 'da': 0.0, 'df': 0.0}
        n = 1800
        # TODO -> Calculate acceleration
        acceleration = 0.0
        aircraft_state = self.integration.get_state()

        while self.integration.time < 1000:
            aircraft_state = np.concatenate(aircraft_state, self.integration.integrate_step(self.integration.get_state(), delta, acceleration, n))

        pass

    def mass_model(self) -> dict:
        """

        Mass model for the DHC2 - Beaver

        @return: Mass model of the DHC2 - Beaver.
        @rtype: dict

        """

        # Definition of the mass model for the aircraft
        return {"Ix": 5368.39, "Iy": 6928.93, "Iz": 11158.75, "Ixz": 117.64, "Ixy": 0.0, "Iyz": 0.0, "m": 2288.231}

    def gravity(self, euler_angles: dict, mass: float) -> np.ndarray:
        """

        Gravity model for the DHC2 - Beaver

        @param euler_angles: Euler angles of the movement.
        @type euler_angles: dict
        @param mass: Mass of the aircraft.
        @type mass: float

        @return: Gravity applied to the aircraft.
        @rtype: np.ndarray
        """

        return mass * self.g * np.array([np.cos(euler_angles['theta']),
                                         np.sin(euler_angles['phi']) * np.cos(euler_angles['theta']),
                                         np.cos(euler_angles['phi']) * np.cos(euler_angles['theta'])])

    def forces(self, n: float, angles: dict, angular_speed: dict, airspeed: float, delta: dict, accelerations: dict,
               euler_angles: dict, mass: float) -> tuple:
        """

        Forces wrapper for the DHC2 - Beaver.

        @param n: RPM of the engine.
        @type n: float
        @param angles: Necessary angles for determining the movement of the aircraft.
        @type angles: dict
        @param angular_speed: Angular speed of the aircraft.
        @type angular_speed: dict
        @param airspeed: Airspeed of the aircraft.
        @type airspeed: float
        @param delta: Control surface deflection.
        @type delta: dict
        @param accelerations: Accelerations needed for the aerodynamic model of the DHC.
        @type accelerations: dict
        @param euler_angles: Euler angles of the movement.
        @type euler_angles: dict
        @param mass: Mass of the aircraft.
        @type mass: float

        @return: Total forces (and torques) applied to the aircraft.
        @rtype: tuple
        """

        # Aerodynamic forces -> For the DHC2 - Beaver the engine is included in the aerodynamic forces
        aero = self.aerodynamic_model(n, angles, angular_speed, airspeed, delta, accelerations)

        # Gravity forces
        gravity = self.gravity(euler_angles, mass)

        # Landing gear forces -> At the moment not included
        # self.landing_gear()

        return aero[0] + gravity, aero[1]

    @staticmethod
    def rotation_matrix(angles):
        """

        Rotation matrix.

        @param angles: angles to compute the rotation (X - Y - Z).
        @type angles: np.ndarray

        @return: Rotation matrix for the angles defined in the input.
        @rtype: np.ndarray

        """

        return np.array([[np.cos(angles[1]) * np.cos(angles[2]), np.sin(angles[0]) * np.sin(angles[1]) * np.cos(angles[2]) - np.cos(angles[0]) * np.sin(angles[2]),
                          np.cos(angles[0]) * np.sin(angles[1]) * np.cos(angles[2]) + np.sin(angles[0]) * np.sin(angles[2])],
                         [np.cos(angles[1]) * np.sin(angles[2]), np.sin(angles[0]) * np.sin(angles[1]) * np.sin(angles[2]) + np.cos(angles[0]) * np.cos(angles[2]),
                          np.cos(angles[0]) * np.sin(angles[1]) * np.sin(angles[2]) - np.sin(angles[0]) * np.cos(angles[2])],
                         [-np.sin(angles[1]), np.sin(angles[0]) * np.cos(angles[1]), np.cos(angles[0]) * np.cos(angles[1])]
                         ])

    def landing_gear(self, *args):
        pass

    def pid(self, *args):
        pass

    def controller(self, *args):
        pass


__all__ = ["DHC_beaver"]
