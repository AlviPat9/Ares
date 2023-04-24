"""

Runge Kutta 4 integration scheme.
==================================

This class defines the integration formula for the forward Runge Kutta4 integration scheme.

Author: Alvaro Marcos Canedo

"""

import numpy as np

from Ares.integrators.base_integrator import BaseIntegrator


class RK4(BaseIntegrator):
    """

    This class defines the integration formula for the forward euler integration scheme.

    """

    def __init__(self, system: object, step_size: float, final_time: float, initial_state: np.ndarray):
        """

        Constructor method

        @param system: System to integrate over time.
        @type system: object
        @param step_size: Step size for the integration.
        @type step_size: float
        @param final_time: Final time for the integration.
        @type final_time: float
        @param initial_state: Initial state for the system.
        @type initial_state: numpy.ndarray

        """

        super().__init__(system, initial_state, final_time, step_size)

        self.time = 0.0

    def integrate(self, args: object):
        """

        Method to integrate for the entire time defined the model.

        @param args: additional arguments of the function.
        @type args: object

        @return Complete state of the model during the simulation.
        @rtype np.ndarray
        """
        complete_state = self._state
        n = int((self._final_time - self.time) / self._step_size) + 1

        for i in range(n + 1):
            complete_state = np.concatenate(complete_state, self.integrate_step(args))

        return complete_state

    # noinspection PyCallingNonCallable
    def integrate_step(self, args: object):
        """

        Method to integrate one step of the model.

        @param args: additional arguments of the function.
        @type args: object

        """
        k1 = self._step_size * self._system(self.time, args)
        k2 = self._step_size * self._system(self.time + self._step_size/2, tuple(map(lambda x, y: x + y/2, args, k1)))
        k3 = self._step_size * self._system(self.time + self._step_size/2, tuple(map(lambda x, y: x + y/2, args, k2)))
        k4 = self._step_size * self._system(self.time + self._step_size, tuple(map(lambda x, y: x + y, args, k3)))

        self._state = self._state + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        self.time += self._step_size

        return self._state

    def get_step_size(self):
        """

        Method to get the step size of the model

        @return: Step size defined for the model
        @rtype: float

        """

        return self._step_size

    def set_step_size(self, step_size: float):
        """

        Method to set the value for the step size of the model.

        @param step_size: Step size for the model.
        @type step_size: float

        """

        if step_size <= 0 or step_size <= 10**-2:
            raise ValueError(f'The step size can not be either negative or too low. \n'
                             f'Current value: {step_size}. Try another.')
        else:
            self._step_size = step_size

    def get_state(self):
        """

        Method to get the current state of the model.

        @return Current state of the system.

        """

        return self._state

    def set_state(self, state: np.ndarray):
        """

        Method to set the state of the model.

        @param state: state of the model.
        @type state: numpy.ndarray

        """
        # TODO -> Implement a method to check if the new state has same number as the functions passed as arguments.

        self._state = state


__all__ = ["RK4"]
