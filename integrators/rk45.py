"""

Runge Kutta 45 integration scheme.
==================================

This class defines the integration formula for the forward Runge Kutta45, a variable step size integration scheme.

Author: Alvaro Marcos Canedo

"""

import numpy as np

from Ares.integrators.base_integrator import BaseIntegrator


class RK45(BaseIntegrator):
    """

    This class defines the integration formula for Runge Kutta 45 variable step size integration scheme.

    """

    def __init__(self, system: object, step_size: float, final_time: float, initial_state: np.ndarray, tol=1e-6 float):
        """

        Constructor method

        :param system: System to integrate over time.
        :type system: object
        :param step_size: Step size for the integration.
        :type step_size: float
        :param final_time: Final time for the integration.
        :type final_time: float
        :param initial_state: Initial state for the system.
        :type initial_state: numpy.ndarray
        :param tol: Tolerance value for the model.
        :type tol: float

        """

        super().__init__(system, initial_state, final_time, step_size)
		
		# Set initial time and tolerance
        self.time = 0.0
		self.tol = tol

        # Safety factor and limits for a robus step size control
        self.safety_factor = 0.9
        self.max_step_increase = 5.0
        self.min_step_decrease = 0.2
        self.max_iteration = 20

    def integrate(self, *args: object):
        """

        Method to integrate for the entire time defined the model.

        :param args: additional arguments of the function.
        :type args: object

        :return Complete state of the model during the simulation.
        :rtype np.ndarray
        """
        complete_state = self._state
        n = int((self._final_time - self.time) / self._step_size) + 1

        for i in range(n + 1):
            complete_state = np.concatenate(complete_state, self.integrate_step(args))

        return complete_state

    # noinspection PyCallingNonCallable
    def integrate_step(self, *args: object):
        """

        Method to integrate one step of the model.

        :param args: additional arguments of the function.
        :type args: object

        """
        # Initialize the number of iterations
        iteration = 0

        while True:
            # Calculate the characteristic values of the runge kutta
            k1 = self._step_size * self._system(self.time, self._state, *args)
            k2 = self._step_size * self._system(self.time + self._step_size * 0.25, self._state + k1 * 0.25, *args)
            k3 = self._step_size * self._system(self.time + self._step_size * 0.375, self._state + k1 * 3/32 + k2 * 9/32, *args)
            k4 = self._step_size * self._system(self.time + self._step_size * 12/13, self._state + k1 * 1932/2197 - k2 * 7200/2197 + k3 * 7296/2197, *args)
            k5 = self._step_size * self._system(self.time + self._step_size, self._state + k1 * 439/216 - 8*k2 + k3 * 3680/513 - k4 * 845/4104, *args)
            k6 = self._step_size * self._system(self.time + self._step_size * 0.5, self._state - k1 * 8/27 + k2 * 2 - k3 * 3544/2565 + k4 * 1859/4104 - k5 * 11/40, *args)

            # Compute new state based on Runge Kutta 4 and 5 methods
            y_4th = self._state + k1 * 25/216 + k3 * 1408/2565 + k4 * 2197/4104 - k5 * 1/5
            y_5th = self._state + k1 * 16/135 + k3 * 6656/12825 + k4 * 28561/56430 - k5 * 9/50 + k6 * 2/55
            
            # Compute error
            error = np.linalg.norm(y_5th - y_4th)

            # Avoid division by 0
            if error < 1e-15:
                error = 1e-15
            
            # Adaptative Step control
            # Calculate optimal time step
            h_opt = self.safety_factor * self._step_size * (self.tol / error) ** 0.2

            # Set limitation for max increase and min decrease
            if h_opt > self._step_size * self.max_step_increase:
                h_opt = self._step_size * self.max_step_increase:
            elif h_opt < self._step_size * self.min_step_decrease:
                h_opt = self._step_size * self.min_step_decrease

            # Accept or reject the new step
            if error <= self.tol:
                # Time step has been accepted
                # Update time
                self.time += self._step_size
                
                # Update new time step
                self._state = y_5th

                # Update new Time step
                self._step_size = h_opt

                # Return the state
                return self._state
            else:
                # Time step has not been accepted
                # Re-do de simulation with the optimal time step calculated (without updating the state)
                self._step_size = h_opt
                iteration +=1

                 if iteration > self.max_iteration:
                    raise RuntimeError(
                        f"RK45 integrator failed. Step size has been reduced "
                        f"{self.max_iteration} times without meeting the error tolerance "
                        f"at t={self.time}. The system may be stiff or a singularity may be present."
                    )

    def get_step_size(self):
        """

        Method to get the step size of the model

        :return: Step size defined for the model
        :rtype: float

        """

        return self._step_size

    def set_step_size(self, step_size: float):
        """

        Method to set the value for the step size of the model.

        :param step_size: Step size for the model.
        :type step_size: float

        """

        if step_size <= 0 or step_size <= 10**-2:
            raise ValueError(f'The step size can not be either negative or too low. \n'
                             f'Current value: {step_size}. Try another.')
        else:
            self._step_size = step_size

    def get_state(self):
        """

        Method to get the current state of the model.

        :return Current state of the system.

        """

        return self._state

    def set_state(self, state: np.ndarray):
        """

        Method to set the state of the model.

        :param state: state of the model.
        :type state: numpy.ndarray

        """
        # TODO -> Implement a method to check if the new state has same number as the functions passed as arguments.

        self._state = state
		

__all__ = ["RK45"]
