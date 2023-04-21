"""

Forward euler integration scheme.
==================================

This class defines the integration formula for the forward euler integration scheme.

Author: Alvaro Marcos Canedo

"""

from Ares.integrators.base_integrator import BaseIntegrator


class ForwardEuler(BaseIntegrator):
    """

    This class defines the integration formula for the forward euler integration scheme.

    """

    def __init__(self, system, step_size, final_time, initial_state):
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

        """
        super().__init__(system, step_size, final_time, initial_state)
        self.time = 0.0

    def integrate(self, *args):
        pass

    def integrate_step(self, *args):
        """

        Method to integrate one step of the model.

        :param args: additional arguments of the function.
        :type args: object

        """

        self.set_state(self.get_state() + self.get_step_size() * self.system())

    def get_step_size(self):
        """

        Method to get the step size of the model

        :return: Step size defined for the model
        """
        return self.step_size

    def set_step_size(self, step_size):
        """

        Method to set the value for the step size of the model.

        :param step_size: Step size for the model.
        :type step_size: float

        :return:
        """
        if step_size <= 0 or step_size <= 10**-2:
            raise ValueError(f'The step size can not be either negative or too low. \n'
                             f'Current value: {step_size}. Try another.')
        else:
            self.step_size = step_size

    def get_state(self):
        """

        Method to get the current state of the model.

        :return Current state of the system.

        """
        return self.state

    def set_state(self, state):
        """

        Method to set the state of the model.

        :param state: state of the model.
        :type state: numpy.ndarray
        :return:
        """
        # TODO -> Implement a method to check if the new state has same number as the functions passed as arguments.

        self.state = state


__all__ = ["ForwardEuler"]
