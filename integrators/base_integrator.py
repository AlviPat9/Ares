"""
Base integrator
====================

This class is an abstract class that defines all common method to all integration schemes.

Author: Alvaro Marcos Canedo

"""

from abc import ABC, abstractmethod

import numpy as np


class BaseIntegrator(ABC):
    """
    This class is an abstract class that defines all common method to all integration schemes. It is an abstract class
    so its methods shall be overwritten.

    """
    _system: object

    def __init__(self, system: object, initial_state: np.ndarray, final_time: float, step_size: float):
        """

        Constructor method.

        @param system: System to integrate over time.
        @type system: object
        @param initial_state: Initial state for the system.
        @type initial_state: numpy.ndarray
        @param final_time: Final time for the integration.
        @type final_time: float
        @param step_size: Step size for the integration.
        @type step_size: float

        """
        self._system = system
        self._state = initial_state
        self._step_size = step_size
        self._final_time = final_time
        self._initial_state = initial_state

    @abstractmethod
    def integrate(self, *args):
        """

        Method responsible for integrating the given equations over a given interval..

        @param args: additional arguments of the integrate method (in the appropriate class).

        """

        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')

    @abstractmethod
    def integrate_step(self, *args):
        """

        Method responsible for integrating the given equations just for one step. It should take as input the current
        time, the current state and the time step size.

        @param args: additional arguments of the integrate method (in the appropriate class).

        """

        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')

    @abstractmethod
    def get_step_size(self, *args):
        """

        Method to get the current step_size of the method.

        @param args: Additional arguments of the function.

        """

        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')

    @abstractmethod
    def set_step_size(self, *args):
        """

        Method to set the value of the step size. It should include warnings and error checks. It should take as input
        new step size.

        @param args: Additional arguments of the function. Defined in the appropriate class.

        """

        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')

    @abstractmethod
    def get_state(self, *args):
        """

        Method to get the current state of the method.

        @param args: Additional arguments of the function.

        """

        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')

    @abstractmethod
    def set_state(self, *args):
        """

        Method to set the current state of the model. It should take as input a new state.

        @param args:

        """

        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')


__all__ = ["BaseIntegrator"]
