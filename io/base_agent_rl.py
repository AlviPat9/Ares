"""

Base Agent for Reinforcement learning projects.
=======================

This is the base class for the reinforcement learning agents for AI projects.

Author: Alvaro Marcos Canedo

"""

from abc import ABC, abstractmethod

from typing import Any


class BaseAgentRL(ABC):
    """

    This is the base class for Agents used in Reinforcement learning projects.

    """

    def __init__(self, action_size: int, env: Any):
        """

        Constructor method.

        @param action_size: Action length for the output layer.
        @type action_size: int
        @param env: Environment defined for training the agent.
        @type env: Any

        """

        # Set action size
        self.action_size = action_size

        # Instantiate environment
        self.env = env

    @abstractmethod
    def train(self, *args):
        """

        Method to train the agent based on the environment provided.

        @args : Additional arguments of the function.

        """

        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')

    @abstractmethod
    def get_action(self, *args):
        """

        Method to retrieve actions based on the current state.

        @args : Additional arguments of the function.

        """

        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')

    @abstractmethod
    def _build_model(self, *args):
        """

        Method to build the Neural network model.

        @args : Additional arguments of the function.

        """

        raise NotImplementedError('As it is an abstract method, it should be overwritten in the appropriate subclass.')


__all__ = ["BaseAgentRL"]
