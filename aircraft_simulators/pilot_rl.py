"""

Reinforcement learning Pilot
=============================

Pilot for aircraft simulators based on AI (Reinforcement learning). In this file it is defined the agent for the
simulation. Here it is defined the environment of the simulator. After, the reinforcement model should be defined
(based on Neural Networks).

Author: Alvaro Marcos Canedo
"""


import gymnasium as gym
import numpy as np


class PilotRL(gym.Env):
    """

    Pilot environment based on Reinforcement learning (AI) for Aircraft simulators.

    """

    def __init__(self, simulator, initial_state: np.ndarray, step_size=0.3, final_time=1000):
        """

        Constructor method.

        """
        # At first, only latitude and longitude
        self.observation_space = gym.spaces.Dict(
            {
                'position': gym.spaces.Box(low=[-90.0, -180.0], high=[90.0, 180.0], shape=(2,), dtype=float),
                'speed': gym.spaces.Box(low=[-120.0, -120.0, -20.0], high=np.array([120.0, 120.0, 20.0]), shape=(3,),
                                        dtype=float),
                'euler_angles': gym.spaces.Box(low=np.array([-np.pi, -np.pi/2, 0.0]), high=np.array([np.pi, np.pi/2,
                                                                                                     2 * np.pi]),
                                               shape=(3,), dtype=float),
                'fuel': gym.spaces.Box(low=0.0, high=550.0, shape=(1,), dtype=float)
            }
        )

        self.action_space = gym.spaces.Dict(
            {
                'elevator': gym.spaces.Box(low=-30.0, high=30.0, shape=(1,), dtype=float),
                'aileron': gym.spaces.Box(low=-30.0, high=30.0, shape=(1,), dtype=float),
                'rudder': gym.spaces.Box(low=-30.0, high=30.0, shape=(1,), dtype=float),
                'n': gym.spaces.Box(low=0.0, high=2400.0, shape=(1,), dtype=float)

            }
        )

        self.simulator = simulator(initial_state, step_size, final_time)
        self.state = initial_state

    def step(self, action):
        """

        Step method of Pilot environment

        :param action:
        :return:
        """

        pass

    def reset(self):
        """

        Reset method of the Pilot environment.

        :return:
        """
        pass

    def render(self):
        """

        Render method of the pilot environment.

        :return:
        """
        pass


__all__ = ["PilotRl"]
