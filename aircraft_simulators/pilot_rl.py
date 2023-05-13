"""

Reinforcement learning Pilot
=============================

Pilot for aircraft simulators based on AI (Reinforcement learning). In this file it is defined the agent for the
simulation. Here it is defined the environment of the simulator. After, the reinforcement model should be defined
(based on Neural Networks).

Author: Alvaro Marcos Canedo
"""

from Ares.utilities.keys import AircraftKeys as Ak
from Ares.utilities.tools import haversine

import gymnasium as gym
import numpy as np
import logging as log


class PilotRL(gym.Env):
    """

    Pilot environment based on Reinforcement learning (AI) for Aircraft simulators.

    """

    def __init__(self, simulator, destination: np.ndarray, initial_state: np.ndarray, step_size=0.3, final_time=1000):
        """

        Constructor method.

        @param simulator: Aircraft simulator.
        @type simulator:
        @param destination: Coordinates (latitude and longitude) of the place to get to.
        @type destination: np.ndarray
        @param initial_state: Initial state of the model.
        @type initial_state: np.ndarray
        @param step_size: Step size for the calculation.
        @type step_size: float, optional
        @param final_time: Final time for the simulation.
        @type final_time: float, optional

        """
        # At first, only latitude and longitude
        self.observation_space = gym.spaces.Dict(
            {
                Ak.position: gym.spaces.Box(low=np.ndarray([-90.0, -180.0]), high=np.ndarray([90.0, 180.0]),
                                            shape=(2,), dtype=np.float32),
                Ak.speed: gym.spaces.Box(low=np.ndarray([-120.0, -120.0, -20.0]), high=np.array([120.0, 120.0, 20.0]),
                                         shape=(3,), dtype=np.float32),
                Ak.euler_angles: gym.spaces.Box(low=np.array([-np.pi, -np.pi / 2, 0.0]),
                                                high=np.array([np.pi, np.pi / 2, 2 * np.pi]),
                                                shape=(3,), dtype=np.float32),
                Ak.fuel: gym.spaces.Box(low=0.0, high=550.0, shape=(1,), dtype=np.float32)
            }
        )

        self.action_space = gym.spaces.Dict(
            {
                Ak.de: gym.spaces.Box(low=-30.0, high=30.0, shape=(1,), dtype=np.float32),
                Ak.da: gym.spaces.Box(low=-30.0, high=30.0, shape=(1,), dtype=np.float32),
                Ak.dr: gym.spaces.Box(low=-30.0, high=30.0, shape=(1,), dtype=np.float32),
                Ak.t_lever: gym.spaces.Box(low=0.0, high=2400.0, shape=(1,), dtype=np.float32)

            }
        )

        # Create instance of the simulator
        self.simulator = simulator(initial_state, step_size, final_time)

        # Set initial state
        self.state = initial_state

        # Set final destination
        self.destination = destination

        # Initialize step of the model
        self.current_step = 0

        # Set max step for the environment
        self.max_step = 1000

        # Set a tolerance for checking if it arrived at the destination port
        self.tol = 1000.0  # Defined as 1km

        # Initialize last distance
        self.last_distance = None

        # Initialize the logger
        self.logger = log.getLogger(__name__)
        self.logger.setLevel(log.INFO)
        self.logger.addHandler(log.StreamHandler())

    def step(self, action: gym.spaces.Dict) -> tuple:
        """

        Step method of Pilot environment

        @param action: Action taken by the pilot to fly the aircraft.
        @type action: gym.spaces.Dict

        @return: Current state, current reward of the function and status of the goal.
        @rtype: tuple
        """

        # Calculate current step
        self.state = self.simulator.step(action)

        # Update current step
        self.current_step += 1

        # Calculate distance to destination based on Heaversine formula
        distance = haversine(self.destination[0], self.destination[1], self.state[Ak.position][0],
                             self.state[Ak.position][1])

        # TODO -> Define reward function. The reward function can take multiple inputs. The first one must be the
        #  distance, then maybe it is a good approach to take fuel and steps as a penalty.
        distance_reward = max(0, self.last_distance - distance)
        # fuel_penalty = self.state[Ak.fuel]
        # time_penalty = self.simulator.integration.time / self.max_step
        # reward = distance_reward - fuel_penalty - time_penalty
        reward = distance_reward

        # TODO -> Define Done
        # Check if done
        if distance < self.tol:
            done = True
        elif self.current_step >= self.max_step:
            done = True
        else:
            done = False

        # TODO -> Define relevant information in the log
        # Log relevant information
        self.logger.info(f"Step: {self.current_step}, Reward: {reward}, Done: {done}")

        # Set last distance
        self.last_distance = distance

        return self.state, reward, done

    def reset(self, **kwargs) -> np.ndarray:
        """

        Reset method of the Pilot environment.

        @return: Initial state of the model.
        @rtype: np.ndarray
        :param **kwargs:
        """

        # Reset simulation
        self.state = self.simulator.reset()

        # Reset step
        self.current_step = 0

        return self.state

    def render(self):
        """

        Render method of the Pilot environment. Used to generate and return a visual representation of the
        current state.

        @return:
        """
        pass


__all__ = ["PilotRL"]
