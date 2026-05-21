import gymnasium as gym
from typing import cast


class IgnoreTimeObservation(gym.ObservationWrapper):
    """Drops the clock entry from Merchant observations."""

    def __init__(self, env):
        super().__init__(env)
        spaces = cast(gym.spaces.Tuple, self.env.observation_space).spaces
        self.observation_space = gym.spaces.Tuple(spaces[:5] + spaces[6:])

    def observation(self, observation):
        return observation[:5] + observation[6:]
