import gymnasium as gym
from typing import Any, cast


class IgnoreWeatherRelevant(gym.ObservationWrapper):
    """Compress Taxi storm-risk states by dropping norm-only fields: home, flood, shelter."""

    def __init__(self, env):
        super().__init__(env)
        # (row, col, passenger, destination, rain, hurricane)
        self.observation_space = gym.spaces.Discrete(5 * 5 * 5 * 4 * 2 * 11)

    def _encode_compact(self, row, col, passenger, destination, rain, hurricane):
        s = row
        s = s * 5 + col
        s = s * 5 + passenger
        s = s * 4 + destination
        s = s * 2 + int(rain)
        s = s * 11 + hurricane
        return s

    def observation(self, observation):
        base_env = cast(Any, self.env.unwrapped)
        decoded = tuple(base_env.decode(int(observation)))
        if len(decoded) == 9:
            row, col, passenger, destination, rain, hurricane, _, _, _ = decoded
            return self._encode_compact(row, col, passenger, destination, rain, hurricane)
        return int(observation)
