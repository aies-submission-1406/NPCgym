from gymnasium.envs.registration import make, pprint_registry, register, registry, spec

register(
    id="NormativeTaxi-v0",
    entry_point="norm_rl_gym.envs.taxi.taxi:TaxiEnv",
    reward_threshold=8,  # optimum = 8.46
    max_episode_steps=200,
)


register(
    id="NormativePacmanPO-v0",
    entry_point="norm_rl_gym.envs.pacman.pacman_env:PacmanEnv",
)

register(
    id="NormativePacman-v0",
    entry_point="norm_rl_gym.envs.pacman.pacman_env:PacmanEnv",  # TODO turn off partial observability
)
