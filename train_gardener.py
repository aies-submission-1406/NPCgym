from gymnasium.wrappers import TimeLimit
import numpy as np
from os import getenv
from pprint import pprint
import time
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from norm_rl_gym.envs.gardener.gardener import GardenerEnv
from norm_rl_gym.wrappers.gardener_wrappers import (
    IllegalActionPenaltyWrapper,
    StateFeatureObsWrapper,
)
from utils.stats import GardenerEvaluationStats, set_global_seeds


SIZE = 15
TRAIN_STEPS = 100_000
TRAIN = getenv("TRAIN", "0") == "1"
RENDER = getenv("RENDER", "0") == "1"
SEED = getenv("SEED")
SEEDS = [int(SEED)] if SEED is not None else list(range(10))


def make_gardener_env(render=False):
    env = GardenerEnv(size=SIZE)
    if render:
        env.set_render_mode("human")
    env = StateFeatureObsWrapper(env)
    env = IllegalActionPenaltyWrapper(env, penalty=-1.0)
    env = TimeLimit(env, max_episode_steps=1_000)
    return env


# we use separate environments for training and evaluation
train_env = make_vec_env(lambda: make_gardener_env(render=RENDER))
eval_env = make_vec_env(lambda: make_gardener_env(render=RENDER))

# evaluation settings
evaluation_stats = GardenerEvaluationStats(
    trial=None,
    eval_envs=eval_env,
    train_envs=train_env,
    n_eval_episodes=1_000,
    monitor_names=[
        "NoCollectMonitor",
        "DrainMonitor",
        "RescueMonitor",
        "CollectOneMonitor",
        "CollectPermMonitor",
    ],
    int_eval_episodes=100,
    csv_prefix=f"models/gardener_{SIZE}_dqn",
    int_eval_frequency=TRAIN_STEPS // 20,
)

all_returns = []
for seed in SEEDS:
    set_global_seeds(seed)
    train_env.seed(seed)
    eval_env.seed(seed + 10_000)
    # reset envs
    train_env.reset()
    eval_env.reset()
    # learn model
    model = DQN(
        "MlpPolicy",
        train_env,
        device="cpu",
        verbose=1,
        seed=seed,
        learning_rate=5e-4,
        buffer_size=50_000,
        learning_starts=5_000,
        batch_size=128,
        gamma=0.99,
        train_freq=4,
        target_update_interval=2_000,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[32, 32]),
    )
    model_path = f"models/gardener_{SIZE}_dqn_{seed:03}"
    if TRAIN:
        evaluation_stats.init_training_step(model, seed=seed)
        model.learn(
            total_timesteps=TRAIN_STEPS,
            log_interval=100,
            callback=evaluation_stats,
        )
        model.save(model_path)
    model = DQN.load(model_path, env=eval_env, device="cpu")
    evaluation_stats.init_eval_step(model)

    def eval_callback(locals_, globals_):
        evaluation_stats.eval_callback(locals_, globals_)
        if RENDER:
            time.sleep(0.1)

    mean_return, std_return = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=1_000,
        callback=eval_callback,
        render=RENDER,
    )
    all_returns.append(mean_return)
    print(f"Seed {seed}: {mean_return}±{std_return}", flush=True)

mean_all = np.mean(all_returns)
std_all = np.std(all_returns)
print(f"Mean return over all seeds: {mean_all}±{std_all}")
print("Avg. number of norm violations per episode:")
pprint(evaluation_stats.get_stats(len(SEEDS) * 1_000))
evaluation_stats.close_writers()
