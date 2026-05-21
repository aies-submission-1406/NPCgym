from gymnasium.wrappers import TimeLimit, TransformReward
import numpy as np
from os import getenv
from pprint import pprint
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from norm_rl_gym.envs.pacman import PacmanEnv
from utils.stats import PacmanEvaluationStats, set_global_seeds


LAYOUT = getenv("LAYOUT", "smallClassic")
FEATURES = getenv("FEATURES", "complete")
TRAIN = getenv("TRAIN", "0") == "1"
RENDER = getenv("RENDER", "0") == "1"
TRAIN_STEPS = 5_000_000
SEED = getenv("SEED")
SEEDS = [int(SEED)] if SEED is not None else list(range(10))


def make_pacman_env(render=False):
    render_mode = "human" if render else None
    env = PacmanEnv(layout=LAYOUT, features=FEATURES, render_mode=render_mode)
    env = TransformReward(env, lambda r: r / 100)
    env = TimeLimit(env, max_episode_steps=300)
    return env


# we use separate environments for training and evaluation
train_env = make_vec_env(lambda: make_pacman_env(render=False))
eval_env = make_vec_env(lambda: make_pacman_env(render=RENDER))

# evaluation settings
evaluation_stats = PacmanEvaluationStats(
    trial=None,  # or optuna trial
    eval_envs=eval_env,
    train_envs=train_env,
    n_eval_episodes=1_000,
    monitor_names=[
        "VeganMonitor",
        "VegetarianBlueMonitor",
        "VegetarianOrangeMonitor",
        "ConditionalVeganMonitor",
        "PenaltyMonitor",
        "HungryMonitor",
        "HungryVeganMonitor",
        "HungryVeganPenaltyMonitor",
        "VeganPreferenceMonitor",
        "HungryVegetarianMonitor",
    ],
    int_eval_episodes=1_000,
    csv_prefix=f"models/pacman_{LAYOUT}_{FEATURES}",
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
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=seed,
    )
    model_path = f"models/pacman_{LAYOUT}_{FEATURES}_{seed:03}.model"
    if TRAIN:
        evaluation_stats.init_training_step(model, seed=seed)
        model.learn(total_timesteps=TRAIN_STEPS, callback=evaluation_stats)
        model.save(model_path)
    model = PPO.load(model_path, env=eval_env)
    # evaluate model
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
