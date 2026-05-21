from gymnasium.wrappers import TimeLimit
import numpy as np
from os import getenv
from pprint import pprint
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from algorithms.qlearning import QLearning
from norm_rl_gym.envs.taxi import TaxiEnv
from norm_rl_gym.wrappers.taxi_wrappers import IgnoreWeatherRelevant
from utils.stats import TaxiEvaluationStats, set_global_seeds


TRAIN = getenv("TRAIN", "0") == "1"
RENDER = getenv("RENDER", "0") == "1"
TRAIN_STEPS = 5_000_000
SEED = getenv("SEED")
SEEDS = [int(SEED)] if SEED is not None else list(range(10))


def make_taxi_env(render=False):
    env = TaxiEnv(storm_risk=True)
    if render:
        env.set_render_mode("human")
    env = IgnoreWeatherRelevant(env)  # ignores state features only relevant for norms
    env = TimeLimit(env, max_episode_steps=50)
    return env


# we use separate environments for training and evaluation
train_env = make_vec_env(lambda: make_taxi_env(render=RENDER))
eval_env = make_vec_env(lambda: make_taxi_env(render=RENDER))

# evaluation settings
evaluation_stats = TaxiEvaluationStats(
    trial=None,  # or optuna trial
    eval_envs=eval_env,
    train_envs=train_env,
    n_eval_episodes=1_000,
    monitor_names=["EmergencyMonitor"],
    int_eval_episodes=1_000,
    csv_prefix="models/taxi",
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
    model = QLearning(train_env, verbose=1, seed=seed)
    model_path = f"models/taxi_{seed:03}.model"
    if TRAIN:
        evaluation_stats.init_training_step(model, seed=seed)
        model.learn(total_timesteps=TRAIN_STEPS, callback=evaluation_stats)
        model.save(model_path)
    model.load(model_path)
    # evaluate model
    evaluation_stats.init_eval_step(model)
    mean_return, std_return = evaluate_policy(
        model, eval_env, n_eval_episodes=1_000, callback=evaluation_stats.eval_callback
    )
    all_returns.append(mean_return)
    print(f"Seed {seed}: {mean_return}±{std_return}", flush=True)

mean_all = np.mean(all_returns)
std_all = np.std(all_returns)
print(f"Mean return over all seeds: {mean_all}±{std_all}")
print("Avg. number of norm violations per episode:")
pprint(evaluation_stats.get_stats(len(SEEDS) * 1_000))
evaluation_stats.close_writers()
