from norm_rl_gym.monitors.pacman_monitors import make_pacman_monitor
from norm_rl_gym.monitors.merchant_monitors import make_merchant_monitor
from norm_rl_gym.monitors.taxi_monitors import make_taxi_monitor
from norm_rl_gym.monitors.gardener_monitors import make_gardener_monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from utils.csvwriter import StatsWriter
from sys import stdout
from typing import Any
from numbers import Integral
import random
import numpy as np


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


class EvaluationStats(BaseCallback):
    def __init__(
        self,
        trial,
        eval_envs,
        n_eval_episodes,
        monitor_names,
        int_eval_episodes,
        int_eval_frequency,
        csv_prefix,
        train_envs,
    ):
        super().__init__(verbose=0)
        self.eval_envs = eval_envs
        self.train_envs = train_envs
        self.trial = trial
        self.results = []
        self.n_eval_episodes = n_eval_episodes
        self.n_evals = 0
        self.monitor_names = monitor_names
        self._train_eval_episodes = int_eval_episodes
        self._train_eval_freq = int_eval_frequency
        self.csv_prefix_base = csv_prefix
        trial_number = trial.number if trial is not None else None
        self.writer = StatsWriter(csv_prefix, trial_number) if csv_prefix is not None else None
        curve_prefix = f"{csv_prefix}{self._curve_csv_suffix()}" if csv_prefix is not None else None
        self.curve_writer = StatsWriter(curve_prefix, trial_number) if curve_prefix is not None else None
        self.monitored_stats = {}
        self.monitored_sq_stats = {}
        self.monitored_episode_counts = {}
        self.monitors = None
        self._monitor_cache_key = None
        self._current_episode_stats = None

    def _monitor_stat_name(self, monitor, key):
        return monitor.__class__.__name__ + "." + key

    def _maybe_register_vec_env(self, model, env):
        register = getattr(model, "register_vec_env", None)
        if callable(register):
            register(env)

    @property
    def _norm_divisor(self):
        return self.n_eval_episodes * self.n_evals

    def _norm(self, value):
        return value / self._norm_divisor

    def _build_monitors(self, name, env):
        raise NotImplementedError()

    def _detect_violation(self, monitor, env, locals_, i_env):
        raise NotImplementedError()

    def _build_monitor_cache_key(self):
        return (len(self.eval_envs.envs), tuple(self.monitor_names))

    def _curve_csv_suffix(self) -> str:
        return ""

    def _extra_state_keys(self):
        return []

    def _snapshot_state(self):
        state = {
            "monitored_stats": dict(self.monitored_stats),
            "monitored_sq_stats": dict(self.monitored_sq_stats),
            "monitored_episode_counts": dict(self.monitored_episode_counts),
            "n_evals": self.n_evals,
            "writer": self.writer,
        }
        for key in self._extra_state_keys():
            state[key] = getattr(self, key)
        return state

    def _restore_state(self, state):
        self.monitored_stats = state["monitored_stats"]
        self.monitored_sq_stats = state["monitored_sq_stats"]
        self.monitored_episode_counts = state["monitored_episode_counts"]
        self.n_evals = state["n_evals"]
        self.writer = state["writer"]
        for key in self._extra_state_keys():
            setattr(self, key, state[key])

    def _add_stat(self, i_env, stat_name, value):
        self.monitored_stats[stat_name] = self.monitored_stats.get(stat_name, 0) + value
        if self._current_episode_stats is not None:
            env_stats = self._current_episode_stats[i_env]
            env_stats[stat_name] = env_stats.get(stat_name, 0) + value

    def _finalize_episode_stats(self, i_env):
        if self._current_episode_stats is None:
            return
        for stat_name, value in self._current_episode_stats[i_env].items():
            self.monitored_sq_stats[stat_name] = self.monitored_sq_stats.get(stat_name, 0) + value * value
            self.monitored_episode_counts[stat_name] = self.monitored_episode_counts.get(stat_name, 0) + 1
        self._current_episode_stats[i_env] = {}

    def _iter_env_monitors(self, i_env):
        assert self.monitors is not None
        return self.monitors[i_env]

    def _iter_all_monitors(self):
        assert self.monitors is not None
        for i in range(len(self.eval_envs.envs)):
            for monitor in self.monitors[i]:
                yield i, monitor

    def _ensure_monitors(self):
        cache_key = self._build_monitor_cache_key()
        recreate = self.monitors is None or self._monitor_cache_key != cache_key
        if recreate:
            self.monitors = [
                [self._build_monitors(mn, e.unwrapped) for mn in self.monitor_names] for e in self.eval_envs.envs
            ]
            self._monitor_cache_key = cache_key
            first_run = self.monitored_stats == {}
            for i, monitor in self._iter_all_monitors():
                if first_run and i == 0:
                    for k, v in monitor.export().items():
                        self.monitored_stats[self._monitor_stat_name(monitor, k)] = v
                monitor.reset()
            return
        for _, monitor in self._iter_all_monitors():
            monitor.reset()

    def _accumulate_monitor_stats(self, i_env):
        for monitor in self._iter_env_monitors(i_env):
            monitor_stats = monitor.export()
            for k, v in monitor_stats.items():
                stat_name = self._monitor_stat_name(monitor, k)
                self._add_stat(i_env, stat_name, v)
            monitor.reset()

    def build_episode_csv_row(self, i_env, locals, env) -> dict[str, Any] | None:
        return None

    def _maybe_write_episode_csv(self, i_env, locals, env):
        if self.writer is None:
            return
        assert self.monitors is not None
        row = self.build_episode_csv_row(i_env, locals, env)
        if row is not None:
            self.writer.write_trial(self.monitors[i_env], row)

    def get_stats(self, n_episodes):
        stats = {}
        for k, total in self.monitored_stats.items():
            mean = total / n_episodes
            sq_total = self.monitored_sq_stats.get(k, 0)
            variance = max(0.0, (sq_total / n_episodes) - (mean * mean))
            stats[k] = mean
            stats[f"{k}_std"] = float(np.sqrt(variance))
        return stats

    def close_writers(self):
        if self.writer is not None:
            self.writer.close()
        if self.curve_writer is not None:
            self.curve_writer.close()

    def _on_step(self):
        if self._train_eval_freq > 0 and self.n_calls % self._train_eval_freq == 0:
            mean_reward, _ = self._evaluate_for_curve(
                self.model,
                self._train_eval_episodes,
                timesteps=self.num_timesteps,
            )
            self.results[-1].append(mean_reward)
        return True

    def _evaluate_for_curve(self, model, n_eval_episodes, timesteps):
        state = self._snapshot_state()
        if self.train_envs is not None:
            self._maybe_register_vec_env(model, self.eval_envs)
        self.init_eval_step(model)
        self.writer = None  # disable per-episode csv during intermediate eval
        mean_reward, std_reward = evaluate_policy(
            model,
            self.eval_envs,
            n_eval_episodes=n_eval_episodes,
            callback=self.eval_callback,
        )
        stdout.flush()
        monitor_deltas = {
            k: self.monitored_stats[k] - state["monitored_stats"].get(k, 0) for k in self.monitored_stats.keys()
        }
        self._restore_state(state)
        if self.train_envs is not None:
            self._maybe_register_vec_env(model, self.train_envs)
        if self.curve_writer is not None:
            row = {
                "Timesteps": timesteps,
                "Mean Reward": mean_reward,
                "Std Reward": std_reward,
            }
            scalar_keys = sorted([k for k in monitor_deltas if "." not in k])
            monitor_keys = sorted([k for k in monitor_deltas if "." in k])
            for k in scalar_keys + monitor_keys:
                row[k] = monitor_deltas[k] / n_eval_episodes
            self.curve_writer.write_trial([], row)
        return mean_reward, std_reward

    def _annotate_learning_curve(self):
        if self.trial is None:
            return
        for i, values in enumerate(zip(*self.results)):
            x = sum(values) / len(self.results)
            try:
                self.trial.report(x, i + 1)
            except NotImplementedError:
                self.trial.set_user_attr(f"return_{i + 1:02}", x)

    def init_training_step(self, model, seed=None):
        if self.writer is not None:
            self.writer.close()
        if self.curve_writer is not None:
            self.curve_writer.close()

        if self.csv_prefix_base is not None:
            suffix = f"_{seed:03}" if seed is not None else ""
            self.writer = StatsWriter(f"{self.csv_prefix_base}{suffix}")
            curve_prefix = f"{self.csv_prefix_base}{suffix}{self._curve_csv_suffix()}"
            self.curve_writer = StatsWriter(curve_prefix)

        if self.train_envs is not None:
            self._maybe_register_vec_env(model, self.train_envs)
        self.results.append([])
        if self.curve_writer is not None and self._train_eval_episodes > 0:
            self._evaluate_for_curve(model, self._train_eval_episodes, timesteps=0)

    def init_eval_step(self, model):
        self.n_calls = 0
        self.num_timesteps = 0
        if self.train_envs is not None:
            self._maybe_register_vec_env(model, self.eval_envs)
        self.n_evals += 1
        self._current_episode_stats = [dict() for _ in self.eval_envs.envs]

    def eval_callback(self, locals, globals_):
        i_env = locals["i"]
        env = locals["env"].envs[i_env].unwrapped
        for monitor in self._iter_env_monitors(i_env):
            self._detect_violation(monitor, env, locals, i_env)
        if locals["done"]:
            self._accumulate_monitor_stats(i_env)
            self._finalize_episode_stats(i_env)

    def _set_norm_attrs(self, trial):
        for k, v in self.monitored_stats.items():
            trial.set_user_attr(f"norm_{k}", self._norm(v))

    def _ordered_norm_tuple(self, order):
        return tuple(self._norm(self.monitored_stats[n]) for n in order)

    def annotate_trial(self, trial, order=None) -> tuple[float] | None:
        self._annotate_learning_curve()
        if trial is None:
            return
        self._set_norm_attrs(trial)
        if order is not None:
            return self._ordered_norm_tuple(order)


class PacmanEvaluationStats(EvaluationStats):
    def __init__(
        self,
        trial,
        eval_envs,
        n_eval_episodes,
        monitor_names,
        int_eval_episodes,
        int_eval_frequency,
        csv_prefix,
        train_envs=None,
    ):
        super().__init__(
            trial,
            eval_envs,
            n_eval_episodes,
            monitor_names,
            int_eval_episodes,
            int_eval_frequency,
            csv_prefix,
            train_envs,
        )
        self.pm_score = 0
        self.eaten_blue = 0
        self.eaten_red = 0
        self.left_food = 0
        self.lost = 0
        self.won = 0

    def init_eval_step(self, model):
        super().init_eval_step(model)
        self._ensure_monitors()
        self.prev_games = [e.unwrapped.game for e in self.eval_envs.envs]

    def _build_monitors(self, name, env):
        return make_pacman_monitor(name, env)

    def _curve_csv_suffix(self) -> str:
        return "_curve"

    def _extra_state_keys(self):
        return ["pm_score", "eaten_blue", "eaten_red", "left_food", "lost", "won"]

    def _detect_violation(self, monitor, env, locals_, i_env):
        monitor.detectViolation(
            self.prev_games[i_env].state,
            # locals["states"][i_env] would not work in a final state
            locals_["actions"][i_env],
        )

    def eval_callback(self, locals, globals_):
        i_env = locals["i"]
        env = locals["env"].envs[i_env].unwrapped
        for monitor in self._iter_env_monitors(i_env):
            self._detect_violation(monitor, env, locals, i_env)
        if locals["done"]:
            # have to use prev_game to not get fresh initial state
            final_state = self.prev_games[i_env].state
            # global stats
            eaten_blue, eaten_red = final_state.getGhostsEaten()
            self.pm_score += final_state.getScore()
            self.eaten_blue += eaten_blue
            self.eaten_red += eaten_red
            self.left_food += final_state.getNumFood()
            self.lost += final_state.isLose()
            self.won += final_state.isWin()
            # output to csv
            self._maybe_write_episode_csv(i_env, locals, env)
            # monitor stats
            self._accumulate_monitor_stats(i_env)
            self._finalize_episode_stats(i_env)
        self.prev_games[i_env] = env.game

    def build_episode_csv_row(self, i_env, locals, env):
        final_state = self.prev_games[i_env].state
        eaten_blue, eaten_red = final_state.getGhostsEaten()
        return {
            "Seed": self.n_evals - 1,
            "Score": final_state.getScore(),
            "Blue Eaten": eaten_blue,
            "Orange Eaten": eaten_red,
            "Win/Lose": "win" if final_state.isWin() else ("lose" if final_state.isLose() else "timeout"),
        }

    def annotate_trial(self, trial, order=None) -> tuple[float] | None:
        super().annotate_trial(trial)
        if trial is None:
            return
        trial.set_user_attr("pm_score", self._norm(self.pm_score))
        trial.set_user_attr("pm_eaten_blue", self._norm(self.eaten_blue))
        trial.set_user_attr("pm_eaten_red", self._norm(self.eaten_red))
        trial.set_user_attr("pm_left_food", self._norm(self.left_food))
        trial.set_user_attr("pm_lost", self._norm(self.lost))
        trial.set_user_attr("pm_won", self._norm(self.won))
        if order is not None:
            return self._ordered_norm_tuple(order) + (self._norm(self.pm_score),)


class MerchantEvaluationStats(EvaluationStats):
    def __init__(
        self,
        trial,
        eval_envs,
        n_eval_episodes,
        monitor_names,
        int_eval_episodes,
        int_eval_frequency,
        csv_prefix,
        train_envs,
    ):
        super().__init__(
            trial,
            eval_envs,
            n_eval_episodes,
            monitor_names,
            int_eval_episodes,
            int_eval_frequency,
            csv_prefix,
            train_envs,
        )
        self.m_score = 0

    @staticmethod
    def _is_unload_action(action: Any) -> bool:
        """Return True if merchant action corresponds to unload (name or index 5)."""
        if isinstance(action, str):
            return action.lower() == "unload"
        if isinstance(action, Integral):
            return action == 5
        return False

    @staticmethod
    def _is_timeout(info: Any) -> bool:
        """Detect timeout from callback info payload in a VecEnv-safe way."""
        if isinstance(info, dict):
            return bool(info.get("TimeLimit.truncated", False) or info.get("truncated", False))
        return False

    def init_eval_step(self, model):
        super().init_eval_step(model)
        self._ensure_monitors()
        self.monitored_stats.setdefault("UnloadMarket", 0)
        self.monitored_stats.setdefault("UnloadDanger", 0)
        self.monitored_stats.setdefault("Death", 0)
        self.monitored_stats.setdefault("Timeout", 0)

    def _build_monitors(self, name, env):
        return make_merchant_monitor(name, env)

    def _extra_state_keys(self):
        return ["m_score"]

    def _detect_violation(self, monitor, env, locals_, i_env):
        monitor.detectViolation(env.state_or_final, env.action)

    def eval_callback(self, locals, globals_):
        i_env = locals["i"]
        env = locals["env"].envs[i_env].unwrapped
        reward = float(locals.get("reward", env.unwrapped.reward))
        action = locals.get("actions", [None])[i_env]
        is_unload = self._is_unload_action(action)

        self.m_score += reward
        if is_unload and reward < 0:
            # Non-terminal unload penalty (typically unloading outside market / at danger).
            self._add_stat(i_env, "UnloadDanger", 1)

        for monitor in self._iter_env_monitors(i_env):
            self._detect_violation(monitor, env, locals, i_env)

        if locals["done"]:
            info = locals.get("info", {})
            timeout = self._is_timeout(info)

            # Exactly one terminal class per episode: success / timeout / death.
            if is_unload and reward > 0:
                self._add_stat(i_env, "UnloadMarket", 1)
            elif timeout:
                self._add_stat(i_env, "Timeout", 1)
            else:
                # Remaining terminal case in MerchantEnv is death.
                self._add_stat(i_env, "Death", 1)
            self._accumulate_monitor_stats(i_env)
            self._finalize_episode_stats(i_env)

    def annotate_trial(self, trial, order=None) -> tuple[float] | None:
        super().annotate_trial(trial)
        if trial is None:
            return
        trial.set_user_attr("m_score", self._norm(self.m_score))
        if order is not None:
            return self._ordered_norm_tuple(order) + (self._norm(self.m_score),)


class TaxiEvaluationStats(EvaluationStats):
    def __init__(
        self,
        trial,
        eval_envs,
        n_eval_episodes,
        monitor_names,
        int_eval_episodes,
        int_eval_frequency,
        csv_prefix,
        train_envs,
    ):
        super().__init__(
            trial,
            eval_envs,
            n_eval_episodes,
            monitor_names,
            int_eval_episodes,
            int_eval_frequency,
            csv_prefix,
            train_envs,
        )

    def init_eval_step(self, model):
        super().init_eval_step(model)
        self._ensure_monitors()

    def _build_monitors(self, name, env):
        return make_taxi_monitor(name, env)

    def _detect_violation(self, monitor, env, locals_, i_env):
        monitor.detectViolation(env.s, env.lastaction)

    # generic eval_callback / annotate_trial from EvaluationStats are sufficient


class GardenerEvaluationStats(EvaluationStats):
    def __init__(
        self,
        trial,
        eval_envs,
        n_eval_episodes,
        monitor_names,
        int_eval_episodes,
        int_eval_frequency,
        csv_prefix,
        train_envs,
    ):
        super().__init__(
            trial,
            eval_envs,
            n_eval_episodes,
            monitor_names,
            int_eval_episodes,
            int_eval_frequency,
            csv_prefix,
            train_envs,
        )

    def init_eval_step(self, model):
        super().init_eval_step(model)
        self._ensure_monitors()

    def _build_monitors(self, name, env):
        return make_gardener_monitor(name, env)

    def _detect_violation(self, monitor, env, locals_, i_env):
        monitor.detectViolation(env.state_or_final, env.action)

    def eval_callback(self, locals, globals_):
        i_env = locals["i"]
        env = locals["env"].envs[i_env].unwrapped
        self._add_stat(i_env, "score", env.score_delta)
        self._add_stat(i_env, "steps", 1)
        for monitor in self._iter_env_monitors(i_env):
            self._detect_violation(monitor, env, locals, i_env)
        if locals["done"]:
            for monitor in self._iter_env_monitors(i_env):
                finish_episode = getattr(monitor, "finish_episode", None)
                if callable(finish_episode):
                    finish_episode()
            self._accumulate_monitor_stats(i_env)
            self._finalize_episode_stats(i_env)

    def annotate_trial(self, trial, order=None) -> tuple[float] | None:
        super().annotate_trial(trial)
        if trial is None:
            return
        if order is not None:
            return self._ordered_norm_tuple(order) + (self._norm(self.monitored_stats.get("score", 0)),)
