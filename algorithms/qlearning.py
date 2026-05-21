import numpy as np
from pickle import dump, load
from typing import Any, Optional, Tuple
from gymnasium.spaces import Discrete
from collections import defaultdict
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv


class QLearning:
    def __init__(
        self,
        vec_env,
        learning_rate: float = 0.2,
        gamma: float = 0.99,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.1,
        seed: Optional[int] = None,
        verbose: int = 0,
        tensorboard_log: Optional[str] = None,
    ):
        self.vec_env = vec_env
        self.num_envs = vec_env.num_envs
        self.lr = float(learning_rate)
        self.gamma = float(gamma)
        self.exploration_fraction = float(exploration_fraction)
        self.eps_start = float(exploration_initial_eps)
        self.eps_final = float(exploration_final_eps)
        self.verbose = verbose
        self.log_interval = 10000
        self._next_log_step = self.log_interval
        self.use_masking = isinstance(vec_env, DummyVecEnv) and hasattr(
            vec_env.envs[0].unwrapped, "exclActions"
        )  # TODO: improve API

        # for logging
        self._log_ep_rets = []
        self._log_ep_lens = []
        self._log_ep_ret = np.zeros(self.num_envs, dtype=np.float32)
        self._log_ep_len = np.zeros(self.num_envs, dtype=np.int32)
        self._log_td_running = []

        # check parameters
        if not 0 <= self.lr <= 1:
            raise ValueError(f"learning_rate must be in [0, 1], got {self.lr}")
        if not 0 <= self.gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not 0 <= self.exploration_fraction <= 1:
            raise ValueError(f"exploration_fraction must be in [0, 1], got {self.exploration_fraction}")
        if not 0 <= self.eps_start <= 1:
            raise ValueError(f"exploration_initial_eps must be in [0, 1], got {self.eps_start}")
        if not 0 <= self.eps_final <= 1:
            raise ValueError(f"exploration_final_eps must be in [0, 1], got {self.eps_final}")

        # setup logging
        logging = []
        if self.verbose >= 1:
            logging.append("stdout")
        if self.verbose >= 2:
            self.log_interval = 1000
        self._next_log_step = self.log_interval
        if tensorboard_log is not None:
            logging += ["csv", "tensorboard"]
        self.logger = configure(tensorboard_log, logging)
        self._no_logging = len(logging) == 0

        # assumes action space is Discrete(n)
        if isinstance(vec_env.action_space, Discrete):
            self.nA = vec_env.action_space.n
        else:
            raise TypeError("Action space must be Discrete(n).")
        # also assumes state space is hashable (no check)

        # Q table and episode tracking
        self._q = defaultdict(lambda: np.zeros(self.nA))
        self._rng = np.random.default_rng(seed)
        self.num_timesteps = 0
        self.num_episodes = 0
        self._epsilon = self.eps_start

    @staticmethod
    def _state_key(state):
        """Create a stable, hashable key for scalar/array/tuple/dict observations."""
        if isinstance(state, np.ndarray):
            if state.ndim == 0:
                return state.item()
            return ("ndarray", state.dtype.str, state.shape, state.tobytes())
        if isinstance(state, np.generic):
            return state.item()
        if isinstance(state, (list, tuple)):
            return tuple(QLearning._state_key(x) for x in state)
        if isinstance(state, dict):
            return tuple(sorted((k, QLearning._state_key(v)) for k, v in state.items()))
        return state

    def _split_obs_batch(self, obs):
        """Convert VecEnv observation structures into a list of per-env observations.

        Supports common VecEnv outputs:
        - ndarray with first axis = num_envs
        - tuple/list/dict structures where leaves are batched on first axis
        """

        def unbatch(x, i):
            if isinstance(x, dict):
                return {k: unbatch(v, i) for k, v in x.items()}
            if isinstance(x, tuple):
                return tuple(unbatch(v, i) for v in x)
            if isinstance(x, list):
                return [unbatch(v, i) for v in x]
            if isinstance(x, np.ndarray):
                if x.ndim > 0 and x.shape[0] == self.num_envs:
                    return x[i]
                return x
            return x

        # Fast path: already a simple batched ndarray
        if isinstance(obs, np.ndarray) and obs.ndim > 0 and obs.shape[0] == self.num_envs:
            return [obs[i] for i in range(self.num_envs)]

        # Generic structured case (Tuple/Dict spaces, etc.)
        return [unbatch(obs, i) for i in range(self.num_envs)]

    # ---------- SB3-like API ----------
    def learn(
        self,
        total_timesteps: int,
        reset_num_timesteps: bool = True,
        callback: Optional[Any] = None,
    ) -> "QLearning":
        if reset_num_timesteps:
            self.num_timesteps = 0
            self.num_episodes = 0
            self._next_log_step = self.log_interval

        # learning loop
        obs = self.vec_env.reset()
        if callback is not None:
            callback.model = self
            callback.on_rollout_start()
        while self.num_timesteps < total_timesteps:
            self._epsilon = self._current_epsilon(total_timesteps)
            actions, _ = self.predict(obs, deterministic=False)
            new_obs, rewards, dones, infos = self.vec_env.step(actions)
            terminal = dones & ~np.array([x.get("TimeLimit.truncated", False) for x in infos], dtype=bool)
            # Q-learning update (vanilla, online).
            self._update(obs, actions, rewards, new_obs, terminal)
            self.num_timesteps += self.num_envs
            # Handle episode ends per env
            if np.any(dones):
                self.num_episodes += int(np.sum(dones))
            self._logging(rewards, dones, self._epsilon)
            obs = new_obs
            if callback is not None:
                callback.update_locals(locals())
                if not callback.on_step():
                    break
        if callback is not None:
            callback.on_rollout_end()
        return self

    def predict(
        self,
        observation,
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        eps = 0.0 if deterministic else self._epsilon
        actions = self._epsilon_greedy(observation, eps)
        return actions.astype(int), state

    def get_vec_normalize_env(self):
        # necessary for callbacks to work
        return None

    def register_vec_env(self, vec_env):
        old_nA = self.nA
        self.vec_env = vec_env
        self.num_envs = vec_env.num_envs
        self.use_masking = isinstance(vec_env, DummyVecEnv) and hasattr(vec_env.envs[0].unwrapped, "exclActions")

        if isinstance(vec_env.action_space, Discrete):
            self.nA = vec_env.action_space.n
        else:
            raise TypeError("Action space must be Discrete(n).")

        if self.nA != old_nA:
            self._q = defaultdict(lambda: np.zeros(self.nA))

        # reset logging buffers to the new env shape
        self._log_ep_ret = np.zeros(self.num_envs, dtype=np.float32)
        self._log_ep_len = np.zeros(self.num_envs, dtype=np.int32)

    def save(self, filename):
        # convert defaultdict to plain dict for pickling
        q_dict = dict(self._q)
        payload = {
            "q_table": q_dict,
            "metadata": {
                "learning_rate": self.lr,
                "gamma": self.gamma,
                "exploration_fraction": self.exploration_fraction,
                "exploration_initial_eps": self.eps_start,
                "exploration_final_eps": self.eps_final,
                "epsilon": self._epsilon,
                "num_timesteps": self.num_timesteps,
                "num_episodes": self.num_episodes,
                "n_actions": self.nA,
            },
        }
        with open(filename, "wb") as f:
            dump(payload, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            payload = load(f)

        if not isinstance(payload, dict) or "q_table" not in payload:
            raise ValueError("Loaded file has invalid format: expected {'q_table': ..., 'metadata': ...}")

        q_dict = payload["q_table"]
        metadata = payload.get("metadata", {})
        if not isinstance(q_dict, dict):
            raise ValueError("Loaded file does not contain a valid Q-table dictionary")

        if "n_actions" in metadata and int(metadata["n_actions"]) != self.nA:
            raise ValueError(f"Loaded model expects {int(metadata['n_actions'])} actions but current env has {self.nA}")

        # restore as defaultdict to keep the zero-init behaviour
        new_q = defaultdict(lambda: np.zeros(self.nA))
        for state, q_values in q_dict.items():
            q_arr = np.asarray(q_values, dtype=np.float64).reshape(-1)
            if len(q_arr) != self.nA:
                raise ValueError(f"Loaded Q-values have length {len(q_arr)} but action space has {self.nA} actions")
            row = np.zeros(self.nA, dtype=np.float64)
            row[:] = q_arr
            new_q[state] = row
        self._q = new_q

        # Optional training state restoration
        if "epsilon" in metadata:
            self._epsilon = float(metadata["epsilon"])
        if "num_timesteps" in metadata:
            self.num_timesteps = int(metadata["num_timesteps"])
        if "num_episodes" in metadata:
            self.num_episodes = int(metadata["num_episodes"])

    # ---------- Internals ----------
    def _current_epsilon(self, total_timesteps):
        if self.exploration_fraction == 0:
            return self.eps_final
        decay_steps = self.exploration_fraction * total_timesteps
        fraction = np.clip(self.num_timesteps / decay_steps, 0.0, 1.0)
        return self.eps_start + fraction * (self.eps_final - self.eps_start)

    def _q_np(self, obs, mask=None):
        obs_batch = self._split_obs_batch(obs)
        matrix = [self._q[self._state_key(o)] for o in obs_batch]
        if mask is not None:
            return np.ma.array(matrix, mask=mask)
        return np.array(matrix)

    def _epsilon_greedy(self, obs: np.ndarray, eps: float) -> np.ndarray:
        # obtain masked actions if possible TODO: improve
        mask = np.zeros((self.num_envs, self.nA), dtype=bool)
        if self.use_masking:
            for i, env in enumerate(self.vec_env.envs):
                excl = env.unwrapped.exclActions()
                if len(excl) >= self.nA:
                    raise ValueError("exclActions() cannot exclude all available actions")
                mask[i, excl] = True
        q_values = self._q_np(obs)
        greedy = np.empty(self.num_envs, dtype=int)
        for i in range(self.num_envs):
            allowed = np.flatnonzero(~mask[i])
            allowed_q = q_values[i, allowed]
            best_q = np.max(allowed_q)
            best_actions = allowed[allowed_q == best_q]
            greedy[i] = self._rng.choice(best_actions)
        random_mask = self._rng.random(self.num_envs) < eps
        random_actions = np.array([self._rng.choice(np.flatnonzero(~row)) for row in mask])
        actions = np.where(random_mask, random_actions, greedy)
        if self.verbose > 2:
            print("---")
            print(f"obs: {obs}")
            print(f"mask: {mask}")
            print(f"q_values: {q_values}")
            print(f"greedy: {greedy}")
            print(f"randoms: {random_actions}")
            print(f"random_mask: {random_mask}")
            print(f"actions: {actions}")
        return actions.astype(int)

    def _update(self, s, a, r, s2, terminal):
        s_batch = self._split_obs_batch(s)
        s2_batch = self._split_obs_batch(s2)
        for i in range(self.num_envs):
            si, ai, ri, s2i, ti = (
                self._state_key(s_batch[i]),
                a[i],
                r[i],
                self._state_key(s2_batch[i]),
                terminal[i],
            )
            future_q_value = 0.0 if ti else self._max_next_q(s2i, i)
            temporal_difference = ri + self.gamma * future_q_value - self._q[si][ai]
            self._q[si][ai] += self.lr * temporal_difference
            self._log_td_running.append(temporal_difference)

    def _max_next_q(self, s2i, env_idx: int) -> float:
        q = self._q[s2i]
        if not self.use_masking:
            return float(np.max(q))

        excl = self.vec_env.envs[env_idx].unwrapped.exclActions()
        if len(excl) >= self.nA:
            raise ValueError("exclActions() cannot exclude all available actions")
        allowed = np.flatnonzero(~np.isin(np.arange(self.nA), excl))
        return float(np.max(q[allowed]))

    def _logging(self, rewards, dones, eps):
        if self._no_logging:
            return
        # otherwise log everything
        self._log_ep_ret += rewards.astype(np.float32)
        self._log_ep_len += 1
        if np.any(dones):
            finished = np.where(dones)[0]
            self._log_ep_rets += tuple(self._log_ep_ret[finished])
            self._log_ep_lens += tuple(self._log_ep_len[finished])
            self._log_ep_ret[finished] = 0.0
            self._log_ep_len[finished] = 0
        if self.num_timesteps >= self._next_log_step:
            ep_rew_mean = float(np.mean(self._log_ep_rets)) if self._log_ep_rets else 0.0
            ep_len_mean = float(np.mean(self._log_ep_lens)) if self._log_ep_lens else 0.0
            self.logger.record("rollout/ep_rew_mean", ep_rew_mean)
            self.logger.record("rollout/ep_len_mean", ep_len_mean)
            self.logger.record("time/total_timesteps", int(self.num_timesteps))
            self.logger.record("time/episodes", int(self.num_episodes))
            td_arr = np.asarray(self._log_td_running, dtype=np.float32)
            self.logger.record("train/explored_states", len(self._q))
            if td_arr.size > 0:
                self.logger.record("train/td_error_mean", float(td_arr.mean()))
                self.logger.record("train/td_error_std", float(td_arr.std()))
                self.logger.record("train/td_error_abs_mean", float(np.abs(td_arr).mean()))
            else:
                self.logger.record("train/td_error_mean", 0.0)
                self.logger.record("train/td_error_std", 0.0)
                self.logger.record("train/td_error_abs_mean", 0.0)
            self.logger.record("train/epsilon", float(eps))
            self.logger.dump(self.num_timesteps)
            self._log_td_running.clear()
            self._log_ep_rets.clear()
            self._log_ep_lens.clear()
            self._next_log_step += self.log_interval
