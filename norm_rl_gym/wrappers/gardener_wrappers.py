import gymnasium as gym
import numpy as np

from norm_rl_gym.envs.gardener.gardener import actions, action_dict

INT_MAX = np.iinfo(np.int32).max


class StateFeatureObsWrapper(gym.ObservationWrapper):
    # dist_lawn          (continuous, 1 - d/size²)
    # dist_puddle        (continuous)
    # dir_lawn one-hot   (5 entries: which action heads toward nearest active grass)
    # dir_puddle one-hot (5 entries)
    n_features = 2 + 2 * len(actions)

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.n_features,), dtype=np.float32)

    def observation(self, obs):
        e = self.env.unwrapped
        ax, ay = int(e.agent[0]), int(e.agent[1])
        grass_active = np.asarray(e.grass_active, dtype=bool)
        puddles_full = np.asarray(e.puddles_full, dtype=bool)

        dist_lawn, dir_lawn = self._best_target(e.grass_dict, grass_active, ax, ay)
        dist_puddle, dir_puddle = self._best_target(e.puddle_dict, puddles_full, ax, ay)

        feat = np.zeros(self.n_features, dtype=np.float32)
        feat[0] = 1.0 / (dist_lawn + 1.0) if dist_lawn < INT_MAX else 0.0
        feat[1] = 1.0 / (dist_puddle + 1.0) if dist_puddle < INT_MAX else 0.0
        if dir_lawn is not None:
            feat[2 + dir_lawn] = 1.0
        if dir_puddle is not None:
            feat[2 + len(actions) + dir_puddle] = 1.0
        return feat

    @staticmethod
    def _best_target(target_dict, mask, ax, ay):
        """Returns distance/best action according to pre-computed BFS
        masked with currently active targets"""
        infos = target_dict.get((ax, ay), ())
        for idx, d, a in infos:
            if mask[idx] and d < INT_MAX:
                return d, a
        return INT_MAX, None


class LocalGridObsWrapper(gym.ObservationWrapper):
    # Flat Box observation:
    #   - (2*radius+1) x (2*radius+1) local view centered on the agent,
    #     one binary channel per entity type, flattened.
    #   - 4-bit Manhattan direction (right, up, left, down) toward the
    #     nearest active grass, then the same toward the nearest full puddle.
    # Channel layout: wall (also marks out-of-bounds), puddle_full,
    # puddle_empty, grass_active, grass_inactive
    N_CHANNELS = 5
    CH_WALL = 0
    CH_PUDDLE_FULL = 1
    CH_PUDDLE_EMPTY = 2
    CH_GRASS_ACTIVE = 3
    CH_GRASS_INACTIVE = 4

    N_DIR_BITS = 4
    n_dir_features = 2 * N_DIR_BITS

    def __init__(self, env, radius=2):
        super().__init__(env)
        self.radius = int(radius)
        self.window = 2 * self.radius + 1
        self.n_grid_features = self.N_CHANNELS * self.window * self.window
        self.n_features = self.n_grid_features + self.n_dir_features
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.n_features,), dtype=np.float32)

    def observation(self, obs):
        e = self.env.unwrapped
        ax, ay = int(e.agent[0]), int(e.agent[1])
        size = e.size
        radius = self.radius
        window = self.window

        grid = np.zeros((self.N_CHANNELS, window, window), dtype=np.float32)

        # Out-of-bounds cells are treated as walls (same effect on movement)
        for i in range(window):
            for j in range(window):
                gx = ax + (i - radius)
                gy = ay + (j - radius)
                if not (0 <= gx < size and 0 <= gy < size):
                    grid[self.CH_WALL, i, j] = 1.0

        def to_local(gx, gy):
            i = int(gx) - ax + radius
            j = int(gy) - ay + radius
            if 0 <= i < window and 0 <= j < window:
                return i, j
            return None

        for wx, wy in e.walls:
            loc = to_local(wx, wy)
            if loc is not None:
                grid[self.CH_WALL, loc[0], loc[1]] = 1.0

        for idx, (px, py) in enumerate(e.puddles):
            loc = to_local(px, py)
            if loc is None:
                continue
            ch = self.CH_PUDDLE_FULL if bool(e.puddles_full[idx]) else self.CH_PUDDLE_EMPTY
            grid[ch, loc[0], loc[1]] = 1.0

        for idx, (gx, gy) in enumerate(e.grass):
            loc = to_local(gx, gy)
            if loc is None:
                continue
            ch = self.CH_GRASS_ACTIVE if bool(e.grass_active[idx]) else self.CH_GRASS_INACTIVE
            grid[ch, loc[0], loc[1]] = 1.0

        feat = np.zeros(self.n_features, dtype=np.float32)
        feat[: self.n_grid_features] = grid.ravel()
        self._fill_target(feat, self.n_grid_features, e.grass, e.grass_active, ax, ay)
        self._fill_target(feat, self.n_grid_features + self.N_DIR_BITS, e.puddles, e.puddles_full, ax, ay)
        return feat

    @staticmethod
    def _fill_target(dirs, offset, positions, active_mask, ax, ay):
        best_d = INT_MAX
        best_dx = 0
        best_dy = 0
        for idx, (tx, ty) in enumerate(positions):
            if not bool(active_mask[idx]):
                continue
            dx = int(tx) - ax
            dy = int(ty) - ay
            d = abs(dx) + abs(dy)
            if d < best_d:
                best_d = d
                best_dx = dx
                best_dy = dy
        if best_d == INT_MAX:
            return
        # Bit order matches `actions`: 0=right(+x), 1=up(+y), 2=left(-x), 3=down(-y)
        if best_dx > 0:
            dirs[offset + 0] = 1.0
        elif best_dx < 0:
            dirs[offset + 2] = 1.0
        if best_dy > 0:
            dirs[offset + 1] = 1.0
        elif best_dy < 0:
            dirs[offset + 3] = 1.0


class IllegalActionPenaltyWrapper(gym.Wrapper):
    """Re-map illegal actions to 'stay' and apply a penalty."""

    def __init__(self, env, penalty=-1.0):
        super().__init__(env)
        self._stay = action_dict["stay"]
        self._penalty = penalty

    def step(self, action):
        excl = set(self.env.unwrapped.exclActions())
        extra = 0.0
        if int(action) in excl:
            action = self._stay
            extra = self._penalty
        obs, r, term, trunc, info = self.env.step(int(action))
        return obs, r + extra, term, trunc, info
