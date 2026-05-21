import gymnasium as gym
from gymnasium import Env
import numpy as np
from collections import deque

from norm_rl_gym.envs.gardener.gardener_rendering import GardenerRenderer


actions = ["right", "up", "left", "down", "stay"]
action_dict = {name: i for i, name in enumerate(actions)}

directions = {
    "right": np.array([1, 0]),
    "up": np.array([0, 1]),
    "left": np.array([-1, 0]),
    "down": np.array([0, -1]),
    "stay": np.array([0, 0]),
}


class GardenerLabeler:
    def __init__(self, env, action_labels):
        self.env = env
        self.action_labels = action_labels

    def getLabels(self, state, action):
        labels = [] if action is None else [self.action_labels[int(action)]]
        if self.collectFrog(state, action):
            labels.append("collectFrog")
        if self.drainNearby(state, action):
            labels.append("drainNearby")
        if self.permittedCollect(state, action):
            labels.append("permittedCollect")
        return labels

    def collectFrog(self, state, action):
        return len(self.collectedFrogs(state, action)) > 0

    def collectedFrogs(self, state, action):
        if action is None:
            return ()
        decoded = self._decode_state(state)
        return tuple(i for i, flag in enumerate(decoded["frog_collected"]) if flag)

    def drainNearby(self, state, action):
        return len(self.drainedNearbyEvents(state, action)) > 0

    def drainedNearbyEvents(self, state, action):
        if action is None:
            return ()
        decoded = self._decode_state(state)
        events = []
        for puddle_idx, drained in enumerate(decoded["puddle_drained"]):
            if not drained:
                continue
            px, py = decoded["puddles"][puddle_idx]
            nearby_frogs = []
            for frog_idx, (fx, fy) in enumerate(decoded["frogs"]):
                if decoded["collected_frogs"][frog_idx]:
                    continue
                if self.env._is_frog_near_puddle(px, py, fx, fy):
                    nearby_frogs.append(frog_idx)
            if nearby_frogs:
                events.append(tuple(nearby_frogs))
        return tuple(events)

    def permittedCollect(self, state, action):
        return self._action_name(action) == "right" and self.collectFrog(state, action)

    def _action_name(self, action):
        if action is None:
            return None
        if isinstance(action, str):
            return action
        return self.action_labels[int(action)]

    def _decode_state(self, state):
        if len(state) == 15 and isinstance(state[2], tuple):
            _, _, _, _, _, puddle_pos, _, _, frog_pos, collected_frogs, _, _, _, frog_collected, puddle_drained = state
        else:
            idx = 2
            idx += 2 * self.env.num_grass
            idx += self.env.num_grass
            idx += self.env.num_grass
            puddle_pos = state[idx : idx + 2 * self.env.num_puddles]
            idx += 2 * self.env.num_puddles
            idx += self.env.num_puddles
            idx += self.env.num_puddles
            frog_pos = state[idx : idx + 2 * self.env.num_frogs]
            idx += 2 * self.env.num_frogs
            collected_frogs = state[idx : idx + self.env.num_frogs]
            idx += self.env.num_frogs
            idx += self.env.num_frogs
            idx += self.env.num_frogs
            idx += 2 * self.env.num_walls
            frog_collected = state[idx : idx + self.env.num_frogs]
            idx += self.env.num_frogs
            puddle_drained = state[idx : idx + self.env.num_puddles]

        def pairs(flat):
            return tuple((int(flat[2 * i]), int(flat[2 * i + 1])) for i in range(len(flat) // 2))

        return {
            "puddles": pairs(puddle_pos),
            "frogs": pairs(frog_pos),
            "collected_frogs": tuple(bool(v) for v in collected_frogs),
            "frog_collected": tuple(bool(v) for v in frog_collected),
            "puddle_drained": tuple(bool(v) for v in puddle_drained),
        }


class GardenerEnv(Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, size=15, grass_respawn=50, puddle_respawn=20, score_limit=300, frog_freeze=5):
        # set environment parameters
        self.size = size
        self.grass_respawn = grass_respawn
        self.puddle_respawn = puddle_respawn
        self.score_limit = score_limit
        self.frog_freeze = frog_freeze
        self.num_frogs = max(1, int(size * size * 0.01))
        self.num_puddles = max(1, int(size * size * 0.02))
        self.num_grass = max(1, int(size * size * 0.04))
        self.num_walls = int(size * size * 0.30)
        self.render_mode = None
        self.display = None
        self.labels = GardenerLabeler(self, actions)
        self.action = None
        self.reward = 0.0
        self.score_delta = 0
        self.state_or_final = None
        # setup all state variables
        self.reset()
        # set action space
        self.action_space = gym.spaces.Discrete(len(action_dict))
        # observation = the whole world state as a flat Tuple of Discretes
        # (same style as MerchantEnv). Wrappers filter this down to whatever
        # feature shape a particular learner wants.
        self.observation_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(size),  # agent x
                gym.spaces.Discrete(size),  # agent y
            )
            # grass: positions, active flags, regrowth timers
            + (gym.spaces.Discrete(size),) * (2 * self.num_grass)
            + (gym.spaces.Discrete(2),) * self.num_grass
            + (gym.spaces.Discrete(grass_respawn + 1),) * self.num_grass
            # puddles: positions, full flags, refill timers
            + (gym.spaces.Discrete(size),) * (2 * self.num_puddles)
            + (gym.spaces.Discrete(2),) * self.num_puddles
            + (gym.spaces.Discrete(puddle_respawn + 1),) * self.num_puddles
            # frogs: positions, collected flags, captured flags, freeze timers
            + (gym.spaces.Discrete(size),) * (2 * self.num_frogs)
            + (gym.spaces.Discrete(2),) * self.num_frogs
            + (gym.spaces.Discrete(2),) * self.num_frogs
            + (gym.spaces.Discrete(frog_freeze + 1),) * self.num_frogs
            # walls: positions
            + (gym.spaces.Discrete(size),) * (2 * self.num_walls)
            # step events: frog collected, puddle drained
            + (gym.spaces.Discrete(2),) * self.num_frogs
            + (gym.spaces.Discrete(2),) * self.num_puddles
        )

    def reset(self, seed=None, *args, **kwargs):
        super().reset(seed=seed)
        self.score = 0
        # place the agent at a random cell
        self.agent = self.np_random.integers(0, self.size, size=2, dtype=int)
        # candidate cells: every grid cell except the agent's
        all_positions = {(x, y) for x in range(self.size) for y in range(self.size)}
        all_positions.discard(tuple(self.agent))
        # place frogs
        frog_positions = self.np_random.choice(list(all_positions), size=self.num_frogs, replace=False)
        for fp in frog_positions:
            all_positions.discard(tuple(fp))
        # place puddles; also exclude their 4-neighborhood so that they remain accessible for the agent
        puddle_positions = self.np_random.choice(list(all_positions), size=self.num_puddles, replace=False)
        for pp in puddle_positions:
            all_positions.discard(tuple(pp))
            px, py = pp
            for nx, ny in [(px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)]:
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    all_positions.discard((nx, ny))
        # place grass
        grass_positions = self.np_random.choice(list(all_positions), size=self.num_grass, replace=False)
        for gp in grass_positions:
            all_positions.discard(tuple(gp))

        # place walls greedily, ensuring all remaining free cells stay reachable
        def is_accessible(excluded):
            free = {(x, y) for x in range(self.size) for y in range(self.size)}
            free -= set(map(tuple, puddle_positions))
            free -= set(excluded)
            if not free:
                return True
            start = next(iter(free))
            stack = [start]
            visited = {start}
            while stack:
                cx, cy = stack.pop()
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in free and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        stack.append((nx, ny))
            return visited == free

        wall_positions = []
        remaining = list(all_positions)
        self.np_random.shuffle(remaining)
        for pos in remaining:
            if len(wall_positions) == self.num_walls:
                break
            trial = wall_positions + [tuple(pos)]
            if is_accessible(trial):
                wall_positions.append(tuple(pos))
        # if the random draw could not fit enough walls, redraw
        if len(wall_positions) < self.num_walls:
            # todo need to change seed here probably?
            return self.reset(seed=seed)

        self.frogs = np.array(frog_positions, dtype=int)
        self.puddles = np.array(puddle_positions, dtype=int)
        self.grass = np.array(grass_positions, dtype=int)
        self.walls = np.array(wall_positions, dtype=int)
        # cached for BFS over walkable cells
        self._walls_set = {tuple(map(int, w)) for w in self.walls}
        self._puddles_set = {tuple(map(int, p)) for p in self.puddles}
        # initialize per-entity state and timers
        self.collected_frogs = np.zeros(self.num_frogs, dtype=bool)
        self.captured_frogs = np.zeros(self.num_frogs, dtype=bool)
        self.frog_timer = np.zeros(self.num_frogs, dtype=int)
        self.puddles_full = np.ones(self.num_puddles, dtype=bool)
        self.puddle_timer = np.ones(self.num_puddles, dtype=int)
        self.grass_active = np.ones(self.num_grass, dtype=bool)
        self.grass_timer = np.zeros(self.num_grass, dtype=int)
        self.frog_collected = np.zeros(self.num_frogs, dtype=bool)
        self.puddle_drained = np.zeros(self.num_puddles, dtype=bool)
        # precompute static helpers
        self._compute_pos_actions()
        self._compute_puddle_dict()
        self._compute_grass_dict()
        self.action = None
        self.reward = 0.0
        self.state_or_final = self.get_state()
        if self.render_mode == "human":
            self.render()
        return self.get_state(), {}

    def _compute_pos_actions(self):
        size = self.size
        walls_set = {tuple(w) for w in self.walls}
        puddles_set = {tuple(p) for p in self.puddles}
        self.pos_actions = np.zeros((size, size, len(actions)), dtype=int)
        for x in range(size):
            for y in range(size):
                if (x, y) in walls_set or (x, y) in puddles_set:
                    continue
                for ai, name in enumerate(actions):
                    d = directions[name]
                    nx, ny = x + int(d[0]), y + int(d[1])
                    if not (0 <= nx < size and 0 <= ny < size):
                        continue
                    if (nx, ny) in walls_set or (nx, ny) in puddles_set:
                        continue
                    self.pos_actions[x, y, ai] = 1

    def _compute_puddle_dict(self):
        # BFS from each puddle; for each cell store (puddle_idx, distance, action toward puddle)
        size = self.size
        walls_set = {tuple(w) for w in self.walls}
        puddles_set = {tuple(p) for p in self.puddles}
        puddle_dist = []
        puddle_best = []
        for px, py in self.puddles:
            dist = np.full((size, size), np.iinfo(np.int32).max, dtype=np.int32)
            best = np.zeros((size, size, 2), dtype=np.int8)
            q = deque([(int(px), int(py))])
            dist[px, py] = 0
            while q:
                x, y = q.popleft()
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < size and 0 <= ny < size):
                        continue
                    if (nx, ny) in walls_set:
                        continue
                    if (nx, ny) in puddles_set and (nx, ny) != (int(px), int(py)):
                        continue
                    if dist[nx, ny] > dist[x, y] + 1:
                        dist[nx, ny] = dist[x, y] + 1
                        # Step from (nx, ny) back toward parent (x, y), i.e. one step closer to the puddle.
                        best[nx, ny] = np.array([-dx, -dy], dtype=np.int8)
                        q.append((nx, ny))
            puddle_dist.append(dist)
            puddle_best.append(best)
        self.puddle_dist = puddle_dist
        self.puddle_dict = {}
        for x in range(size):
            for y in range(size):
                if (x, y) in walls_set or (x, y) in puddles_set:
                    continue
                infos = []
                for i in range(len(self.puddles)):
                    d = int(puddle_dist[i][x, y])
                    sx, sy = int(puddle_best[i][x, y][0]), int(puddle_best[i][x, y][1])
                    a = action_dict["stay"]
                    for ai, name in enumerate(actions):
                        if name == "stay":
                            continue
                        dd = directions[name]
                        if int(dd[0]) == sx and int(dd[1]) == sy:
                            a = ai
                            break
                    infos.append((i, d, a))
                infos.sort(key=lambda v: v[1])
                self.puddle_dict[(x, y)] = infos

    def _compute_grass_dict(self):
        # BFS from each grass cell; for each walkable cell store (grass_idx, distance, action toward grass).
        size = self.size
        walls_set = self._walls_set
        puddles_set = self._puddles_set
        grass_dist = []
        grass_best = []
        for gx, gy in self.grass:
            dist = np.full((size, size), np.iinfo(np.int32).max, dtype=np.int32)
            best = np.zeros((size, size, 2), dtype=np.int8)
            q = deque([(int(gx), int(gy))])
            dist[gx, gy] = 0
            while q:
                x, y = q.popleft()
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < size and 0 <= ny < size):
                        continue
                    if (nx, ny) in walls_set or (nx, ny) in puddles_set:
                        continue
                    if dist[nx, ny] > dist[x, y] + 1:
                        dist[nx, ny] = dist[x, y] + 1
                        best[nx, ny] = np.array([-dx, -dy], dtype=np.int8)
                        q.append((nx, ny))
            grass_dist.append(dist)
            grass_best.append(best)
        self.grass_dict = {}
        for x in range(size):
            for y in range(size):
                if (x, y) in walls_set or (x, y) in puddles_set:
                    continue
                infos = []
                for i in range(len(self.grass)):
                    d = int(grass_dist[i][x, y])
                    sx, sy = int(grass_best[i][x, y][0]), int(grass_best[i][x, y][1])
                    a = action_dict["stay"]
                    for ai, name in enumerate(actions):
                        if name == "stay":
                            continue
                        dd = directions[name]
                        if int(dd[0]) == sx and int(dd[1]) == sy:
                            a = ai
                            break
                    infos.append((i, d, a))
                infos.sort(key=lambda v: v[1])
                self.grass_dict[(x, y)] = infos

    def __obs(self):
        return (
            int(self.agent[0]),
            int(self.agent[1]),
            tuple(int(v) for p in self.grass for v in p),
            tuple(self.grass_active.astype(int).tolist()),
            tuple(self.grass_timer.astype(int).tolist()),
            tuple(int(v) for p in self.puddles for v in p),
            tuple(self.puddles_full.astype(int).tolist()),
            tuple(self.puddle_timer.astype(int).tolist()),
            tuple(int(v) for p in self.frogs for v in p),
            tuple(self.collected_frogs.astype(int).tolist()),
            tuple(self.captured_frogs.astype(int).tolist()),
            tuple(self.frog_timer.astype(int).tolist()),
            tuple(int(v) for p in self.walls for v in p),
            tuple(self.frog_collected.astype(int).tolist()),
            tuple(self.puddle_drained.astype(int).tolist()),
        )

    def exclActions(self):
        # An action is excluded iff stepping that direction would leave the grid or hit a wall/puddle.
        x, y = int(self.agent[0]), int(self.agent[1])
        result = set()
        for ai in range(len(actions)):
            if self.pos_actions[x, y, ai] == 0:
                result.add(ai)
        return list(result)

    def _proceed(self, reward, terminal):
        # remember the resulting (possibly terminal) observation and reward
        self.state_or_final = self.__obs()
        self.reward = reward
        # refresh visualization on every state transition
        if self.render_mode in ["human"]:
            self.render()
        return self.get_state(), self.reward, terminal, False, {}

    def step(self, action):
        action = int(action)
        self.action = action
        self.frog_collected.fill(False)
        self.puddle_drained.fill(False)
        # validate the action against the position-based mask
        ax, ay = int(self.agent[0]), int(self.agent[1])
        if self.pos_actions[ax, ay, action] == 0:
            raise ValueError(f"Illegal action '{actions[action]}' at {(ax, ay)}")
        # move the agent
        self.agent = self.agent + directions[actions[action]]
        # Frogs move next, biased toward the nearest full puddle.
        self._move_frogs()
        # If the agent ends up sharing a cell with an uncollected frog, the frog is collected.
        if self.num_frogs > 0 and np.any(np.all(self.agent == self.frogs, axis=1)):
            for i, (fx, fy) in enumerate(self.frogs):
                if self.collected_frogs[i]:
                    continue
                if self.agent[0] == fx and self.agent[1] == fy:
                    self.collected_frogs[i] = True
                    self.captured_frogs[i] = True
                    self.frog_collected[i] = True
        reward = 0
        # tick frog freeze timers
        for f in range(self.num_frogs):
            if self.frog_timer[f] > 0:
                self.frog_timer[f] -= 1
        # grass: stepping on active grass yields +10 and starts the regrowth timer
        for i, (gx, gy) in enumerate(self.grass):
            if self.agent[0] == gx and self.agent[1] == gy:
                if self.grass_active[i]:
                    self.grass_active[i] = False
                    reward += 10
                    self.grass_timer[i] = self.grass_respawn
            else:
                if not self.grass_active[i] and self.grass_timer[i] > 0:
                    self.grass_timer[i] -= 1
                    if self.grass_timer[i] == 0:
                        self.grass_active[i] = True
        # Puddles: standing 4-adjacent to a full puddle drains it for +5 and freezes nearby frogs.
        for i, (px, py) in enumerate(self.puddles):
            if self.puddle_timer[i] > 0:
                self.puddle_timer[i] -= 1
                if self.puddle_timer[i] == 0:
                    self.puddles_full[i] = True
            if self.puddles_full[i] and abs(self.agent[0] - px) + abs(self.agent[1] - py) == 1:
                reward += 5
                self.puddles_full[i] = False
                self.puddle_timer[i] = self.puddle_respawn
                self.puddle_drained[i] = True
                for f, (c, r) in enumerate(self.frogs):
                    if self.collected_frogs[f]:
                        continue
                    if self._is_frog_near_puddle(px, py, c, r):
                        self.frog_timer[f] = self.frog_freeze
        # cap reported score at score_limit; agent reward stays uncapped
        self.score_delta = min(reward, self.score_limit - self.score)
        self.score += self.score_delta
        terminal = self.score >= self.score_limit
        # training signal only: punish unnecessary steps
        reward -= 0.1
        return self._proceed(reward, terminal)

    @staticmethod
    def _is_frog_near_puddle(px, py, fx, fy):
        dx, dy = abs(int(px) - int(fx)), abs(int(py) - int(fy))
        return (dx + dy == 1) or (dx + dy == 2 and (dx == 1 or dy == 1))

    def _move_frogs(self):
        if self.num_frogs == 0:
            return
        new_positions = []
        for i in range(self.num_frogs):
            fx, fy = int(self.frogs[i][0]), int(self.frogs[i][1])
            if self.collected_frogs[i] or self.frog_timer[i] > 0:
                new_positions.append([fx, fy])
                continue
            # frogs only consider the four cardinal directions, never stay
            valid = [a for a in range(4) if self.pos_actions[fx, fy, a] == 1]
            if not valid:
                new_positions.append([fx, fy])
                continue
            preferred = None
            if (fx, fy) in self.puddle_dict:
                for puddle_idx, _, a in self.puddle_dict[(fx, fy)]:
                    if self.puddles_full[puddle_idx]:
                        preferred = a
                        break
            chosen = None
            if preferred is not None and preferred in valid and self.np_random.random() < 0.7:
                chosen = preferred
            if chosen is None:
                if preferred is not None:
                    other = [a for a in valid if a != preferred]
                    if other:
                        chosen = int(self.np_random.choice(other))
                    elif preferred in valid:
                        chosen = preferred
                else:
                    chosen = int(self.np_random.choice(valid))
            if chosen is not None:
                d = directions[actions[chosen]]
                new_positions.append([fx + int(d[0]), fy + int(d[1])])
            else:
                new_positions.append([fx, fy])
        self.frogs = np.array(new_positions, dtype=int)

    def set_render_mode(self, render_mode=None):
        self.render_mode = render_mode
        if render_mode in ["human"]:
            self.display = GardenerRenderer()
        else:
            self.display = None

    def render(self):
        if self.render_mode not in ["human"]:
            return None
        if self.display is None:
            self.display = GardenerRenderer()
        self.display.draw(self)
        return None

    def close(self):
        if self.display is not None:
            self.display = None

    def get_state(self):
        (
            ax,
            ay,
            grass_pos,
            grass_active,
            grass_timer,
            puddle_pos,
            puddles_full,
            puddle_timer,
            frog_pos,
            collected_frogs,
            captured_frogs,
            frog_timer,
            wall_pos,
            frog_collected,
            puddle_drained,
        ) = self.__obs()
        return (
            (ax, ay)
            + grass_pos
            + grass_active
            + grass_timer
            + puddle_pos
            + puddles_full
            + puddle_timer
            + frog_pos
            + collected_frogs
            + captured_frogs
            + frog_timer
            + wall_pos
            + frog_collected
            + puddle_drained
        )
