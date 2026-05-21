import gymnasium as gym
from gymnasium import Env
import numpy as np
import os
from .py_game_merchant_display import MerchantGraphicsPyGame


actions = ["north", "south", "east", "west", "extract", "unload", "fight"]
action_dict = {name: i for i, name in enumerate(actions)}

labels = ["H", "D", "M", "R", "T", "C", "."]
labels_dict = {name: i for i, name in enumerate(labels)}

directions = {
    "north": np.array([0, -1]),
    "south": np.array([0, 1]),
    "east": np.array([1, 0]),
    "west": np.array([-1, 0]),
}


class MerchantEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        layout="basic",
        risk_fight=1.0,
        risk_death=0.0,
        capacity=5,
        sunset=28,
        render_mode=None,
        step_delay_ms=0,
    ):
        # set environment parameters
        self.layout = layout
        self.risk_fight = risk_fight
        self.risk_death = risk_death
        self.capacity = capacity
        self.sunset = sunset
        self.render_mode = None
        self.render_step_delay_ms = 0
        self.display = None
        self.set_render_mode(render_mode, step_delay_ms)
        self.load_map()
        self.wood_positions = {tuple(pos): i for i, pos in enumerate(np.argwhere(self.map == "T"))}
        self.ore_positions = {tuple(pos): i for i, pos in enumerate(np.argwhere(self.map == "R"))}
        self.max_carried_wood = min(self.capacity, len(self.wood_positions))
        self.max_carried_ore = min(self.capacity, len(self.ore_positions))
        # setup all state variables
        self.reset()
        # set action space
        self.action_space = gym.spaces.Discrete(len(action_dict))
        # set observation space
        self.observation_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(self.map.shape[1]),  # x in [0, width-1], 0
                gym.spaces.Discrete(self.map.shape[0]),  # y in [0, height-1], 1
                gym.spaces.Discrete(len(labels)),  # label indicating what is on the current grid cell
                gym.spaces.Discrete(self.max_carried_wood + 1),  # carried wood count, 3
                gym.spaces.Discrete(self.max_carried_ore + 1),  # carried ore count, 4
                gym.spaces.Discrete(self.sunset + 1),  # clock that counts up till sunset, 5
                gym.spaces.Discrete(len(actions) + 1),  # last performed action
            )
            + (gym.spaces.Discrete(2),) * len(self.wood_positions)
            + (gym.spaces.Discrete(2),) * len(self.ore_positions)
        )

    def reset(self, seed=None, *args, **kwargs):
        super().reset(seed=seed)
        self.pos = self.home
        self.label = "H"
        self.carried_wood, self.carried_ore = 0, 0
        self.wood = [1 for _ in self.wood_positions]
        self.ore = [1 for _ in self.ore_positions]
        self.clock = 0  # clock counts up from 0 until sundown, time then stops being tracked
        self.action = None
        if self.render_mode == "human":
            self.render()
        return self.get_state(), {}

    def __obs(self):
        return (
            int(self.pos[0]),
            int(self.pos[1]),
            labels_dict[self.label],
            self.carried_wood,
            self.carried_ore,
            self.clock,
            action_dict[self.action] if self.action is not None else len(actions),
            tuple(self.wood),
            tuple(self.ore),
        )

    def load_map(self):
        # compute filename
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        mapfile = f"{path}/merchant_layouts/{self.layout}.txt"
        # build map
        with open(mapfile) as file:
            rows = [row.strip() for row in file.readlines()]
            np_rows = []
            for y, row in enumerate(rows):
                np_rows.append(np.array(list(row), dtype="U1"))
                for x, entry in enumerate(row):
                    if entry == "H":
                        self.home = np.array([x, y])
                    if entry == "M":
                        self.market = np.array([x, y])
            self.map = np.stack(np_rows, axis=0)

    def exclActions(self):
        # this should be based purely on observations, so it can be determined by the learning agent
        x, y, label, carried_wood, carried_ore, _, action, _, _ = self.__obs()
        result = set()
        label = labels[label]
        # we are not allowed to turn on the spot
        if action < len(actions) and actions[action] in directions:
            for a, d in directions.items():
                if all(d == -directions[actions[action]]):
                    result.add(action_dict[a])
        # we are not allowed to run into a wall
        for a, d in directions.items():
            target = np.array([x, y]) + d
            if self.map[target[1], target[0]] == "X":
                result.add(action_dict[a])
        # we are only allowed to collect on an appropriate cell given we have capacity
        if label not in ["T", "R"] or (carried_wood + carried_ore >= self.capacity):
            result.add(action_dict["extract"])
        # we are only allowed to unload if we are attacked / at the market and have something to unload
        if (carried_wood + carried_ore == 0) or label not in ["D", "M"]:
            result.add(action_dict["unload"])
        # we are allowed to fight only if we are attacked
        if label != "D":
            result.add(action_dict["fight"])
        # if we are attacked we are only allowed to fight or unload
        if label == "D":
            result |= {action_dict[a] for a in ["north", "south", "east", "west", "extract"]}
        return list(result)

    def _proceed(self, reward, terminal):
        # saves the new state (which can be a terminal state) in self.state_or_final
        self.state_or_final = self.__obs()
        # also remembers the current reward
        self.reward = reward
        # refresh visualization on every state transition
        if self.render_mode in ["human", "rgb_array"]:
            self.render()
        return self.get_state(), self.reward, terminal, False, {}

    def step(self, action):
        # we use the action string
        self.action = actions[action]
        # increase clock, stop at sunset (afterwards we don't have to keep track of time anymore)
        if self.clock < self.sunset:
            self.clock += 1
        # if the agent is on a danger cell, they will directly be attacked
        # in this case, they can unload or try to fight
        if self.label == "D":
            # unloading will always work
            if self.action == "unload":
                resources = self.carried_wood + self.carried_ore
                self.label = "."  # no more danger (for now, if agent re-visits the field the danger is back)
                self.carried_ore, self.carried_wood = 0, 0  # but also no more resources
                return self._proceed(-50 * resources, False)  # agent has to give back the 50 per resource
            # fighting will often lead to death and sometimes work
            elif self.action == "fight":
                if self.np_random.random() <= self.risk_death:
                    return self._proceed(-100, True)  # agent dies with -100 reward, episode terminates
                else:
                    self.label = "."  # no more danger (for now, if agent re-visits the field the danger is back)
                    return self._proceed(0, False)
            else:
                # all other actions have no further effect (except progressing time)
                return self._proceed(0, False)
        # if agent is not attacked, they can walk around, collect, and unload
        else:
            # the agent can simply move around
            if self.action in ["north", "east", "south", "west"]:
                newpos = self.pos + directions[self.action]
                if (
                    (0 <= newpos[0] < self.map.shape[1])
                    and (0 <= newpos[1] < self.map.shape[0])
                    and self.map[newpos[1], newpos[0]] != "X"
                ):
                    self.pos = newpos
                    # we set the label so the agent can observe what is at their new position
                    self.label = self.map[self.pos[1], self.pos[0]]
                    if self.label == "T" and self.wood[self.wood_positions[(self.pos[1], self.pos[0])]] == 0:
                        self.label = "C"  # wood already collected
                    if self.label == "R" and self.ore[self.ore_positions[(self.pos[1], self.pos[0])]] == 0:
                        self.label = "C"  # ore already collected
                    if self.label == "D" and self.np_random.random() >= self.risk_fight:
                        self.label = "."  # we got lucky: no fight!
                return self._proceed(0, False)
            # the agent can extract a resource
            elif self.action == "extract":
                if self.label == "T":  # extracting trees
                    self.carried_wood += 1
                    self.wood[self.wood_positions[(self.pos[1], self.pos[0])]] = 0  # has now been collected
                    self.label = "C"
                    return self._proceed(50, False)  # the agent gains 50
                elif self.label == "R":  # extracting ore
                    self.carried_ore += 1
                    self.ore[self.ore_positions[(self.pos[1], self.pos[0])]] = 0  # has now been collected
                    self.label = "C"
                    return self._proceed(50, False)  # the agent gains 50
                else:  # extracting has no effect if there is nothing to extract
                    return self._proceed(0, False)
            # the agent can unload at the goal or somewhere else
            elif self.action == "unload":
                resources = self.carried_wood + self.carried_ore
                self.carried_wood, self.carried_ore = 0, 0
                if self.label == "M":  # unloading at the market ends the episode with 100 reward per resource
                    return self._proceed(100 * resources, True)  # episode is terminal
                else:  # unloading somewhere else loses the resources and 50 reward per resource
                    return self._proceed(-50 * resources, False)
            # all other actions (there is only "fight") have no effects
            else:
                return self._proceed(0, False)

    def set_render_mode(self, render_mode=None, step_delay_ms=0):
        self.render_mode = render_mode
        self.render_step_delay_ms = step_delay_ms
        if render_mode in ["human", "rgb_array"]:
            self.display = MerchantGraphicsPyGame(
                show_window=render_mode == "human", step_delay_ms=self.render_step_delay_ms
            )
        else:
            self.display = None

    def render(self):
        if self.render_mode not in ["human", "rgb_array"]:
            return None
        if self.display is None:
            self.display = MerchantGraphicsPyGame(
                show_window=self.render_mode == "human", step_delay_ms=self.render_step_delay_ms
            )
        image = self.display.render(self)
        if self.render_mode == "rgb_array":
            return image
        return None

    def close(self):
        if self.display is not None:
            self.display.close()
            self.display = None

    def get_state(self):
        x, y, label, carried_wood, carried_ore, _, action, wood, ore = self.__obs()
        return (x, y, label, carried_wood, carried_ore, _, action) + wood + ore

    def get_labels(self):
        return self.labels.getLabels(self.get_state(), None if self.action is None else action_dict[self.action])

    @property
    def labels(self):
        return Labeler(
            {
                "atTree": lambda state, _: state[2] == labels_dict["T"],
                "atRock": lambda state, _: state[2] == labels_dict["R"],
                "atHome": lambda state, _: state[2] == labels_dict["H"],
                "atMarket": lambda state, _: state[2] == labels_dict["M"],
                "atDanger": lambda state, _: state[2] == labels_dict["D"],
                "attack": lambda state, _: state[2] == labels_dict["D"],
                "sundown": lambda state, _: state[5] == self.sunset,
                "hasWood": lambda state, _: state[3] > 0,
                "hasOre": lambda state, _: state[4] > 0,
            },
            actions,
        )


class Labeler:
    def __init__(self, label_dict, action_labels):
        self.label_dict = label_dict
        self.action_labels = action_labels

    def __getattr__(self, label):
        return self.label_dict.get(label)

    def getLabels(self, state, action):
        labels = [] if action is None else [self.action_labels[action]]
        for label, condition in self.label_dict.items():
            if condition(state, action):
                labels.append(label)
        return labels
