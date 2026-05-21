# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"Feature extractors for Pacman game states"

import math
import queue
from collections import defaultdict

import gymnasium.spaces
import numpy as np

from .game import Directions, Actions
from .pacman import SCARED_TIME, COLLISION_TOLERANCE
from .util import Counter, nearestPoint
from .labels import Labels
import norm_rl_gym.automata.pacman_DFAs as pdfa
from norm_rl_gym.automata.pacman_DFAs import *
from norm_rl_gym.monitors.pacman_monitors import *


def features_dict_to_array(features: Counter):
    pac_map = features.pop("map", None)
    sorted_features = sorted(list(features.items()), key=lambda x: x[0])
    ext_features = list(
        filter(
            lambda x: x[0].startswith("#-of-ghosts-1-step-away-") or x[0].startswith("#-of-scared-ghosts-1-step-away-"),
            sorted_features,
        )
    )
    non_ext_features = list(
        filter(
            lambda x: (
                not (x[0].startswith("#-of-ghosts-1-step-away-") or x[0].startswith("#-of-scared-ghosts-1-step-away-"))
            ),
            sorted_features,
        )
    )
    sorted_features = non_ext_features + ext_features
    other_features = np.array(list(zip(*sorted_features))[1])
    # print(list(enumerate(sorted_features)))
    if pac_map is not None:
        return np.concatenate((other_features, pac_map.flatten()))
    else:
        return other_features


class FeatureExtractor:
    def __init__(self, height, width):
        self.width = width
        self.height = height
        self.legal_neighbor_cache = dict()

    def getFeatures(self, state):
        """
        Returns a dict from features to counts
        Usually, the count will just be 1.0 for
        indicator functions.
        """
        raise Exception("Not implemented")

    def get_obs_space(self, nr_ghosts):
        raise Exception("Not implemented")

    def get_nr_dfa_states(self):
        return 0


class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = Counter()
        feats[(state, action)] = 1.0
        return feats


class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = Counter()
        feats[state] = 1.0
        feats["x=%d" % state[0]] = 1.0
        feats["y=%d" % state[0]] = 1.0
        feats["action=%s" % action] = 1.0
        return feats


def closestCapsule(pos, capsules, walls, legal_neighbor_cache=None):
    if len(capsules) == 0:
        return -1, -1
    fringe = [(pos[0], pos[1], 0, None)]  # additionally return direction of first move toward food
    expanded = set()
    while fringe:
        pos_x, pos_y, dist, dir = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if (pos_x, pos_y) in capsules:
            return dist, dir
        # otherwise spread out from the location to its neighbours
        if legal_neighbor_cache is not None:
            nbrs = legal_neighbor_cache.get((pos_x, pos_y), None)
            if nbrs is None:
                nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
                legal_neighbor_cache[(pos_x, pos_y)] = nbrs
            else:
                nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            next_dir = dir
            if next_dir is None:
                if nbr_y - pos_y == 1:
                    next_dir = Directions.NORTH
                if nbr_y - pos_y == -1:
                    next_dir = Directions.SOUTH
                if nbr_x - pos_x == 1:
                    next_dir = Directions.EAST
                if nbr_x - pos_x == -1:
                    next_dir = Directions.WEST
            fringe.append((nbr_x, nbr_y, dist + 1, next_dir))


def closestFood(pos, food, walls, legal_neighbor_cache=None, return_dir=False):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0, None)]  # additionally return direction of first move toward food
    expanded = set()
    while fringe:
        pos_x, pos_y, dist, dir = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            if return_dir:
                return dist, dir
            else:
                return dist
        # otherwise spread out from the location to its neighbours
        if legal_neighbor_cache is not None:
            nbrs = legal_neighbor_cache.get((pos_x, pos_y), None)
            if nbrs is None:
                nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
                legal_neighbor_cache[(pos_x, pos_y)] = nbrs
            else:
                nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            next_dir = dir
            if next_dir is None:
                if nbr_y - pos_y == 1:
                    next_dir = Directions.NORTH
                if nbr_y - pos_y == -1:
                    next_dir = Directions.SOUTH
                if nbr_x - pos_x == 1:
                    next_dir = Directions.EAST
                if nbr_x - pos_x == -1:
                    next_dir = Directions.WEST
            fringe.append((nbr_x, nbr_y, dist + 1, next_dir))
    # no food found
    return None


def ghostDistance(pac, ghost, walls, legal_neighbor_cache=None, return_dir=False):
    # fringe = [(pac[0], pac[1], 0, None)]
    fringe = queue.SimpleQueue()
    fringe.put((pac[0], pac[1], 0, None))
    expanded = set()
    while not fringe.empty():
        pos_x, pos_y, dist, dir = fringe.get()  # fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if abs(pos_x - ghost[0]) <= COLLISION_TOLERANCE and abs(pos_y - ghost[1]) <= COLLISION_TOLERANCE:
            offset = abs(pos_x - ghost[0]) + abs(pos_y - ghost[1])
            if return_dir:
                return dist + offset, dir
            else:
                return dist + offset
        # otherwise spread out from the location to its neighbours
        if legal_neighbor_cache is not None:
            nbrs = legal_neighbor_cache.get((pos_x, pos_y), None)
            if nbrs is None:
                nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
                legal_neighbor_cache[(pos_x, pos_y)] = nbrs
        else:
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            next_dir = dir
            if next_dir is None:
                if nbr_y - pos_y == 1:
                    next_dir = Directions.NORTH
                if nbr_y - pos_y == -1:
                    next_dir = Directions.SOUTH
                if nbr_x - pos_x == 1:
                    next_dir = Directions.EAST
                if nbr_x - pos_x == -1:
                    next_dir = Directions.WEST
            # fringe.append((nbr_x, nbr_y, dist+1,next_dir))
            fringe.put((nbr_x, nbr_y, dist + 1, next_dir))
    print(f"PAC: {pac}")
    print(f"ghost: {ghost}")
    raise Exception("no ghost found")


class HungryExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostStates()

        features = Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        if action is None:
            next_x = x
            next_y = y
        else:
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)

        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) for g in ghosts
        )
        scared = []
        for g in ghosts:
            if g.isScared():
                scared.append(g)
        features["#-of-scared-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) for g in scared
        )

        cap_dist, cap_dir = closestCapsule(
            (next_x, next_y), state.getCapsules(), walls, legal_neighbor_cache=self.legal_neighbor_cache
        )
        features["closest-capsule-dist"] = cap_dist

        dist = closestFood((next_x, next_y), food, walls, self.legal_neighbor_cache)
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        # features["map"] = constract_map_array(self.height,self.width,state)
        features.divideAll(10.0)
        return features

    def get_obs_space(self, nr_ghosts):
        other_obs_size = 6
        # map_size = self.width * self.height
        obs_size = other_obs_size + nr_ghosts * 13
        low = np.zeros(obs_size)
        high = np.ones(obs_size) * max(self.width, self.height)
        # high[other_obs_size:] = 3 + 4 + 4
        return gymnasium.spaces.Box(low=low, high=high)


class DeepRLBaseExtractor(FeatureExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.legal_neighbor_cache = dict()

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostStates()

        features = Counter()

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        features["x"] = x
        features["y"] = y

        x_int, y_int = int(x + 0.5), int(y + 0.5)
        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if next_x < walls.width and next_y < walls.height:
                if not walls[next_x][next_y]:
                    features[f"poss-dir-{dir}"] = 1
                else:
                    features[f"poss-dir-{dir}"] = 0
                    # possible.append(dir)

        features["#-of-ghosts-1-step-away"] = 0
        features["#-of-scared-ghosts-1-step-away"] = 0
        for i, g in enumerate(ghosts):
            is_scared = 1 if g.isScared() else 0
            g_x, g_y = g.getPosition()
            features[f"ghost-{i}-scared"] = is_scared
            features[f"ghost-{i}-scaredtime"] = g.scaredTimer / SCARED_TIME
            features[f"ghost-{i}-x"] = g_x
            features[f"ghost-{i}-y"] = g_y
            g_dist, g_dir = ghostDistance(
                (x, y), g.getPosition(), walls, legal_neighbor_cache=self.legal_neighbor_cache, return_dir=True
            )

            features[f"ghost-{i}-dist"] = g_dist
            # features[f"ghost-{i}-dir"] = g_dir if g_dir is not None else -1

            add_direction_ohe(features, g_dir if g_dir is not None else Directions.STOP, f"ghost-{i}-dir")
            add_direction_ohe(features, g.getDirection(), f"ghost-{i}-heading")
            # features[f"ghost-{i}-heading"] = g.getDirection()
            ghost_vector_x = g_x - x
            ghost_vector_y = g_y - y
            if abs(ghost_vector_x) <= COLLISION_TOLERANCE + abs(ghost_vector_y) <= COLLISION_TOLERANCE:
                ghost_approx_angle = 0
            else:
                if ghost_vector_x > 0:
                    ghost_angle = math.atan(ghost_vector_y / ghost_vector_x)
                elif ghost_vector_x < 0 and ghost_vector_y >= 0:
                    ghost_angle = math.atan(ghost_vector_y / ghost_vector_x) + math.pi
                elif ghost_vector_x < 0 and ghost_vector_y < 0:
                    ghost_angle = math.atan(ghost_vector_y / ghost_vector_x) - math.pi
                elif ghost_vector_x == 0 and ghost_vector_y > 0:
                    ghost_angle = math.pi / 2
                elif ghost_vector_x == 0 and ghost_vector_y < 0:
                    ghost_angle = -math.pi / 2

                # assign approximate angles in cardinal directions
                if math.pi / 2 - math.pi / 8 <= ghost_angle <= math.pi / 2 + math.pi / 8:
                    ghost_approx_angle = 1  # NORTH ( different directions than used for actions)
                elif math.pi / 4 - math.pi / 8 <= ghost_angle <= math.pi / 4 + math.pi / 8:
                    ghost_approx_angle = 2  # NORTHEAST
                elif -math.pi / 8 <= ghost_angle <= +math.pi / 8:
                    ghost_approx_angle = 3  # EAST
                elif -math.pi / 4 - math.pi / 8 <= ghost_angle <= -math.pi / 4 + math.pi / 8:
                    ghost_approx_angle = 4  # SOUTHEAST
                elif -math.pi / 2 - math.pi / 8 <= ghost_angle <= -math.pi / 2 + math.pi / 8:
                    ghost_approx_angle = 5  # SOUTH
                elif -3 * math.pi / 4 - math.pi / 8 <= ghost_angle <= -3 * math.pi / 4 + math.pi / 8:
                    ghost_approx_angle = 6  # SOUTHWEST
                elif ghost_angle <= -math.pi + math.pi / 8 or ghost_angle >= 3 * math.pi / 4 + math.pi / 8:
                    ghost_approx_angle = 7  # WEST
                elif 3 * math.pi / 4 - math.pi / 8 <= ghost_angle <= 3 * math.pi / 4 + math.pi / 8:
                    ghost_approx_angle = 8  # NORTHWEST
            add_angle_direction_ohe(features, ghost_approx_angle, f"ghost-{i}-angle")
            # features[f"ghost-{i}-angle"] = ghost_angle
        if abs(g_x - x) + abs(g_y - y) <= 1:
            if is_scared:
                features["#-of-scared-ghosts-1-step-away"] += 1
            else:
                features["#-of-ghosts-1-step-away"] += 1

        cap_dist, cap_dir = closestCapsule(
            (x, y), state.getCapsules(), walls, legal_neighbor_cache=self.legal_neighbor_cache
        )
        features["closest-capsule-dist"] = cap_dist
        # features["closest-capsule-dir"] = cap_dir

        add_direction_ohe(features, cap_dir, "closest-capsule-dir")

        dist_dir = closestFood((x, y), food, walls, legal_neighbor_cache=self.legal_neighbor_cache, return_dir=True)
        if dist_dir is not None:
            dist, dir = dist_dir
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = dist
            add_direction_ohe(features, dir, "closest-food-dir")
        else:
            features["closest-food"] = max(self.width, self.height)
            # features["closest-food-dir"] = 6
            add_direction_ohe(features, Directions.STOP, "closest-food-dir")
        return features

    def get_obs_space(self, nr_ghosts):
        other_obs_size = 21
        # map_size = self.width * self.height
        obs_size = other_obs_size + nr_ghosts * 24
        low = np.zeros(obs_size)
        high = np.ones(obs_size) * max(self.width, self.height)
        # high[other_obs_size:] = 3 + 4 + 4
        return gymnasium.spaces.Box(low=low, high=high)


class EssentialInfoExtractor(FeatureExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.legal_neighbor_cache = dict()

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostStates()

        features = Counter()

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()

        x_int, y_int = int(x + 0.5), int(y + 0.5)
        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if next_x < walls.width and next_y < walls.height:
                if not walls[next_x][next_y]:
                    features[f"poss-dir-{dir}"] = 1
                else:
                    features[f"poss-dir-{dir}"] = 0
                    # possible.append(dir)

        features["#-of-ghosts-1-step-away"] = 0
        features["#-of-scared-ghosts-1-step-away"] = 0
        for i, g in enumerate(ghosts):
            is_scared = 1 if g.isScared() else 0
            g_x, g_y = g.getPosition()
            features[f"ghost-{i}-scared"] = is_scared
            features[f"ghost-{i}-scaredtime"] = g.scaredTimer / SCARED_TIME
            g_dist, g_dir = ghostDistance(
                (x, y), g.getPosition(), walls, legal_neighbor_cache=self.legal_neighbor_cache, return_dir=True
            )

            features[f"ghost-{i}-dist"] = g_dist
            # features[f"ghost-{i}-dir"] = g_dir if g_dir is not None else -1

            add_direction_ohe(features, g_dir if g_dir is not None else Directions.STOP, f"ghost-{i}-dir")
            add_direction_ohe(features, g.getDirection(), f"ghost-{i}-heading")

            if abs(g_x - x) + abs(g_y - y) <= 1:
                if is_scared:
                    features["#-of-scared-ghosts-1-step-away"] += 1
                else:
                    features["#-of-ghosts-1-step-away"] += 1

        cap_dist, cap_dir = closestCapsule(
            (x, y), state.getCapsules(), walls, legal_neighbor_cache=self.legal_neighbor_cache
        )
        features["closest-capsule-dist"] = cap_dist

        add_direction_ohe(features, cap_dir, "closest-capsule-dir")

        dist_dir = closestFood((x, y), food, walls, legal_neighbor_cache=self.legal_neighbor_cache, return_dir=True)
        if dist_dir is not None:
            dist, dir = dist_dir
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = dist
            add_direction_ohe(features, dir, "closest-food-dir")
        else:
            features["closest-food"] = max(self.width, self.height)
            # features["closest-food-dir"] = 6
            add_direction_ohe(features, Directions.STOP, "closest-food-dir")

        # features["map"] = constract_map_array(self.height,self.width,state)
        return features

    def get_obs_space(self, nr_ghosts):
        other_obs_size = 19
        # map_size = self.width * self.height
        obs_size = other_obs_size + nr_ghosts * 13
        low = np.zeros(obs_size)
        high = np.ones(obs_size) * max(self.width, self.height)
        # high[other_obs_size:] = 3 + 4 + 4
        return gymnasium.spaces.Box(low=low, high=high)


def add_direction_ohe(features, direction, feature_name, with_out_stop=False):
    for d in Actions._directions.keys():
        if with_out_stop and d == Directions.STOP:
            continue
        else:
            features[f"{feature_name}-{d}"] = d == direction


def add_angle_direction_ohe(features, angle, feature_name):
    for d in range(0, 9):
        features[f"{feature_name}-{d}"] = d == angle


def construct_map_array(height, width, game_state):
    map = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            food, walls = game_state.data.food, game_state.data.layout.walls
            if food[x][y]:
                map[x][y] = 2
            if walls[x][y]:
                map[x][y] = 3

    for agentState in game_state.data.agentStates:
        if agentState == None:
            continue
        if agentState.configuration == None:
            continue
        x, y = [int(i) for i in nearestPoint(agentState.configuration.pos)]
        agent_dir = agentState.configuration.direction
        if agentState.isPacman:
            map[x][y] = 3 + agent_dir
        else:
            map[x][y] = 3 + 4 + (agent_dir)

    for x, y in game_state.data.capsules:
        map[x][y] = 1
    return map


class NextActionExtractor(FeatureExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.legal_neighbor_cache = dict()

    def getFeatures(self, state, _action):
        food = state.getFood()
        x, y = state.getPacmanPosition()
        walls = state.getWalls()
        ghosts = state.getGhostStates()
        features = Counter()
        for action in Actions._directions.keys():
            if action == Directions.STOP:
                continue
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            features[f"#-of-ghosts-1-step-away-{action}"] = sum(
                (next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) for g in ghosts
            )
            features[f"#-of-scared-ghosts-1-step-away-{action}"] = sum(
                (next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) for g in ghosts if g.isScared()
            )

            if not features[f"#-of-ghosts-1-step-away-{action}"] and food[next_x][next_y]:
                features[f"eats-food-{action}"] = 1.0
            else:
                features[f"eats-food-{action}"] = 0.0

            dist = closestFood(
                (next_x, next_y), food, walls, legal_neighbor_cache=self.legal_neighbor_cache, return_dir=False
            )
            if dist is not None:
                # make the distance a number less than one otherwise the update
                # will diverge wildly
                features[f"closest-food-{action}"] = float(dist) / (walls.width + walls.height)
            else:
                features[f"closest-food-{action}"] = 1

        x_int, y_int = int(x + 0.5), int(y + 0.5)
        for dir, vec in Actions._directionsAsList:
            if dir != Directions.STOP:
                dx, dy = vec
                next_y = y_int + dy
                next_x = x_int + dx
                if next_x < walls.width and next_y < walls.height:
                    if not walls[next_x][next_y]:
                        features[f"poss-dir-{dir}"] = 1
                    else:
                        features[f"poss-dir-{dir}"] = 0
        return features

    def get_obs_space(self, nr_ghosts):
        obs_size = 20
        low = np.zeros(obs_size)
        high = np.ones(obs_size)
        high[:8] *= nr_ghosts
        return gymnasium.spaces.Box(low=low, high=high)


class EssentialInfoAndNextActionExtractor(FeatureExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.legal_neighbor_cache = dict()

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostStates()

        features = Counter()

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()

        dist_dir = closestFood((x, y), food, walls, legal_neighbor_cache=self.legal_neighbor_cache, return_dir=True)
        if dist_dir is not None:
            dist, food_dir = dist_dir
            features["closest-food"] = dist
            add_direction_ohe(features, food_dir, "closest-food-dir")
        else:
            features["closest-food"] = self.height + self.width
            add_direction_ohe(features, Directions.STOP, "closest-food-dir")
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if next_x < walls.width and next_y < walls.height:
                if not walls[next_x][next_y]:
                    features[f"poss-dir-{dir}"] = 1
                else:
                    features[f"poss-dir-{dir}"] = 0

        features["#-of-ghosts-1-step-away"] = 0
        features["#-of-scared-ghosts-1-step-away"] = 0
        for i, g in enumerate(ghosts):
            is_scared = 1 if g.isScared() else 0
            g_x, g_y = g.getPosition()
            features[f"ghost-{i}-scared"] = is_scared
            features[f"ghost-{i}-scaredtime"] = g.scaredTimer / SCARED_TIME
            g_dist, g_dir = ghostDistance(
                (x, y), g.getPosition(), walls, legal_neighbor_cache=self.legal_neighbor_cache, return_dir=True
            )

            features[f"ghost-{i}-dist"] = g_dist

            add_direction_ohe(features, g_dir if g_dir is not None else Directions.STOP, f"ghost-{i}-dir")
            add_direction_ohe(features, g.getDirection(), f"ghost-{i}-heading")
            if abs(g_x - x) + abs(g_y - y) <= 1:
                if is_scared:
                    features["#-of-scared-ghosts-1-step-away"] += 1
                else:
                    features["#-of-ghosts-1-step-away"] += 1
        for action in Actions._directions.keys():
            if action == Directions.STOP:
                continue
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            features[f"#-of-ghosts-1-step-away-{action}"] = (
                sum((next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) for g in ghosts)
                - features["#-of-ghosts-1-step-away"]
            )
            features[f"#-of-scared-ghosts-1-step-away-{action}"] = (
                sum(
                    (next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls)
                    for g in ghosts
                    if g.isScared()
                )
                - features["#-of-scared-ghosts-1-step-away"]
            )

            if not features[f"#-of-ghosts-1-step-away-{action}"] and food[next_x][next_y]:
                features[f"eats-food-{action}"] = 1.0
            else:
                features[f"eats-food-{action}"] = 0.0

            dist = closestFood(
                (next_x, next_y), food, walls, legal_neighbor_cache=self.legal_neighbor_cache, return_dir=False
            )
            if dist is not None:
                # make the distance a number less than one otherwise the update
                # will diverge wildly
                features[f"closest-food-{action}"] = dist - features["closest-food"]
            else:
                features[f"closest-food-{action}"] = self.height + self.width

        cap_dist, cap_dir = closestCapsule(
            (x, y), state.getCapsules(), walls, legal_neighbor_cache=self.legal_neighbor_cache
        )
        features["closest-capsule-dist"] = cap_dist

        add_direction_ohe(features, cap_dir, "closest-capsule-dir")

        features["x"] = x
        features["y"] = y
        return features

    def get_obs_space(self, nr_ghosts):
        other_obs_size = 37
        # map_size = self.width * self.height
        obs_size = other_obs_size + nr_ghosts * 13
        low = np.zeros(obs_size)
        high = np.ones(obs_size) * max(self.width, self.height)
        # high[other_obs_size:] = 3 + 4 + 4
        return gymnasium.spaces.Box(low=low, high=high)


class DeepRLCompleteExtractor(FeatureExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.legal_neighbor_cache = dict()

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostStates()
        n_ghosts = len(ghosts)
        features = Counter()
        max_dist = self.height + self.width

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()

        dist_dir = closestFood((x, y), food, walls, legal_neighbor_cache=self.legal_neighbor_cache, return_dir=True)
        if dist_dir is not None:
            dist, food_dir = dist_dir
            features["closest-food"] = dist / max_dist
            add_direction_ohe(features, food_dir, "closest-food-dir", with_out_stop=True)
        else:
            features["closest-food"] = 1
            add_direction_ohe(features, Directions.STOP, "closest-food-dir", with_out_stop=True)
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        for dir, vec in Actions._directionsAsList:
            if dir == Directions.STOP:
                continue
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if next_x < walls.width and next_y < walls.height:
                if not walls[next_x][next_y]:
                    features[f"poss-dir-{dir}"] = 1
                else:
                    features[f"poss-dir-{dir}"] = 0

        ghost_distances = defaultdict(list)
        for i, g in enumerate(ghosts):
            is_scared = 1 if g.isScared() else 0
            features[f"ghost-{i}-scared"] = is_scared
            features[f"ghost-{i}-scaredtime"] = g.scaredTimer / SCARED_TIME
            g_dist, g_dir = ghostDistance(
                (x, y), g.getPosition(), walls, legal_neighbor_cache=self.legal_neighbor_cache, return_dir=True
            )
            ghost_distances["curr"].append((g_dist, g.isScared()))
            features[f"ghost-{i}-dist"] = g_dist / max_dist

            add_direction_ohe(
                features, g_dir if g_dir is not None else Directions.STOP, f"ghost-{i}-dir", with_out_stop=True
            )
            add_direction_ohe(features, g.getDirection(), f"ghost-{i}-heading", with_out_stop=True)
            for action in Actions._directions.keys():
                if action == Directions.STOP:
                    continue
                dx, dy = Actions.directionToVector(action)
                next_x, next_y = int(x + dx), int(y + dy)

                g_dist, g_dir = ghostDistance(
                    (next_x, next_y),
                    g.getPosition(),
                    walls,
                    legal_neighbor_cache=self.legal_neighbor_cache,
                    return_dir=True,
                )
                ghost_distances[action].append((g_dist, g.isScared()))

        features["#-of-non-scared-ghosts-1-step-away"] = len(
            [(d, sc) for (d, sc) in ghost_distances["curr"] if not sc and d <= 1]
        )
        features["#-of-scared-ghosts-1-step-away"] = len(
            [(d, sc) for (d, sc) in ghost_distances["curr"] if sc and d <= 1]
        )
        features["#-of-non-scared-ghosts-le3-step-away"] = len(
            [(d, sc) for (d, sc) in ghost_distances["curr"] if not sc and d <= 3]
        )
        features["#-of-scared-ghosts-le3-step-away"] = len(
            [(d, sc) for (d, sc) in ghost_distances["curr"] if sc and d <= 3]
        )

        for action in Actions._directions.keys():
            if action == Directions.STOP:
                continue

            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            features[f"#-of-non-scared-ghosts-1-step-away-{action}"] = (
                len([(d, sc) for (d, sc) in ghost_distances[action] if not sc and d <= 1])
                - features["#-of-non-scared-ghosts-1-step-away"]
            ) / n_ghosts
            features[f"#-of-scared-ghosts-1-step-away-{action}"] = (
                len([(d, sc) for (d, sc) in ghost_distances[action] if sc and d <= 1])
                - features["#-of-scared-ghosts-1-step-away"]
            ) / n_ghosts
            features[f"#-of-non-scared-ghosts-le3-step-away-{action}"] = (
                len([(d, sc) for (d, sc) in ghost_distances[action] if not sc and d <= 3])
                - features["#-of-non-scared-ghosts-le3-step-away"]
            ) / n_ghosts
            features[f"#-of-scared-ghosts-le3-step-away-{action}"] = (
                len([(d, sc) for (d, sc) in ghost_distances[action] if sc and d <= 3])
                - features["#-of-scared-ghosts-le3-step-away"]
            ) / n_ghosts

            if action == Directions.STOP:
                continue
            dist = closestFood(
                (next_x, next_y), food, walls, legal_neighbor_cache=self.legal_neighbor_cache, return_dir=False
            )
            if dist is not None:
                # make the distance a number less than one otherwise the update
                # will diverge wildly
                features[f"closest-food-{action}"] = (dist - features["closest-food"]) / max_dist
            else:
                features[f"closest-food-{action}"] = 1  # self.height + self.width

        features["#-of-non-scared-ghosts-1-step-away"] /= n_ghosts
        features["#-of-scared-ghosts-1-step-away"] /= n_ghosts
        features["#-of-non-scared-ghosts-le3-step-away"] /= n_ghosts
        features["#-of-scared-ghosts-le3-step-away"] /= n_ghosts
        cap_dist, cap_dir = closestCapsule(
            (x, y), state.getCapsules(), walls, legal_neighbor_cache=self.legal_neighbor_cache
        )
        features["closest-capsule-dist"] = cap_dist

        add_direction_ohe(features, cap_dir, "closest-capsule-dir", with_out_stop=True)

        features["x"] = x / self.width
        features["y"] = y / self.height

        # ADDED PROVISIONALLY
        features["score"] = float(state.getScore()) / 100.0

        return features

    def get_obs_space(self, nr_ghosts):
        other_obs_size = 40
        obs_size = other_obs_size + nr_ghosts * 11 + 1
        low = np.zeros(obs_size)
        high = np.ones(obs_size)
        return gymnasium.spaces.Box(low=low, high=high)


class LabelExtractor(FeatureExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()

    def getFeatures(self, state, action):
        features = Counter()
        for l in self.labels.labels:
            fn = getattr(self.labels, l)
            if fn(state, action):
                features[l] = 1.0
            else:
                features[1] = 0.0

    def get_obs_space(self, nr_ghosts):
        other_obs_size = len(self.labels.labels)
        obs_size = other_obs_size + nr_ghosts * 11
        low = np.zeros(obs_size)
        high = np.ones(obs_size)
        return gymnasium.spaces.Box(low=low, high=high)


class DeepRLLabelledCompleteExtractor(FeatureExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.legal_neighbor_cache = dict()
        self.labels = Labels()

    def getFeatures(self, state, action, final=None):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostStates()
        n_ghosts = len(ghosts)
        features = Counter()
        max_dist = self.height + self.width
        if final is None:
            final = []

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()

        dist_dir = closestFood((x, y), food, walls, legal_neighbor_cache=self.legal_neighbor_cache, return_dir=True)
        if dist_dir is not None:
            dist, food_dir = dist_dir
            features["closest-food"] = dist / max_dist
            add_direction_ohe(features, food_dir, "closest-food-dir", with_out_stop=True)
        else:
            features["closest-food"] = 1
            add_direction_ohe(features, Directions.STOP, "closest-food-dir", with_out_stop=True)
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        for dir, vec in Actions._directionsAsList:
            if dir == Directions.STOP:
                continue
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if next_x < walls.width and next_y < walls.height:
                if not walls[next_x][next_y]:
                    features[f"poss-dir-{dir}"] = 1
                else:
                    features[f"poss-dir-{dir}"] = 0

        ghost_distances = defaultdict(list)
        for i, g in enumerate(ghosts):
            is_scared = 1 if g.isScared() else 0
            features[f"ghost-{i}-scared"] = is_scared
            features[f"ghost-{i}-scaredtime"] = g.scaredTimer / SCARED_TIME
            g_dist, g_dir = ghostDistance(
                (x, y), g.getPosition(), walls, legal_neighbor_cache=self.legal_neighbor_cache, return_dir=True
            )
            ghost_distances["curr"].append((g_dist, g.isScared()))
            features[f"ghost-{i}-dist"] = g_dist / max_dist

            add_direction_ohe(
                features, g_dir if g_dir is not None else Directions.STOP, f"ghost-{i}-dir", with_out_stop=True
            )
            add_direction_ohe(features, g.getDirection(), f"ghost-{i}-heading", with_out_stop=True)
            for action in Actions._directions.keys():
                if action == Directions.STOP:
                    continue
                dx, dy = Actions.directionToVector(action)
                next_x, next_y = int(x + dx), int(y + dy)

                g_dist, g_dir = ghostDistance(
                    (next_x, next_y),
                    g.getPosition(),
                    walls,
                    legal_neighbor_cache=self.legal_neighbor_cache,
                    return_dir=True,
                )
                ghost_distances[action].append((g_dist, g.isScared()))

        features["#-of-non-scared-ghosts-1-step-away"] = len(
            [(d, sc) for (d, sc) in ghost_distances["curr"] if not sc and d <= 1]
        )
        features["#-of-scared-ghosts-1-step-away"] = len(
            [(d, sc) for (d, sc) in ghost_distances["curr"] if sc and d <= 1]
        )
        features["#-of-non-scared-ghosts-le3-step-away"] = len(
            [(d, sc) for (d, sc) in ghost_distances["curr"] if not sc and d <= 3]
        )
        features["#-of-scared-ghosts-le3-step-away"] = len(
            [(d, sc) for (d, sc) in ghost_distances["curr"] if sc and d <= 3]
        )

        for action in Actions._directions.keys():
            if action == Directions.STOP:
                continue

            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            features[f"#-of-non-scared-ghosts-1-step-away-{action}"] = (
                len([(d, sc) for (d, sc) in ghost_distances[action] if not sc and d <= 1])
                - features["#-of-non-scared-ghosts-1-step-away"]
            ) / n_ghosts
            features[f"#-of-scared-ghosts-1-step-away-{action}"] = (
                len([(d, sc) for (d, sc) in ghost_distances[action] if sc and d <= 1])
                - features["#-of-scared-ghosts-1-step-away"]
            ) / n_ghosts
            features[f"#-of-non-scared-ghosts-le3-step-away-{action}"] = (
                len([(d, sc) for (d, sc) in ghost_distances[action] if not sc and d <= 3])
                - features["#-of-non-scared-ghosts-le3-step-away"]
            ) / n_ghosts
            features[f"#-of-scared-ghosts-le3-step-away-{action}"] = (
                len([(d, sc) for (d, sc) in ghost_distances[action] if sc and d <= 3])
                - features["#-of-scared-ghosts-le3-step-away"]
            ) / n_ghosts

            if action == Directions.STOP:
                continue
            dist = closestFood(
                (next_x, next_y), food, walls, legal_neighbor_cache=self.legal_neighbor_cache, return_dir=False
            )
            if dist is not None:
                # make the distance a number less than one otherwise the update
                # will diverge wildly
                features[f"closest-food-{action}"] = (dist - features["closest-food"]) / max_dist
            else:
                features[f"closest-food-{action}"] = 1  # self.height + self.width

        features["#-of-non-scared-ghosts-1-step-away"] /= n_ghosts
        features["#-of-scared-ghosts-1-step-away"] /= n_ghosts
        features["#-of-non-scared-ghosts-le3-step-away"] /= n_ghosts
        features["#-of-scared-ghosts-le3-step-away"] /= n_ghosts
        cap_dist, cap_dir = closestCapsule(
            (x, y), state.getCapsules(), walls, legal_neighbor_cache=self.legal_neighbor_cache
        )
        features["closest-capsule-dist"] = cap_dist

        add_direction_ohe(features, cap_dir, "closest-capsule-dir", with_out_stop=True)

        features["x"] = x / self.width
        features["y"] = y / self.height

        for l in self.labels.labels:
            fn = getattr(self.labels, l)
            if fn(state, action):
                features[l] = 1.0
            else:
                features[l] = 0.0
        i = 1
        for fin in final:
            features["violation-dfa-" + str(i)] = fin
            i *= 1

        return features

    def get_obs_space(self, nr_ghosts):
        other_obs_size = 40 + len(self.labels.labels)
        obs_size = other_obs_size + nr_ghosts * 11
        low = np.zeros(obs_size)
        high = np.ones(obs_size)
        return gymnasium.spaces.Box(low=low, high=high)


class DeepRLDFACompleteExtractor(FeatureExtractor):
    def __init__(self, height, width, dfa_dict=None):
        super().__init__(height, width)
        if dfa_dict is None:
            dfa_dict = {}
        self.legal_neighbor_cache = dict()
        self.labels = Labels()
        self.dfa_list = []
        for dfa in dfa_dict.keys():
            dfa_class = getattr(pdfa, dfa)
            dfa_obj = dfa_class(self.labels, reward=dfa_dict[dfa])
            self.dfa_list.append(dfa_obj)

    def getFeatures(self, state, action, final=None):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostStates()
        n_ghosts = len(ghosts)
        features = Counter()
        max_dist = self.height + self.width
        if final is None:
            final = []

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()

        dist_dir = closestFood((x, y), food, walls, legal_neighbor_cache=self.legal_neighbor_cache, return_dir=True)
        if dist_dir is not None:
            dist, food_dir = dist_dir
            features["closest-food"] = dist / max_dist
            add_direction_ohe(features, food_dir, "closest-food-dir", with_out_stop=True)
        else:
            features["closest-food"] = 1
            add_direction_ohe(features, Directions.STOP, "closest-food-dir", with_out_stop=True)
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        for dir, vec in Actions._directionsAsList:
            if dir == Directions.STOP:
                continue
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if next_x < walls.width and next_y < walls.height:
                if not walls[next_x][next_y]:
                    features[f"poss-dir-{dir}"] = 1
                else:
                    features[f"poss-dir-{dir}"] = 0

        ghost_distances = defaultdict(list)
        for i, g in enumerate(ghosts):
            is_scared = 1 if g.isScared() else 0
            features[f"ghost-{i}-scared"] = is_scared
            features[f"ghost-{i}-scaredtime"] = g.scaredTimer / SCARED_TIME
            g_dist, g_dir = ghostDistance(
                (x, y), g.getPosition(), walls, legal_neighbor_cache=self.legal_neighbor_cache, return_dir=True
            )
            ghost_distances["curr"].append((g_dist, g.isScared()))
            features[f"ghost-{i}-dist"] = g_dist / max_dist

            add_direction_ohe(
                features, g_dir if g_dir is not None else Directions.STOP, f"ghost-{i}-dir", with_out_stop=True
            )
            add_direction_ohe(features, g.getDirection(), f"ghost-{i}-heading", with_out_stop=True)
            for action in Actions._directions.keys():
                if action == Directions.STOP:
                    continue
                dx, dy = Actions.directionToVector(action)
                next_x, next_y = int(x + dx), int(y + dy)

                g_dist, g_dir = ghostDistance(
                    (next_x, next_y),
                    g.getPosition(),
                    walls,
                    legal_neighbor_cache=self.legal_neighbor_cache,
                    return_dir=True,
                )
                ghost_distances[action].append((g_dist, g.isScared()))

        features["#-of-non-scared-ghosts-1-step-away"] = len(
            [(d, sc) for (d, sc) in ghost_distances["curr"] if not sc and d <= 1]
        )
        features["#-of-scared-ghosts-1-step-away"] = len(
            [(d, sc) for (d, sc) in ghost_distances["curr"] if sc and d <= 1]
        )
        features["#-of-non-scared-ghosts-le3-step-away"] = len(
            [(d, sc) for (d, sc) in ghost_distances["curr"] if not sc and d <= 3]
        )
        features["#-of-scared-ghosts-le3-step-away"] = len(
            [(d, sc) for (d, sc) in ghost_distances["curr"] if sc and d <= 3]
        )

        for action in Actions._directions.keys():
            if action == Directions.STOP:
                continue

            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            features[f"#-of-non-scared-ghosts-1-step-away-{action}"] = (
                len([(d, sc) for (d, sc) in ghost_distances[action] if not sc and d <= 1])
                - features["#-of-non-scared-ghosts-1-step-away"]
            ) / n_ghosts
            features[f"#-of-scared-ghosts-1-step-away-{action}"] = (
                len([(d, sc) for (d, sc) in ghost_distances[action] if sc and d <= 1])
                - features["#-of-scared-ghosts-1-step-away"]
            ) / n_ghosts
            features[f"#-of-non-scared-ghosts-le3-step-away-{action}"] = (
                len([(d, sc) for (d, sc) in ghost_distances[action] if not sc and d <= 3])
                - features["#-of-non-scared-ghosts-le3-step-away"]
            ) / n_ghosts
            features[f"#-of-scared-ghosts-le3-step-away-{action}"] = (
                len([(d, sc) for (d, sc) in ghost_distances[action] if sc and d <= 3])
                - features["#-of-scared-ghosts-le3-step-away"]
            ) / n_ghosts

            if action == Directions.STOP:
                continue
            dist = closestFood(
                (next_x, next_y), food, walls, legal_neighbor_cache=self.legal_neighbor_cache, return_dir=False
            )
            if dist is not None:
                # make the distance a number less than one otherwise the update
                # will diverge wildly
                features[f"closest-food-{action}"] = (dist - features["closest-food"]) / max_dist
            else:
                features[f"closest-food-{action}"] = 1  # self.height + self.width

        features["#-of-non-scared-ghosts-1-step-away"] /= n_ghosts
        features["#-of-scared-ghosts-1-step-away"] /= n_ghosts
        features["#-of-non-scared-ghosts-le3-step-away"] /= n_ghosts
        features["#-of-scared-ghosts-le3-step-away"] /= n_ghosts
        cap_dist, cap_dir = closestCapsule(
            (x, y), state.getCapsules(), walls, legal_neighbor_cache=self.legal_neighbor_cache
        )
        features["closest-capsule-dist"] = cap_dist

        add_direction_ohe(features, cap_dir, "closest-capsule-dir", with_out_stop=True)

        features["x"] = x / self.width
        features["y"] = y / self.height

        # ADDED PROVISIONALLY
        features["score"] = float(state.getScore()) / 100.0

        i = 1
        for dfa in self.dfa_list:
            inpt = self.labels.getLabels(state, action)
            st = dfa.transition(inpt)
            for q in dfa.states:
                if q == st:
                    features["DFA-" + str(i) + "-state-" + str(q)] = 1.0
                else:
                    features["DFA-" + str(i) + "-state-" + str(q)] = 0.0
            i += 1

        return features

    def get_obs_space(self, nr_ghosts):
        states = 0
        for dfa in self.dfa_list:
            states += len(dfa.states)
        other_obs_size = 40 + states + 1
        obs_size = other_obs_size + nr_ghosts * 11
        low = np.zeros(obs_size)
        high = np.ones(obs_size)
        return gymnasium.spaces.Box(low=low, high=high)

    def get_nr_dfa_states(self):
        return sum([len(dfa.states) for dfa in self.dfa_list])


class DeepRLDFACompleteExtractorFine(FeatureExtractor):
    def __init__(self, height, width, dfa_dict=None):
        super().__init__(height, width)
        if dfa_dict is None:
            dfa_dict = {}
        self.legal_neighbor_cache = dict()
        self.labels = Labels()
        self.dfa_list = []
        for dfa in dfa_dict.keys():
            dfa_class = getattr(pdfa, dfa)
            dfa_obj = dfa_class(self.labels, reward=dfa_dict[dfa])
            self.dfa_list.append(dfa_obj)

    def getFeatures(self, state, action, final=None):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostStates()
        n_ghosts = len(ghosts)
        features = Counter()
        max_dist = self.height + self.width
        if final is None:
            final = []

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()

        dist_dir = closestFood((x, y), food, walls, legal_neighbor_cache=self.legal_neighbor_cache, return_dir=True)
        if dist_dir is not None:
            dist, food_dir = dist_dir
            features["closest-food"] = dist / max_dist
            add_direction_ohe(features, food_dir, "closest-food-dir", with_out_stop=True)
        else:
            features["closest-food"] = 1
            add_direction_ohe(features, Directions.STOP, "closest-food-dir", with_out_stop=True)
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        for dir, vec in Actions._directionsAsList:
            if dir == Directions.STOP:
                continue
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if next_x < walls.width and next_y < walls.height:
                if not walls[next_x][next_y]:
                    features[f"poss-dir-{dir}"] = 1
                else:
                    features[f"poss-dir-{dir}"] = 0

        ghost_distances = defaultdict(list)
        for i, g in enumerate(ghosts):
            is_scared = 1 if g.isScared() else 0
            features[f"ghost-{i}-scared"] = is_scared
            features[f"ghost-{i}-scaredtime"] = g.scaredTimer / SCARED_TIME
            g_dist, g_dir = ghostDistance(
                (x, y), g.getPosition(), walls, legal_neighbor_cache=self.legal_neighbor_cache, return_dir=True
            )
            ghost_distances["curr"].append((g_dist, g.isScared()))
            features[f"ghost-{i}-dist"] = g_dist / max_dist

            add_direction_ohe(
                features, g_dir if g_dir is not None else Directions.STOP, f"ghost-{i}-dir", with_out_stop=True
            )
            add_direction_ohe(features, g.getDirection(), f"ghost-{i}-heading", with_out_stop=True)

            features[f"ghost-{i}-non-scared-1-step-away"] = 1 if not is_scared and g_dist <= 1 else 0
            features[f"ghost-{i}-scared-1-step-away"] = 1 if is_scared and g_dist <= 1 else 0
            features[f"ghost-{i}-non-scared-le3-step-away"] = 1 if not is_scared and g_dist <= 3 else 0
            features[f"ghost-{i}-scared-le3-step-away"] = 1 if is_scared and g_dist <= 1 else 0

            for action in Actions._directions.keys():
                if action == Directions.STOP:
                    continue
                dx, dy = Actions.directionToVector(action)
                next_x, next_y = int(x + dx), int(y + dy)

                g_dist, g_dir = ghostDistance(
                    (next_x, next_y),
                    g.getPosition(),
                    walls,
                    legal_neighbor_cache=self.legal_neighbor_cache,
                    return_dir=True,
                )
                ghost_distances[action].append((g_dist, g.isScared()))

                features[f"ghost-{i}-non-scared-1-step-away-{action}"] = 1 if not is_scared and g_dist <= 1 else 0
                features[f"ghost-{i}-scared-1-step-away-{action}"] = 1 if is_scared and g_dist <= 1 else 0
                features[f"ghost-{i}-non-scared-le3-step-away-{action}"] = 1 if not is_scared and g_dist <= 3 else 0
                features[f"ghost-{i}-scared-le3-step-away-{action}"] = 1 if is_scared and g_dist <= 3 else 0

        for action in Actions._directions.keys():
            if action == Directions.STOP:
                continue

            dist = closestFood(
                (next_x, next_y), food, walls, legal_neighbor_cache=self.legal_neighbor_cache, return_dir=False
            )
            if dist is not None:
                # make the distance a number less than one otherwise the update
                # will diverge wildly
                features[f"closest-food-{action}"] = (dist - features["closest-food"]) / max_dist
            else:
                features[f"closest-food-{action}"] = 1  # self.height + self.width

        # features["#-of-non-scared-ghosts-1-step-away"] /= n_ghosts
        # features["#-of-scared-ghosts-1-step-away"] /= n_ghosts
        # features["#-of-non-scared-ghosts-le3-step-away"] /= n_ghosts
        # features["#-of-scared-ghosts-le3-step-away"] /= n_ghosts
        cap_dist, cap_dir = closestCapsule(
            (x, y), state.getCapsules(), walls, legal_neighbor_cache=self.legal_neighbor_cache
        )
        features["closest-capsule-dist"] = cap_dist

        add_direction_ohe(features, cap_dir, "closest-capsule-dir", with_out_stop=True)

        features["x"] = x / self.width
        features["y"] = y / self.height

        # ADDED PROVISIONALLY
        features["score"] = float(state.getScore()) / 100.0

        i = 1
        for dfa in self.dfa_list:
            inpt = self.labels.getLabels(state, action)
            st = dfa.transition(inpt)
            for q in dfa.states:
                if q == st:
                    features["DFA-" + str(i) + "-state-" + str(q)] = 1.0
                else:
                    features["DFA-" + str(i) + "-state-" + str(q)] = 0.0
            i += 1

        return features

    def get_obs_space(self, nr_ghosts):
        states = 0
        for dfa in self.dfa_list:
            states += len(dfa.states)
        other_obs_size = 60 + states + 1
        obs_size = other_obs_size + nr_ghosts * 11
        low = np.zeros(obs_size)
        high = np.ones(obs_size)
        return gymnasium.spaces.Box(low=low, high=high)

    def get_nr_dfa_states(self):
        return sum([len(dfa.states) for dfa in self.dfa_list])


# --------------------------OLD DFA CODE


class DeepRLVeganCompleteExtractor(DeepRLDFACompleteExtractor):
    def __init__(self, height, width, rewards):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = VegBlueDFA(self.labels, reward=1500.0)
        dfa2 = VegOrangeDFA(self.labels, reward=1500.0)
        self.dfa_list = [dfa1, dfa2]


class DeepRLVeganPrefCompleteExtractor(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = VegBlueDFA(self.labels, reward=1000.0)
        dfa2 = VegOrangeDFA(self.labels, reward=6000.0)
        self.dfa_list = [dfa1, dfa2]


class DeepRLVegetarianPermCompleteExtractor(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = VegBlueDFA(self.labels, reward=1500.0)
        dfa2 = VegOrangeDFA(self.labels, reward=1500.0)
        dfa3 = PermBlueDFA(self.labels, reward=1500.0)
        self.dfa_list = [dfa1, dfa2, dfa3]


class DeepRLPassiveCompleteExtractor(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = VegBlueDFA(self.labels, reward=1000.0)
        dfa2 = VegOrangeDFA(self.labels, reward=1000.0)
        dfa3 = CTDBlueDFA(self.labels, reward=4000.0)
        dfa4 = CTDOrangeDFA(self.labels, reward=4000.0)
        self.dfa_list = [dfa1, dfa2, dfa3, dfa4]


class DeepRLHighScoreCompleteExtractor(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = TrappedDFA(self.labels, reward=100.0)
        self.dfa_list = [dfa1]


class DeepRLEarlyBirdCompleteExtractor1(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = EarlyBirdDFA1(self.labels, reward=1000.0)
        self.dfa_list = [dfa1]


class DeepRLContradictionCompleteExtractor(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = VegBlueDFA(self.labels, reward=500.0)
        dfa2 = VegOrangeDFA(self.labels, reward=500.0)
        dfa3 = EarlyBirdDFA1(self.labels, reward=5000.0)
        self.dfa_list = [dfa1, dfa2, dfa3]


class DeepRLSolutionCompleteExtractor(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = VegBlueDFA(self.labels, reward=500.0)
        dfa2 = VegOrangeDFA(self.labels, reward=500.0)
        dfa3 = EarlyBirdDFA1(self.labels, reward=5000.0)
        dfa4 = PermBlueDFA(self.labels, reward=500.0)
        self.dfa_list = [dfa1, dfa2, dfa3, dfa4]


class DeepRLEarlyBirdCompleteExtractor2(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = EarlyBirdDFA2(self.labels, reward=5000.0)
        self.dfa_list = [dfa1]


class DeepRLEarlyBirdCompleteExtractor3(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = EarlyBirdFulfillmentDFA(self.labels, reward=1000.0)
        self.dfa_list = [dfa1]


class DeepRLEarlyBirdCompleteExtractor4(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = EarlyBirdFulfillmentDFA2(self.labels, reward=1000.0)
        self.dfa_list = [dfa1]


class DeepRLContradictionCompleteExtractor2(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = VegBlueDFA(self.labels, reward=1000.0)
        dfa2 = VegOrangeDFA(self.labels, reward=1000.0)
        dfa3 = EarlyBirdFulfillmentDFA(self.labels, reward=4000.0)
        self.dfa_list = [dfa1, dfa2, dfa3]


class DeepRLVeganConflictCompleteExtractor(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = VegBlueDFA(self.labels, reward=1000.0)
        dfa2 = VegOrangeDFA(self.labels, reward=1000.0)
        dfa3 = OblBlueDFA(self.labels, reward=3000.00)
        self.dfa_list = [dfa1, dfa2, dfa3]


class DeepRLErrandCompleteExtractor1a(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = ErrandDFA1a(self.labels, reward=1000.0)
        self.dfa_list = [dfa1]


class DeepRLErrandCompleteExtractor1b(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = ErrandDFA1b(self.labels, reward=1000.0)
        self.dfa_list = [dfa1]


class DeepRLErrandCompleteExtractor2a(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = ErrandDFA2a(self.labels, reward=1000.0)
        self.dfa_list = [dfa1]


class DeepRLErrandCompleteExtractor2b(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = ErrandDFA2b(self.labels, reward=1000.0)
        self.dfa_list = [dfa1]


class DeepRLVeganSimpleExtractor(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = VegBlueDFA(self.labels, reward=20.0)
        dfa2 = VegOrangeDFA(self.labels, reward=20.0)
        self.dfa_list = [dfa1, dfa2]


class DeepRLVeganPrefSimpleExtractor(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = VegBlueDFA(self.labels, reward=20.0)
        dfa2 = VegOrangeDFA(self.labels, reward=40.0)
        self.dfa_list = [dfa1, dfa2]


class DeepRLVegetarianPermSimpleExtractor(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = VegBlueDFA(self.labels, reward=20.0)
        dfa2 = VegOrangeDFA(self.labels, reward=20.0)
        dfa3 = PermBlueDFA(self.labels, reward=20.0)
        self.dfa_list = [dfa1, dfa2, dfa3]


class DeepRLPassiveSimpleExtractor(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = VegBlueDFA(self.labels, reward=20.0)
        dfa2 = VegOrangeDFA(self.labels, reward=20.0)
        dfa3 = CTDBlueDFA(self.labels, reward=50.0)
        dfa4 = CTDOrangeDFA(self.labels, reward=50.0)
        self.dfa_list = [dfa1, dfa2, dfa3, dfa4]


class DeepRLHighScoreSimpleExtractor(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = TrappedDFA(self.labels, reward=10.0)
        self.dfa_list = [dfa1]


class DeepRLEarlyBirdSimpleExtractor1(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = EarlyBirdDFA1(self.labels, reward=20.0)
        self.dfa_list = [dfa1]


class DeepRLContradictionSimpleExtractor(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = VegBlueDFA(self.labels, reward=20.0)
        dfa2 = VegOrangeDFA(self.labels, reward=20.0)
        dfa3 = EarlyBirdDFA1(self.labels, reward=50.0)
        self.dfa_list = [dfa1, dfa2, dfa3]


class DeepRLSolutionSimpleExtractor(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = VegBlueDFA(self.labels, reward=20.0)
        dfa2 = VegOrangeDFA(self.labels, reward=20.0)
        dfa3 = EarlyBirdDFA1(self.labels, reward=50.0)
        dfa4 = PermBlueDFA(self.labels, reward=20.0)
        self.dfa_list = [dfa1, dfa2, dfa3, dfa4]


class DeepRLEarlyBirdSimpleExtractor3(DeepRLDFACompleteExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.labels = Labels()
        dfa1 = EarlyBirdFulfillmentDFA(self.labels, reward=20.0)
        self.dfa_list = [dfa1]
