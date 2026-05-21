from collections import deque
import numpy as np

coords = {"red": (0, 0), "green": (4, 0), "yellow": (0, 4), "blue": (3, 4)}

colors = {0: "red", 1: "green", 2: "yellow", 3: "blue"}

act_dict = {0: "south", 1: "north", 2: "east", 3: "west", 4: "pickup", 5: "dropoff", 6: "stall"}


MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]


def shortest_route(coord1, coord2):
    rows, cols = 5, 5
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    queue = deque([(coord1, [coord1])])

    def in_bounds(x, y):
        return 0 <= x < rows and 0 <= y < cols

    while len(queue) > 0:
        (x, y), path = queue.popleft()
        if not in_bounds(x, y) or MAP[y + 1][2 * x + 1] in ["+", "-", "|"] or visited[x][y]:
            continue
        visited[x][y] = True
        if (x, y) == coord2:
            break
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if in_bounds(nx, ny) and not visited[nx][ny]:
                if (dx > 0 and MAP[ny + 1][2 * nx] != "|") or (dx < 0 and MAP[ny + 1][2 * nx + 2] != "|") or dx == 0:
                    queue.append(((nx, ny), path + [(nx, ny)]))
    return len(path) - 1


def toward(state, action, obj):
    cur_xy = (get_x(state), get_y(state))
    # cur_xy = state
    dist_now = shortest_route(cur_xy, obj)
    if action == 0:
        new_xy = (cur_xy[0], min(4, cur_xy[1] + 1))
    elif action == 1:
        new_xy = (cur_xy[0], max(0, cur_xy[1] - 1))
    elif action == 2:
        new_xy = (min(4, cur_xy[0] + 1), cur_xy[1])
        if MAP[new_xy[1] + 1][2 * new_xy[0]] == "|":
            new_xy = cur_xy
    elif action == 3:
        new_xy = (min(4, cur_xy[0] - 1), cur_xy[1])
        if MAP[new_xy[1] + 1][2 * new_xy[0] + 2] == "|":
            new_xy = cur_xy
    else:
        new_xy = cur_xy
    dist_a = shortest_route(new_xy, obj)
    return dist_now > dist_a


def get_shelter(state):
    return state % 4


def get_floodrisk(state):
    state = state // 4
    return state % 2


def get_home(state):
    state = state // 4
    state = state // 2
    return state % 4


def get_hurricane(state):
    state = state // 4
    state = state // 2
    state = state // 4
    return state % 11


def get_rain(state):
    state = state // 4
    state = state // 2
    state = state // 4
    state = state // 11
    return state % 2


def get_dest(state):
    state = state // 4
    state = state // 2
    state = state // 4
    state = state // 11
    state = state // 2
    return state % 4


def get_passenger(state):
    state = state // 4
    state = state // 2
    state = state // 4
    state = state // 11
    state = state // 2
    state = state // 4
    return state % 5


def get_x(state):
    state = state // 4
    state = state // 2
    state = state // 4
    state = state // 11
    state = state // 2
    state = state // 4
    state = state // 5
    return state % 5


def get_y(state):
    state = state // 4
    state = state // 2
    state = state // 4
    state = state // 11
    state = state // 2
    state = state // 4
    state = state // 5
    state = state // 5
    return state


class Labels:
    def __init__(self, constitutive=False):
        self.c = constitutive
        self.labels = ["hurricane", "floodrisk", "rain", "atHome", "atShelter", "atDestination", "hasPassenger"]
        self.act_dict = {0: "south", 1: "north", 2: "east", 3: "west", 4: "pickup", 5: "dropoff", 6: "warn"}

    def warn(self, state, action):
        return action == 6

    def isEmergency(self, state, action):
        if self.c:
            pass
        else:
            return self.hurricane(state, action)

    def isRisk(self, state, action):
        if self.c:
            pass
        else:
            return self.rain(state, action)

    def floodrisk(self, state, action):
        return get_floodrisk(state) == 1

    def atHome(self, state, action):
        coord = (get_x(state), get_y(state))
        if action == 0:
            coord = (get_x(state), get_y(state) + 1)
        if action == 1:
            coord = (get_x(state), get_y(state) - 1)
        if action == 2:
            coord = (get_x(state) + 1, get_y(state))
        if action == 3:
            coord = (get_x(state) - 1, get_y(state))
        return coords[colors[get_home(state)]] == coord

    def atSafety(self, state, action):
        return (self.atHome(state, action) and not self.floodrisk(state, action)) or self.atShelter(state, action)

    def atShelter(self, state, action):
        coord = (get_x(state), get_y(state))
        if action == 0:
            coord = (get_x(state), get_y(state) + 1)
        if action == 1:
            coord = (get_x(state), get_y(state) - 1)
        if action == 2:
            coord = (get_x(state) + 1, get_y(state))
        if action == 3:
            coord = (get_x(state) - 1, get_y(state))
        return coords[colors[get_shelter(state)]] == coord

    def atBlue(self, state, action):
        return (get_x(state), get_y(state)) == coords[colors[3]]

    def atGreen(self, state, action):
        return (get_x(state), get_y(state)) == coords[colors[1]]

    def atRed(self, state, action):
        return (get_x(state), get_y(state)) == coords[colors[0]]

    def atYellow(self, state, action):
        return (get_x(state), get_y(state)) == coords[colors[2]]

    def atDestination(self, state, action):
        coord = (get_x(state), get_y(state))
        if action == 0:
            coord = (get_x(state), get_y(state) + 1)
        if action == 1:
            coord = (get_x(state), get_y(state) - 1)
        if action == 2:
            coord = (get_x(state) + 1, get_y(state))
        if action == 3:
            coord = (get_x(state) - 1, get_y(state))
        return coord == coords[colors[get_dest(state)]]

    def hasPassenger(self, state, action):
        return get_passenger(state) == 4

    def hurricane(self, state, action):
        return get_hurricane(state) > 0

    def newHurricane(self, state, action):
        return get_hurricane(state) == 1

    def rain(self, state, action):
        return get_rain(state) == 1

    def getLabels(self, state, action=None):
        if action is None:
            labels = []
        else:
            labels = [act_dict[action]]
            if toward(state, action, coords[colors[get_home(state)]]):
                labels.append("toward(atHome)")
            if toward(state, action, coords[colors[get_dest(state)]]):
                labels.append("toward(atDestination)")
            if toward(state, action, coords[colors[get_shelter(state)]]):
                labels.append("toward(atShelter)")
        for l in self.labels:
            fn = getattr(self, l)
            if fn(state, action):
                labels.append(l)

        return labels
