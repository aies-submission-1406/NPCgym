# from .game import *

act_map = {0: "Stop", 1: "North", 2: "South", 3: "East", 4: "West"}


class Labels:
    def __init__(self):
        self.labels = [
            "eatBlueGhost",
            "eatOrangeGhost",
            "adjacentBlueGhost",
            "adjacentOrangeGhost",
            "stayedStill",
            "lose",
            "score0",
            "scoreGreater70",
            "scoreGreater80",
            "scoreGreater90",
            "scoreGreater100",
            "scoreGreater150",
            "scoreGreater500",
            "westSide",
            "eatPowerPellet",
            "inCorner",
            "inSouthEast",
        ]

    def eatBlueGhost(self, state, action=None):
        return state.killedBlue

    def eatOrangeGhost(self, state, action=None):
        return state.killedOrange

    def adjacentBlueGhost(self, state, action=None):
        pos = state.getGhostPosition(1)
        dx = pos[0] - state.getPacmanPosition()[0]
        dy = pos[1] - state.getPacmanPosition()[1]
        return (
            state.getGhostState(1).isScared()
            and dx <= 1.0
            and dx >= -1.0
            and dy <= 1.0
            and dy >= -1.0
            and dx != 0
            and dy != 0
        )

    def adjacentOrangeGhost(self, state, action=None):
        pos = state.getGhostPosition(2)
        dx = pos[0] - state.getPacmanPosition()[0]
        dy = pos[1] - state.getPacmanPosition()[1]
        return (
            state.getGhostState(2).isScared()
            and dx <= 1.0
            and dx >= -1.0
            and dy <= 1.0
            and dy >= -1.0
            and dx != 0
            and dy != 0
        )

    def eatPowerPellet(self, state, action=None):
        return state.ateCapsule

    def stayedStill(self, state, action=None):
        return state.lastAct == 0

    def lose(self, state, action=None):
        return state.lost

    def score0(self, state, action=None):
        return state.data.score == 0

    def scoreGreater70(self, state, action=None):
        return state.data.score > 70

    def scoreGreater80(self, state, action=None):
        return state.data.score > 80

    def scoreGreater90(self, state, action=None):
        return state.data.score > 90

    def scoreGreater100(self, state, action=None):
        return state.data.score > 100

    def scoreGreater150(self, state, action=None):
        return state.data.score > 150

    def scoreGreater500(self, state, action=None):
        return state.data.score > 500

    def westSide(self, state, action=None):
        return state.data.layout.width / 2 > state.getPacmanPosition()[0]

    def getLabels(self, state, action=None):
        if action is None:
            labels = []
        else:
            labels = [act_map[action]]
        for l in self.labels:
            fn = getattr(self, l)
            if fn(state, action):
                labels.append(l)
        return labels

    def inCorner(self, state, action=None):
        return state.getPacmanPosition() in state.data.layout.getAllCorners()

    def inSouthEast(self, state, action=None):
        return state.getPacmanPosition() == state.data.layout.getSouthEastCorner()
