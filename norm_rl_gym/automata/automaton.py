class DFA:
    def __init__(self, alphabet, reward=0.0, reset="punctual", permission=False):
        self.alphabet = alphabet.labels
        self.state0 = 0
        self.reward = reward
        self.states = []
        self.final = []
        self.type = reset
        self.permission = permission
        self.state = 0

    def transition(self, inpt, state):
        pass

    def reset(self, prevState, state):
        if state in self.final:
            if self.type == "maintenance":
                return prevState
            elif self.type in ["achievement", "punctual"]:
                return self.state0
            else:
                return state
        return state
