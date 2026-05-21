from gymnasium.wrappers import *


class Monitor:
    def __init__(self, env, verbose=False):
        self.violations = 0
        self.env = env.unwrapped
        self.labels = self.env.labels
        self.verbose = verbose
        self.reset()

    def detectViolation(self, state, action):
        return False

    def reset(self):
        self.violations = 0
        self.detectViolation(self.env.get_state(), None)

    def export(self):
        exp = {self.__class__.__name__[:-7]: self.violations}
        return exp


class ComposingMonitor(Monitor):
    def __init__(self, env, monitors, verbose=False):
        self.monitors = monitors
        super().__init__(env, verbose)

    def reset(self):
        self.violations = 0
        for monitor in self.monitors:
            monitor.reset()
            self.violations += monitor.violations

    def detectViolation(self, state, action):
        for monitor in self.monitors:
            if monitor.detectViolation(state, action):
                self.violations += 1
