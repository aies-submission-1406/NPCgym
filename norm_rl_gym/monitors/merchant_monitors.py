from norm_rl_gym.monitors.monitor import Monitor


def _action_name(monitor, action):
    """Normalize merchant action representation (int index or str label) to str."""
    if action is None:
        return None
    if isinstance(action, str):
        return action.lower()
    if isinstance(action, int):
        labels = getattr(monitor.labels, "action_labels", None)
        if labels is not None and 0 <= action < len(labels):
            return str(labels[action]).lower()
    return str(action).lower()


def make_merchant_monitor(name, env):
    # all monitors defined in this module
    monitors = {name: obj for name, obj in globals().items() if isinstance(obj, type) and issubclass(obj, Monitor)}
    # now return the appropriate one
    if name is None:
        return Monitor(env)
    elif name in monitors:
        return monitors[name](env)
    # or
    else:
        print(f"ERROR. No such merchant monitor: {name}.\n  Available monitors:")
        print("  " + "\n  ".join(sorted(monitors)))
        raise SystemExit


class DangerMonitor(Monitor):
    def __init__(self, env, verbose=False):
        super().__init__(env, verbose)

    def detectViolation(self, state, action):
        if self.labels.atDanger(state, action):
            self.violations += 1
            if self.verbose:
                print("Danger Prohibition Violated!")
            return True
        else:
            return False


class PacifistMonitor(Monitor):
    def __init__(self, env, verbose=False):
        self.danger = DangerMonitor(env)
        self.ctdViolations = 0
        self.wasDanger = False
        super().__init__(env, verbose)

    def detectViolation(self, state, action):
        ctd = False
        if self.wasDanger:
            if action == "fight":
                ctd = True
                self.ctdViolations += 1
                self.violations += 1
        self.wasDanger = self.danger.detectViolation(state, action)
        if self.wasDanger:
            self.violations += 1
            return True
        return ctd

    # TODO:
    def reset(self):
        self.danger.reset()
        self.ctdViolations = 0
        self.wasDanger = False
        self.danger.reset()
        super().reset()

    def export(self):
        exp = {
            self.__class__.__name__[:-7] + "(total)": self.violations,
            self.danger.__class__.__name__[:-7]: self.danger.violations,
            "CTD": self.ctdViolations,
        }
        return exp


class DeliveryMonitor(Monitor):
    def __init__(self, env, verbose=False):
        self.oblInForce = False
        super().__init__(env, verbose)

    def detectViolation(self, state, action):
        if self.labels.atHome(state, action):
            self.oblInForce = True
        if self.oblInForce and self.labels.sundown(state, action):
            self.violations += 1
            if self.verbose:
                print("Failed to make delivery!")
            self.oblInForce = False
            return True
        if self.labels.atMarket(state, action):
            self.oblInForce = False
        return False

    def reset(self):
        self.oblInForce = False
        super().reset()


class EnvFriendlyMonitor(Monitor):
    def __init__(self, env, verbose=False):
        self.justAtTree = False
        super().__init__(env, verbose)

    def detectViolation(self, state, action):
        if self.justAtTree and action == "extract":
            self.violations += 1
            return True
        self.justAtTree = False
        self.justAtTree = self.labels.hasWood(state, action) and self.labels.atTree(state, action)
        return False

    def reset(self):
        self.justAtTree = False
        super().reset()


class EvolvingMonitor(Monitor):
    def __init__(self, env, verbose=False):
        self.counter = 0
        self.justAtTree = False
        super().__init__(env, verbose)

    # TODO: update after EnvFriendly is confirmed
    def detectViolation(self, state, action):
        action_name = _action_name(self, action)
        self.counter += 1
        if self.counter >= 15:
            if self.justAtTree and action == "extract":
                self.violations += 1
                self.justAtTree = False
                return True
            self.justAtTree = False
            self.justAtTree = self.labels.hasWood(state, action) and self.labels.atTree(state, action)
        else:
            if self.justAtTree and action == "extract":
                self.violations += 1
                self.justAtTree = False
                return True
            self.justAtTree = False
            self.justAtTree = self.labels.atTree(state, action)
        return False

    def reset(self):
        self.justAtTree = False
        self.counter = 0
        super().reset()
