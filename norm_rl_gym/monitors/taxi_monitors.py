from norm_rl_gym.monitors.monitor import Monitor


def make_taxi_monitor(name, env):
    # all monitors defined in this module
    monitors = {name: obj for name, obj in globals().items() if isinstance(obj, type) and issubclass(obj, Monitor)}
    # now return the appropriate one
    if name is None:
        return Monitor(env)
    elif name in monitors:
        return monitors[name](env)
    # or
    else:
        print(f"ERROR. No such taxi monitor: {name}.\n  Available monitors:")
        print("  " + "\n  ".join(sorted(monitors)))
        raise SystemExit


class EmergencyMonitor(Monitor):
    def __init__(self, env):
        self.already_rain = False
        self.warn_violations = 0
        self.new_hurricane = 0
        self.warned_hurricane = 0
        self.shelter_violations = 0
        self.stay_violations = 0
        Monitor.__init__(self, env)

    def detectViolation(self, state, action, verbose=False):
        house_safe = not self.labels.floodrisk(state, action)
        viol = False
        if not self.already_rain and self.labels.rain(state, action) and not self.labels.warn(state, action):
            self.warn_violations += 1
            self.violations += 1
            if verbose:
                print("Violated Warning!")
            viol = True
        if self.labels.newHurricane(state, action) and not self.already_rain:
            self.new_hurricane = 1
        if self.labels.newHurricane(state, action) and self.already_rain:
            self.warned_hurricane = 1
        if self.labels.atShelter(state, action) or (house_safe and self.labels.atHome(state, action)):
            self.new_hurricane = 0
            self.warned_hurricane = 0
        if self.new_hurricane == 8:
            self.shelter_violations += 1
            self.violations += 1
            if verbose:
                print("Violated Shelter!")
            viol = True
        if self.warned_hurricane == 4:
            self.shelter_violations += 1
            self.violations += 1
            if verbose:
                print("Violated Shelter!")
            viol = True
        if self.new_hurricane > 0:
            self.new_hurricane += 1
        if self.warned_hurricane > 0:
            self.warned_hurricane += 1

        if self.labels.rain(state, action):
            self.already_rain = True
        if (
            self.new_hurricane == 0
            and self.warned_hurricane == 0
            and self.labels.hurricane(state, action)
            and not self.labels.atShelter(state, action)
            and (not house_safe or not self.labels.atHome(state, action))
        ):
            self.violations += 1
            self.stay_violations += 1
            if verbose:
                print("Violated Stay!")

        return viol

    def reset(self):
        self.already_rain = False
        self.new_hurricane = 0
        self.warned_hurricane = 0
        self.warn_violations = 0
        self.shelter_violations = 0
        self.stay_violations = 0
        super().reset()
        return

    def export(self):
        exp = {
            self.__class__.__name__[:-7] + " Violations": self.violations,
            "Warn Violations": self.warn_violations,
            "Safety Violations": self.shelter_violations,
            "Stay Violations": self.stay_violations,
        }
        return exp
