import csv


class StatsWriter:
    def __init__(self, study_name, trial_number=None):
        suffix = f"_{trial_number:03}" if trial_number is not None else ""
        self.filename = f"{study_name}{suffix}.csv"
        self.file = None
        self.writer = None

    def _write_line(self, line):
        if self.writer is None:
            self.file = open(self.filename, mode="w", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=list(line.keys()))
            self.writer.writeheader()
        assert self.file is not None
        self.writer.writerow(line)
        self.file.flush()

    def write_trial(self, monitors, extra):
        row = {}
        for monitor in monitors:
            row.update(monitor.export())
        row.update(extra)
        self._write_line(row)

    def close(self):
        if self.file is not None and not self.file.closed:
            self.file.close()
