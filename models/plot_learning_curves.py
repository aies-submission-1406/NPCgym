#!/usr/bin/env python3
"""
Create per-instance and combined learning-curve plots from seeded CSV files.

The script expects one CSV file per instance and seed. The filename template must
contain the literal token ``<SEED>``, for example ``merchant_basic_<SEED>.csv``.
For each template, the script:

1. finds all matching seed files;
2. averages ``Mean Reward`` and the selected monitor columns across seeds at each
   timestep;
3. writes one individual plot per instance; and
4. writes one combined paper-style figure with the same-height panels in a row.

Usage
-----
    python plot_learning_curves.py --outdir plots

By default, the script reads CSV files from the current working directory.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib as mpl

# Use a non-interactive backend so the script also works on servers/CI.
mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pandas as pd


# ---------------------------------------------------------------------------
# User-editable configuration
# ---------------------------------------------------------------------------

PLOT_FILENAME_TEMPLATES = [
    "pacman_smallClassic_complete_<SEED>_curve.csv",
    "merchant_basic_<SEED>.csv",
    "taxi_<SEED>.csv",
    "gardener_15_dqn_<SEED>.csv",
]

# Use keys that are prefixes before <SEED>. Missing columns are ignored with a
# warning. If an instance is not listed here, the script automatically plots all
# columns containing "Monitor" in their name, capped at MAX_AUTO_MONITOR_COLUMNS.
INSTANCE_FIELDS = {
    "gardener_15_dqn_": [
        "CollectOneMonitor.CollectOne",
        "RescueMonitor.Rescue",
        "DrainMonitor.Drain",
        "NoCollectMonitor.NoCollect",
        "CollectPermMonitor.CollectPerm",
    ],
    "merchant_basic_": [
        "DangerMonitor.Danger",
        "DeliveryMonitor.Delivery",
        "EnvFriendlyMonitor.EnvFriendly",
        "PacifistMonitor.CTD",
    ],
    "merchant_dangerous_": [
        "DangerMonitor.Danger",
        "DeliveryMonitor.Delivery",
        "EnvFriendlyMonitor.EnvFriendly",
        "PacifistMonitor.CTD",
    ],
    "pacman_smallClassic_complete_": [
        "HungryVeganPenaltyMonitor.Hungr",
        "VeganMonitor.Vegan",
        "HungryVeganPenaltyMonitor.CTD Violations",
    ],
    "taxi_": [
        "EmergencyMonitor.Safety Violations",
        "EmergencyMonitor.Stay Violations",
        "EmergencyMonitor.Warn Violations",
    ],
}

# Long CSV column names can be shortened here for the plot legends.
RENAME_MONITORS = {
    "CollectOneMonitor.CollectOne": "collectOne",
    "RescueMonitor.Rescue": "rescue",
    "DrainMonitor.Drain": "drain",
    "NoCollectMonitor.NoCollect": "noCollect",
    "CollectPermMonitor.CollectPerm": "collectPerm",
    "DangerMonitor.Danger": "danger",
    "DeliveryMonitor.Delivery": "delivery",
    "EnvFriendlyMonitor.EnvFriendly": "environment",
    "PacifistMonitor.CTD": "pacifist",
    "EmergencyMonitor.Safety Violations": "shelter",
    "EmergencyMonitor.Stay Violations": "stay",
    "EmergencyMonitor.Warn Violations": "warn",
    "HungryVeganPenaltyMonitor.Hungr": "hungry",
    "VeganMonitor.Vegan": "vegBlue+vegOrange",
    "HungryVeganPenaltyMonitor.CTD Violations": "penalty",
}

# Plot titles. Keys should match entries in PLOT_FILENAME_TEMPLATES.
PLOT_NAMES = {
    "gardener_15_dqn_<SEED>.csv": "Gardener",
    "merchant_basic_<SEED>.csv": "Merchant basic",
    "merchant_dangerous_<SEED>.csv": "Merchant dangerous",
    "pacman_smallClassic_complete_<SEED>_curve.csv": "Pac-Man",
    "taxi_<SEED>.csv": "Taxi",
}

MAX_AUTO_MONITOR_COLUMNS = 5

# Standard column names in the CSVs.
TIMESTEP_COLUMN = "Timesteps"
REWARD_COLUMN = "Mean Reward"

# Figure defaults.
INDIVIDUAL_FIGSIZE = (5.6, 3.7)
COMBINED_PANEL_SIZE = (3.8, 3.3)
DPI = 300
PACMAN_REWARD_RESCALE = 100.0


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CsvSource:
    """A matched CSV file and the seed extracted from its filename."""

    name: str
    seed: str


def template_to_regex(template: str) -> re.Pattern[str]:
    """Convert a filename template with <SEED> to a compiled regex."""

    if "<SEED>" not in template:
        raise ValueError(f"Template must contain '<SEED>': {template!r}")
    pattern = re.escape(template).replace(re.escape("<SEED>"), r"(?P<seed>\d+)")
    return re.compile(rf"^{pattern}$")


def template_prefix(template: str) -> str:
    """Return the part of a template before the <SEED> token."""

    return template.split("<SEED>", maxsplit=1)[0]


def safe_stem_from_template(template: str) -> str:
    """Create a filesystem-friendly stem from a filename template."""

    stem = Path(template.replace("<SEED>", "")).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_.-")
    return stem or "learning_curve"


def discover_csvs(input_path: Path, template: str) -> list[CsvSource]:
    """Find all CSV files in a directory matching a template."""

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path is not a directory: {input_path}")

    regex = template_to_regex(template)
    matches = []
    for path in input_path.glob("*.csv"):
        match = regex.match(path.name)
        if match:
            matches.append(CsvSource(name=str(path), seed=match.group("seed")))

    return sorted(matches, key=lambda source: int(source.seed))


def read_csv(input_path: Path, source_name: str) -> pd.DataFrame:
    """Read one CSV from disk."""

    return pd.read_csv(source_name)


def select_monitor_columns(
    template: str,
    available_columns: Iterable[str],
    instance_fields: Mapping[str, list[str]],
    max_auto_columns: int,
) -> list[str]:
    """Return the configured monitor columns for a template.

    Preference order:
    1. INSTANCE_FIELDS entry keyed by the template prefix before <SEED>;
    2. INSTANCE_FIELDS entry keyed by the full template string;
    3. automatically detected columns containing "Monitor".
    """

    available = list(available_columns)
    prefix = template_prefix(template)
    configured = instance_fields.get(prefix) or instance_fields.get(template)

    if configured is not None:
        missing = [column for column in configured if column not in available]
        if missing:
            print(
                f"Warning: {template}: ignoring missing monitor columns: {missing}",
                file=sys.stderr,
            )
        return [column for column in configured if column in available]

    detected = [
        column for column in available if "Monitor" in column and column not in {TIMESTEP_COLUMN, REWARD_COLUMN}
    ]
    return detected[:max_auto_columns]


def load_and_average_instance(
    input_path: Path, template: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Load all seed CSVs for one template and average numeric columns by timestep."""

    matches = discover_csvs(input_path, template)
    if not matches:
        raise FileNotFoundError(f"No files matched template: {template}")

    frames = []
    all_columns: set[str] | None = None

    for source in matches:
        frame = read_csv(input_path, source.name)
        frame["Seed"] = int(source.seed)
        if all_columns is None:
            all_columns = set(frame.columns)
        else:
            all_columns &= set(frame.columns)
        frames.append(frame)

    assert all_columns is not None
    if TIMESTEP_COLUMN not in all_columns or REWARD_COLUMN not in all_columns:
        raise ValueError(f"{template}: expected columns {TIMESTEP_COLUMN!r} and {REWARD_COLUMN!r}.")

    # Use only columns present in every seed file, because those are safe to average.
    common_columns = [column for column in frames[0].columns if column in all_columns]
    monitor_columns = select_monitor_columns(
        template=template,
        available_columns=common_columns,
        instance_fields=INSTANCE_FIELDS,
        max_auto_columns=MAX_AUTO_MONITOR_COLUMNS,
    )
    value_columns = [REWARD_COLUMN, *monitor_columns]

    tidy_frames = [frame[[TIMESTEP_COLUMN, "Seed", *value_columns]].copy() for frame in frames]
    combined = pd.concat(tidy_frames, ignore_index=True)

    # Convert values defensively. Non-numeric values become NaN and are ignored
    # by the mean operation.
    for column in [TIMESTEP_COLUMN, *value_columns]:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")

    grouped = combined.groupby(TIMESTEP_COLUMN, as_index=False)[value_columns]
    averaged_mean = pd.DataFrame(grouped.mean(numeric_only=True)).sort_values(TIMESTEP_COLUMN)
    averaged_std = pd.DataFrame(grouped.std(numeric_only=True)).sort_values(TIMESTEP_COLUMN).fillna(0.0)

    if "pacman" in template.lower():
        averaged_mean[REWARD_COLUMN] *= PACMAN_REWARD_RESCALE
        averaged_std[REWARD_COLUMN] *= PACMAN_REWARD_RESCALE

    return averaged_mean, averaged_std, monitor_columns, [source.name for source in matches]


def timestep_formatter(value: float, _: int) -> str:
    """Format timesteps compactly for x-axis tick labels."""

    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def label_for_monitor(column: str) -> str:
    """Return a user-friendly label for a monitor column."""

    return RENAME_MONITORS.get(column, column)


def plot_instance_on_axes(
    ax_reward: Axes,
    averaged_mean: pd.DataFrame,
    averaged_std: pd.DataFrame | None,
    monitor_columns: list[str],
    title: str,
    show_right_ylabel: bool,
    compact_legend: bool,
    show_errors: bool,
) -> tuple[Axes, Axes]:
    """Draw one learning curve with reward on the left axis and monitors on the right."""

    ax_monitor = ax_reward.twinx()
    x = averaged_mean[TIMESTEP_COLUMN]

    reward_line = ax_reward.plot(
        x,
        averaged_mean[REWARD_COLUMN],
        linewidth=2.4,
        color="black",
        label="Avg. return",
        zorder=4,
    )

    if show_errors and averaged_std is not None:
        reward_mean = averaged_mean[REWARD_COLUMN]
        reward_std = averaged_std[REWARD_COLUMN]
        ax_reward.fill_between(x, reward_mean - reward_std, reward_mean + reward_std, color="black", alpha=0.12)

    monitor_lines = []
    for column in monitor_columns:
        line = ax_monitor.plot(
            x,
            averaged_mean[column],
            linewidth=1.55,
            alpha=0.92,
            label=label_for_monitor(column),
            zorder=3,
        )
        if show_errors and averaged_std is not None:
            color = line[0].get_color()
            monitor_mean = averaged_mean[column]
            monitor_std = averaged_std[column]
            ax_monitor.fill_between(x, monitor_mean - monitor_std, monitor_mean + monitor_std, color=color, alpha=0.12)
        monitor_lines.extend(line)

    ax_reward.set_title(title, fontsize=11.5, pad=8)
    ax_reward.set_xlabel("Timesteps")
    ax_reward.set_ylabel("Average return")
    if show_right_ylabel:
        ax_monitor.set_ylabel("Monitor value")
    else:
        ax_monitor.set_ylabel("")

    ax_reward.xaxis.set_major_formatter(FuncFormatter(timestep_formatter))
    ax_reward.xaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
    ax_reward.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_monitor.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax_reward.grid(True, axis="both", linewidth=0.55, alpha=0.35)
    ax_reward.set_axisbelow(True)

    # Anchor monitor axis at zero for consistent count interpretation.
    ax_monitor.set_ylim(bottom=0)

    # Give both axes a small amount of breathing room.
    ax_reward.margins(x=0.03, y=0.08)
    ax_monitor.margins(x=0.03, y=0.08)

    lines = reward_line + monitor_lines
    labels = [str(line.get_label()) for line in lines]

    if compact_legend:
        ax_reward.legend(
            lines,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.28),
            ncol=2,
            fontsize=7.0,
            frameon=False,
            columnspacing=0.9,
            handlelength=1.7,
        )
    else:
        ax_reward.legend(
            lines,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            fontsize=8.0,
            frameon=False,
            columnspacing=1.0,
            handlelength=2.0,
        )

    return ax_reward, ax_monitor


def save_figure(fig: Figure, output_stem: Path) -> list[Path]:
    """Save a figure as PNG and PDF and return the output paths."""

    output_paths = [
        output_stem.with_suffix(".png"),
        output_stem.with_suffix(".pdf"),
    ]
    for path in output_paths:
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
    return output_paths


def make_individual_plot(
    averaged_mean: pd.DataFrame,
    averaged_std: pd.DataFrame | None,
    monitor_columns: list[str],
    title: str,
    output_stem: Path,
    show_errors: bool,
) -> list[Path]:
    """Create and save one individual instance plot."""

    fig, ax = plt.subplots(figsize=INDIVIDUAL_FIGSIZE)
    plot_instance_on_axes(
        ax_reward=ax,
        averaged_mean=averaged_mean,
        averaged_std=averaged_std,
        monitor_columns=monitor_columns,
        title=title,
        show_right_ylabel=True,
        compact_legend=False,
        show_errors=show_errors,
    )
    fig.tight_layout()
    paths = save_figure(fig, output_stem)
    plt.close(fig)
    return paths


def make_combined_plot(
    plotted_instances: list[tuple[str, pd.DataFrame, pd.DataFrame | None, list[str]]],
    output_stem: Path,
    show_errors: bool,
) -> list[Path]:
    """Create and save a single-row combined plot for all instances."""

    n_panels = len(plotted_instances)
    width = COMBINED_PANEL_SIZE[0] * n_panels
    height = COMBINED_PANEL_SIZE[1]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_panels,
        figsize=(width, height),
        squeeze=False,
    )
    axes_row = list(axes[0])

    for index, (title, averaged_mean, averaged_std, monitor_columns) in enumerate(plotted_instances):
        ax = axes_row[index]
        plot_instance_on_axes(
            ax_reward=ax,
            averaged_mean=averaged_mean,
            averaged_std=averaged_std,
            monitor_columns=monitor_columns,
            title=title,
            show_right_ylabel=(index == n_panels - 1),
            compact_legend=True,
            show_errors=show_errors,
        )

        # Avoid repeating the left y-axis label on every panel.
        if index > 0:
            ax.set_ylabel("")

    # Leave space for the per-panel legends below the axes.
    fig.subplots_adjust(wspace=0.45, bottom=0.31, top=0.84)
    paths = save_figure(fig, output_stem)
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create averaged learning-curve plots from seeded CSV files.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("."),
        help="Directory containing the CSVs. Defaults to the current directory.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("learning_curve_plots"),
        help="Directory where plots will be written.",
    )
    parser.add_argument(
        "--combined-name",
        default="combined_learning_curves",
        help="Filename stem for the combined plot.",
    )
    parser.set_defaults(show_errors=True)
    parser.add_argument(
        "--no-errors",
        action="store_false",
        dest="show_errors",
        help="Disable standard-deviation error shading.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    plotted_instances: list[tuple[str, pd.DataFrame, pd.DataFrame | None, list[str]]] = []

    for template in PLOT_FILENAME_TEMPLATES:
        title = PLOT_NAMES.get(template, template_prefix(template).rstrip("_") or template)
        try:
            averaged_mean, averaged_std, monitor_columns, matched_files = load_and_average_instance(
                args.input, template
            )
        except Exception as exc:
            print(f"Skipping {template}: {exc}", file=sys.stderr)
            continue

        if not monitor_columns:
            print(f"Skipping {template}: no monitor columns selected.", file=sys.stderr)
            continue

        output_stem = args.outdir / f"{safe_stem_from_template(template)}_learning_curve"
        make_individual_plot(
            averaged_mean=averaged_mean,
            averaged_std=averaged_std if args.show_errors else None,
            monitor_columns=monitor_columns,
            title=title,
            output_stem=output_stem,
            show_errors=args.show_errors,
        )
        plotted_instances.append((title, averaged_mean, averaged_std if args.show_errors else None, monitor_columns))

        print(f"Plotted {title}: {len(matched_files)} seed files, {len(monitor_columns)} monitor columns.")

    if not plotted_instances:
        raise RuntimeError("No plots were created. Check templates and input path.")

    make_combined_plot(
        plotted_instances=plotted_instances,
        output_stem=args.outdir / args.combined_name,
        show_errors=args.show_errors,
    )
    print(f"Finished. Wrote plots to: {args.outdir.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
