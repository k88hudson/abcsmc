# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.0", "matplotlib>=3.8", "scipy>=1.11", "numpy>=1.26"]
# ///
"""Plot the renewal example's calibration output.

Run after `cargo run --example renewal`, from anywhere:

    uv run examples/output/examine_output.py

Reads particles.csv and trajectories.csv next to this file and writes one PNG
per figure into examples/output/plots/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
PLOTS = OUTPUT / "plots"

# Values the sample incidence was generated with, and the fitting window used
# by examples/renewal/calibration.rs.
TRUE_R0 = 1.5
TRUE_INITIAL_INFECTIONS = 1
FITTING_DAYS = 7 * 6

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# One blue hue, light to dark: generations are an ordered sequence, so each gets a
# step of the same ramp rather than its own hue.
GENERATION_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
HELD_OUT = "#eb6834"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "medium",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.titleweight": "medium",
    }
)


def generation_colors(n: int) -> list[str]:
    cmap = LinearSegmentedColormap.from_list("generation", GENERATION_RAMP)
    return [matplotlib.colors.to_hex(cmap(t)) for t in np.linspace(0, 1, n)]


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, qs: list[float]) -> np.ndarray:
    """Weighted quantiles down axis 0 of `values` (particles x days). Returns
    an array of shape (len(qs), days)."""
    order = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, order, axis=0)
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights, axis=0)
    cumulative /= cumulative[-1]
    rows = []
    for q in qs:
        index = (cumulative >= q).argmax(axis=0)
        rows.append(sorted_values[index, np.arange(values.shape[1])])
    return np.array(rows)


def weighted_kde(values: np.ndarray, weights: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return gaussian_kde(values, weights=weights)(grid)


def observed_points(ax, incidence: pd.DataFrame, size: float = 9):
    fitted = incidence[incidence.day < FITTING_DAYS]
    held_out = incidence[incidence.day >= FITTING_DAYS]
    ax.scatter(fitted.day, fitted.symptomatic_incidence, s=size, color=INK, zorder=3, linewidths=0)
    ax.scatter(held_out.day, held_out.symptomatic_incidence, s=size, color=HELD_OUT, zorder=3, linewidths=0)


def observed_legend():
    return [
        Line2D([], [], marker="o", linestyle="", color=INK, markersize=4, label="observed, fitted (first 6 weeks)"),
        Line2D([], [], marker="o", linestyle="", color=HELD_OUT, markersize=4, label="observed, held out"),
    ]


def save(fig, name: str):
    path = PLOTS / name
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def parameter_scatter(particles: pd.DataFrame, generations: list[int], colors: list[str]):
    fig, axes = plt.subplots(1, len(generations), figsize=(2.6 * len(generations) + 1.5, 4.2), sharey=True)
    cmap = LinearSegmentedColormap.from_list("distance", SEQUENTIAL[::-1])
    log_distance = np.log10(particles.distance + 1)
    norm = matplotlib.colors.Normalize(log_distance.min(), log_distance.max())
    rng = np.random.default_rng(0)
    for ax, generation in zip(axes, generations):
        rows = particles[particles.generation == generation]
        jitter = rng.uniform(-0.3, 0.3, len(rows))
        size = 4 + 60 * rows.weight / rows.weight.max()
        ax.scatter(
            rows.initial_infections + jitter, rows.r0,
            c=np.log10(rows.distance + 1), cmap=cmap, norm=norm,
            s=size, linewidths=0, alpha=0.85,
        )
        ax.scatter([TRUE_INITIAL_INFECTIONS], [TRUE_R0], marker="+", s=90, color=INK, linewidths=1.5, zorder=4)
        ax.set_yscale("log")
        ax.set_title(f"generation {generation}")
        ax.set_xticks(sorted(particles.initial_infections.unique()))
        ax.set_xlabel("initial infections")
    axes[0].set_ylabel("R0 (log scale)")
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, pad=0.02, shrink=0.8)
    colorbar.set_label("log10(distance + 1), darker is closer", color=INK_SECONDARY)
    colorbar.outline.set_visible(False)
    axes[0].legend(
        handles=[Line2D([], [], marker="+", linestyle="", color=INK, markersize=8, label="true value")],
        loc="lower left",
    )
    fig.suptitle("Accepted particles by generation (marker size is weight)", x=0.45)
    save(fig, "1_parameter_scatter.png")


def trajectories_plot(trajectories: pd.DataFrame, particles: pd.DataFrame, incidence: pd.DataFrame,
                      generations: list[int], colors: list[str]):
    fig, axes = plt.subplots(1, len(generations), figsize=(3 * len(generations) + 1, 4), sharey=True, sharex=True)
    for ax, generation, color in zip(axes, generations, colors):
        curves, weights = generation_matrix(trajectories, particles, generation)
        days = np.arange(curves.shape[1])
        segments = [np.column_stack([days, row]) for row in curves]
        alpha = np.clip(0.08 * weights / weights.max(), 0.004, 1)
        rgba = np.tile(matplotlib.colors.to_rgba(color), (len(segments), 1))
        rgba[:, 3] = alpha
        ax.add_collection(LineCollection(segments, colors=rgba, linewidths=0.8))
        observed_points(ax, incidence, size=6)
        ax.set_xlim(0, days.max())
        ax.set_ylim(0, 500)
        ax.set_title(f"generation {generation}")
        ax.set_xlabel("day")
    axes[0].set_ylabel("symptomatic incidence")
    fig.legend(handles=observed_legend(), loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Simulated trajectories (opacity is weight)", x=0.3)
    fig.tight_layout()
    save(fig, "2_trajectories.png")


def generation_matrix(trajectories: pd.DataFrame, particles: pd.DataFrame, generation: int):
    """Pivot one generation's trajectories to (particles x days) plus normalized weights."""
    rows = trajectories[trajectories.generation == generation]
    curves = rows.pivot(index="particle_number", columns="day", values="symptomatic_incidence")
    weights = particles[particles.generation == generation].set_index("particle_number").weight.loc[curves.index]
    return curves.to_numpy(dtype=float), (weights / weights.sum()).to_numpy()


def trajectory_quantiles(trajectories, particles, incidence, generations, colors):
    fig, axes = plt.subplots(len(generations), 1, figsize=(8, 2.1 * len(generations) + 0.8), sharex=True, sharey=True)
    for ax, generation, color in zip(axes, generations, colors):
        curves, weights = generation_matrix(trajectories, particles, generation)
        days = np.arange(curves.shape[1])
        low, iqr_low, mid, iqr_high, high = weighted_quantiles(curves, weights, [0.025, 0.25, 0.5, 0.75, 0.975])
        ax.fill_between(days, low, high, color=color, alpha=0.22, linewidth=0)
        ax.fill_between(days, iqr_low, iqr_high, color=color, alpha=0.4, linewidth=0)
        ax.plot(days, mid, color=color, linewidth=1.6)
        ax.axvline(FITTING_DAYS, color=AXIS, linewidth=0.8, zorder=1)
        observed_points(ax, incidence)
        ax.set_ylabel("incidence")
        ax.text(0.99, 0.9, f"generation {generation}", transform=ax.transAxes, ha="right", color=INK, fontsize=10)
    axes[-1].set_xlabel("day")
    axes[0].text(FITTING_DAYS + 1.5, axes[0].get_ylim()[1] * 0.9, "fit window ends", color=MUTED, fontsize=8, va="top")
    handles = observed_legend() + [
        Patch(facecolor=colors[-1], alpha=0.4, label="weighted 50% interval"),
        Patch(facecolor=colors[-1], alpha=0.22, label="weighted 95% interval"),
        Line2D([], [], color=colors[-1], linewidth=1.6, label="weighted median"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Trajectory quantiles by generation", y=1.03)
    fig.tight_layout()
    save(fig, "3_trajectory_quantiles.png")


def r0_densities(particles, generations, colors):
    fig, ax = plt.subplots(figsize=(7, 4))
    grid = np.linspace(0, 3, 400)
    for generation, color in zip(generations, colors):
        rows = particles[particles.generation == generation]
        ax.plot(grid, weighted_kde(rows.r0.to_numpy(), rows.weight.to_numpy(), grid),
                color=color, linewidth=1.8, label=f"generation {generation}")
    ax.axvline(TRUE_R0, color=INK, linewidth=0.9)
    ax.text(TRUE_R0 + 0.03, ax.get_ylim()[1] * 0.97, "true R0", color=INK_SECONDARY, fontsize=8, va="top")
    ax.set_xlim(0, 3)
    ax.set_xlabel("R0")
    ax.set_ylabel("weighted density")
    ax.legend(loc="upper right")
    ax.set_title("Posterior of R0 by generation")
    save(fig, "4_parameter_distributions.png")


def initial_infection_shares(particles, generations, colors):
    shares = (
        particles.groupby(["generation", "initial_infections"]).weight.sum()
        .unstack("initial_infections", fill_value=0.0)
    )
    shares = shares.div(shares.sum(axis=1), axis=0)
    values = shares.columns.to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    group_width = 0.8
    bar_width = group_width / len(generations)
    for i, (generation, color) in enumerate(zip(generations, colors)):
        offsets = values - group_width / 2 + bar_width * (i + 0.5)
        ax.bar(offsets, shares.loc[generation], width=bar_width * 0.92, color=color, label=f"generation {generation}")
    ax.set_xticks(values)
    ax.set_xticklabels([f"{v}\n(true value)" if v == TRUE_INITIAL_INFECTIONS else str(v) for v in values])
    ax.set_xlabel("initial infections")
    ax.set_ylabel("weighted share of particles")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Posterior of initial infections by generation")
    save(fig, "5_initial_infections.png")


def outbreak_size(trajectories, particles, incidence, generations, colors):
    generation = generations[-1]
    curves, weights = generation_matrix(trajectories, particles, generation)
    totals = curves.sum(axis=1)
    observed_total = incidence.symptomatic_incidence.sum()
    grid = np.linspace(0, max(totals.max(), observed_total) * 1.05, 400)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, weighted_kde(totals, weights, grid), color=colors[-1], linewidth=1.8)
    ax.axvline(observed_total, color=INK, linewidth=0.9)
    ax.text(observed_total + grid[-1] * 0.01, ax.get_ylim()[1] * 0.97, "observed total",
            color=INK_SECONDARY, fontsize=8, va="top")
    ax.set_xlabel("total symptomatic incidence over the full horizon")
    ax.set_ylabel("weighted density")
    ax.set_title(f"Outbreak size, generation {generation}")
    save(fig, "6_outbreak_size.png")


def main():
    PLOTS.mkdir(exist_ok=True)
    incidence = pd.read_csv(ROOT / "examples/input/sample_incidence.csv")
    particles = pd.read_csv(OUTPUT / "particles.csv")
    particles["weight"] = particles.weight / particles.groupby("generation").weight.transform("sum")
    trajectories = pd.read_csv(OUTPUT / "trajectories.csv")
    generations = sorted(particles.generation.unique())
    colors = generation_colors(len(generations))

    parameter_scatter(particles, generations, colors)
    trajectories_plot(trajectories, particles, incidence, generations, colors)
    trajectory_quantiles(trajectories, particles, incidence, generations, colors)
    r0_densities(particles, generations, colors)
    initial_infection_shares(particles, generations, colors)
    outbreak_size(trajectories, particles, incidence, generations, colors)


if __name__ == "__main__":
    main()
