#!/usr/bin/env python3

import argparse
import gc
import glob
import math
import os
import pickle
import re
import resource
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import spearmanr, t as student_t
from matplotlib.colors import LogNorm

DEFAULT_PROJECT_ROOT = "/hepusers2/fuscomus/DRToM"
DEFAULT_RAID_STORAGE = "/raid/adisk06/users/fuscomus/DRToM/Analysis"
DEFAULT_ENERGY_FOLDER = "TeV13p0_22000"
DEFAULT_PROCESS = "2to4"
DEFAULT_DR_SCALE = 11.0
DEFAULT_PLOTS_ROOT = "Plots"

FONT_SIZE = 14
DEFAULT_ENERGY_TEV = 13.0
DEFAULT_PROCESS_TYPE = "Phase Space"
DEFAULT_ETA_MAX = 4.0


def part_id(path):
    match = re.search(r"part_(\d+)\.pkl$", os.path.basename(path))
    if not match:
        raise RuntimeError(f"Cannot determine task ID from {path}")
    return int(match.group(1))


def max_rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


def validate(payload, path, process, dr_scale, required_key):
    if payload.get("what_process") != process:
        raise RuntimeError(
            f"Process mismatch in {path}: {payload.get('what_process')!r} != {process!r}"
        )

    try:
        stored_dr = float(payload.get("DR_scale"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid DR_scale in {path}") from exc

    if not math.isclose(stored_dr, dr_scale, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError(f"DR mismatch in {path}: {stored_dr} != {dr_scale}")

    if required_key not in payload:
        raise RuntimeError(f"Missing {required_key!r} in {path}")


def discover_pairs(raid_storage, energy_folder, process, dr_scale):
    root = os.path.join(
        raid_storage,
        "PartialOutputs",
        energy_folder,
        process,
        f"DR_{dr_scale}",
    )

    shape_by_id = {
        part_id(path): path
        for path in glob.glob(os.path.join(root, "event_shapes", "part_*.pkl"))
    }
    mass_by_id = {
        part_id(path): path
        for path in glob.glob(os.path.join(root, "masses", "part_*.pkl"))
    }

    missing_mass = sorted(set(shape_by_id) - set(mass_by_id))
    missing_shape = sorted(set(mass_by_id) - set(shape_by_id))

    if missing_mass:
        raise RuntimeError(f"No matching mass partials for TASK_IDs: {missing_mass[:20]}")
    if missing_shape:
        raise RuntimeError(f"No matching event-shape partials for TASK_IDs: {missing_shape[:20]}")

    ids = sorted(set(shape_by_id) & set(mass_by_id))
    if not ids:
        raise RuntimeError(f"No paired event_shapes/masses partials found under {root}")

    return root, [(task_id, shape_by_id[task_id], mass_by_id[task_id]) for task_id in ids]


def as_float_array(values):
    if values is None:
        return np.empty(0, dtype=float)
    try:
        return np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError):
        flat = []
        for value in values:
            if isinstance(value, np.ndarray):
                flat.extend(value.ravel().tolist())
            elif isinstance(value, (list, tuple)):
                flat.extend(np.asarray(value).ravel().tolist())
            else:
                flat.append(value)
        return np.asarray(flat, dtype=float)


class PrioritySample:
    def __init__(self, size, seed=42):
        self.size = int(size)
        self.rng = np.random.default_rng(seed)
        self.priority = np.empty(0, dtype=float)
        self.a = np.empty(0, dtype=float)
        self.b = np.empty(0, dtype=float)

    def update(self, a_values, b_values):
        if self.size <= 0 or len(a_values) == 0:
            return

        new_priority = self.rng.random(len(a_values))
        priority = np.concatenate((self.priority, new_priority))
        a_all = np.concatenate((self.a, a_values))
        b_all = np.concatenate((self.b, b_values))

        if len(priority) > self.size:
            keep = np.argpartition(priority, self.size - 1)[: self.size]
            priority = priority[keep]
            a_all = a_all[keep]
            b_all = b_all[keep]

        self.priority = priority
        self.a = a_all
        self.b = b_all


class Accumulator:
    def __init__(self, mass_min, mass_max, mass_bins, a_cut, b_cut, sample_size):
        self.mass_edges = np.linspace(mass_min, mass_max, mass_bins + 1)
        self.mass_centres = 0.5 * (self.mass_edges[:-1] + self.mass_edges[1:])
        self.a_cut = a_cut
        self.b_cut = b_cut

        self.weight_total = np.zeros(mass_bins, dtype=float)
        self.weight_a = np.zeros(mass_bins, dtype=float)
        self.weight_b = np.zeros(mass_bins, dtype=float)

        self.a2d_edges = np.linspace(0.0, 0.5, 51)
        self.b2d_edges = np.linspace(0.0, 1.0, 51)
        self.ab_counts = np.zeros((50, 50), dtype=np.int64)

        self.mean_a_edges = np.linspace(0.0, 0.5, 30)
        n_mean_bins = len(self.mean_a_edges) - 1
        self.mean_b_sum = np.zeros(n_mean_bins, dtype=float)
        self.mean_b_count = np.zeros(n_mean_bins, dtype=np.int64)

        self.n = 0
        self.sum_a = 0.0
        self.sum_b = 0.0
        self.sum_a2 = 0.0
        self.sum_b2 = 0.0
        self.sum_ab = 0.0

        self.sample = PrioritySample(sample_size)

    def update(self, a_values, b_values, masses_gev, weights):
        masses_tev = masses_gev * 1.0e-3

        valid = (
            np.isfinite(a_values)
            & np.isfinite(b_values)
            & np.isfinite(masses_tev)
            & np.isfinite(weights)
        )
        a_values = a_values[valid]
        b_values = b_values[valid]
        masses_tev = masses_tev[valid]
        weights = weights[valid]

        if len(a_values) == 0:
            return

        mass_bin = np.searchsorted(self.mass_edges, masses_tev, side="right") - 1
        mass_bin[masses_tev == self.mass_edges[-1]] = len(self.weight_total) - 1
        in_range = (mass_bin >= 0) & (mass_bin < len(self.weight_total))

        idx = mass_bin[in_range]
        w = weights[in_range]
        a_mass = a_values[in_range]
        b_mass = b_values[in_range]

        np.add.at(self.weight_total, idx, w)
        np.add.at(self.weight_a, idx, w * (a_mass < self.a_cut))
        np.add.at(self.weight_b, idx, w * (b_mass > self.b_cut))

        hist2d, _, _ = np.histogram2d(
            a_values,
            b_values,
            bins=[self.a2d_edges, self.b2d_edges],
        )
        self.ab_counts += hist2d.astype(np.int64)

        a_bin = np.searchsorted(self.mean_a_edges, a_values, side="right") - 1
        valid_a = (a_bin >= 0) & (a_bin < len(self.mean_b_sum))
        np.add.at(self.mean_b_sum, a_bin[valid_a], b_values[valid_a])
        np.add.at(self.mean_b_count, a_bin[valid_a], 1)

        self.n += len(a_values)
        self.sum_a += float(np.sum(a_values, dtype=np.float64))
        self.sum_b += float(np.sum(b_values, dtype=np.float64))
        self.sum_a2 += float(np.sum(a_values * a_values, dtype=np.float64))
        self.sum_b2 += float(np.sum(b_values * b_values, dtype=np.float64))
        self.sum_ab += float(np.sum(a_values * b_values, dtype=np.float64))

        self.sample.update(a_values, b_values)

    def fractions(self):
        frac_a = np.divide(
            self.weight_a,
            self.weight_total,
            out=np.zeros_like(self.weight_a),
            where=self.weight_total > 0,
        )
        frac_b = np.divide(
            self.weight_b,
            self.weight_total,
            out=np.zeros_like(self.weight_b),
            where=self.weight_total > 0,
        )
        return frac_a, frac_b

    def pearson(self):
        if self.n < 3:
            return np.nan, np.nan

        numerator = self.n * self.sum_ab - self.sum_a * self.sum_b
        da = self.n * self.sum_a2 - self.sum_a * self.sum_a
        db = self.n * self.sum_b2 - self.sum_b * self.sum_b
        denominator = math.sqrt(max(da, 0.0) * max(db, 0.0))

        if denominator == 0.0:
            return np.nan, np.nan

        r = float(np.clip(numerator / denominator, -1.0, 1.0))
        if abs(r) == 1.0:
            return r, 0.0

        t_value = r * math.sqrt((self.n - 2) / (1.0 - r * r))
        p_value = float(2.0 * student_t.sf(abs(t_value), df=self.n - 2))
        return r, p_value

    def mean_b(self):
        return np.divide(
            self.mean_b_sum,
            self.mean_b_count,
            out=np.full_like(self.mean_b_sum, np.nan),
            where=self.mean_b_count > 0,
        )


def process_pair(task_id, shape_path, mass_path, process, dr_scale, frame, dr_module, accumulator):
    with open(shape_path, "rb") as handle:
        shape_payload = pickle.load(handle)
    validate(shape_payload, shape_path, process, dr_scale, "event_shapes_dict")

    suffix = "_lab" if frame == "lab" else "_CoM"
    a_key = f"aplanarity{suffix}"
    b_key = f"B_values{suffix}"

    compact_shapes = {}
    shape_dict = shape_payload["event_shapes_dict"]

    for directory, directory_data in shape_dict.items():
        if not isinstance(directory_data, dict):
            continue
        for lprup, record in directory_data.items():
            if not isinstance(record, dict):
                continue
            a_values = as_float_array(record.get(a_key, []))
            b_values = as_float_array(record.get(b_key, []))
            if len(a_values) != len(b_values):
                raise RuntimeError(
                    f"A/B mismatch in TASK_ID={task_id}, directory={directory!r}, "
                    f"lprup={lprup!r}: A={len(a_values)}, B={len(b_values)}"
                )
            if len(a_values):
                compact_shapes[(directory, lprup)] = (a_values, b_values)
            record.clear()

    del shape_dict
    del shape_payload
    gc.collect()

    with open(mass_path, "rb") as handle:
        mass_payload = pickle.load(handle)
    validate(mass_payload, mass_path, process, dr_scale, "masses_dict")

    mass_key = "M_lab" if frame == "lab" else "M_CoM"
    mass_dict = mass_payload["masses_dict"]
    seen = set()

    for directory, directory_data in mass_dict.items():
        if not isinstance(directory_data, dict):
            continue
        for lprup, record in directory_data.items():
            if not isinstance(record, dict):
                continue

            key = (directory, lprup)
            masses = as_float_array(record.get(mass_key, []))

            if key not in compact_shapes:
                if len(masses):
                    raise RuntimeError(
                        f"Masses have no matching A/B record in TASK_ID={task_id}, "
                        f"directory={directory!r}, lprup={lprup!r}"
                    )
                continue

            a_values, b_values = compact_shapes[key]
            if not (len(a_values) == len(b_values) == len(masses)):
                raise RuntimeError(
                    f"A/B/M mismatch in TASK_ID={task_id}, directory={directory!r}, "
                    f"lprup={lprup!r}: A={len(a_values)}, B={len(b_values)}, M={len(masses)}"
                )

            weights = np.asarray(
                dr_module.bg3(masses * 1.0e-3, 1.50e2, 7.38e0, -4.68e0),
                dtype=float,
            )
            if len(weights) != len(masses):
                raise RuntimeError(
                    f"bg3 returned {len(weights)} weights for {len(masses)} masses"
                )

            accumulator.update(a_values, b_values, masses, weights)
            seen.add(key)
            record.clear()

    missing = set(compact_shapes) - seen
    if missing:
        raise RuntimeError(
            f"{len(missing)} A/B records have no matching mass record in TASK_ID={task_id}"
        )

    compact_shapes.clear()
    del compact_shapes
    del mass_dict
    del mass_payload
    gc.collect()



def configure_axis(axis, x_minor_intervals=2, y_minor_intervals=2):
    """Apply the common axis font and tick styling."""
    axis.xaxis.set_minor_locator(
        AutoMinorLocator(x_minor_intervals)
    )
    axis.yaxis.set_minor_locator(
        AutoMinorLocator(y_minor_intervals)
    )

    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=7,
        width=1.0,
        labelsize=FONT_SIZE,
    )

    axis.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=4,
        width=0.8,
    )


def configure_colorbar(colorbar):
    """Apply matching font and tick styling to a colorbar."""
    colorbar.ax.minorticks_on()

    colorbar.ax.tick_params(
        which="major",
        direction="in",
        length=7,
        width=1.0,
        labelsize=FONT_SIZE,
    )

    colorbar.ax.tick_params(
        which="minor",
        direction="in",
        length=4,
        width=0.8,
    )


def add_info_box(
    axis,
    x=0.97,
    y=0.97,
    horizontal_alignment="right",
    vertical_alignment="top",
):

    lines = [
        f"DRToM {DEFAULT_PROCESS_TYPE}",
        rf"$\sqrt{{s}} = {DEFAULT_ENERGY_TEV:g}$ TeV",
        rf"3D, $2\to 4$",
        rf"$|\eta| < {DEFAULT_ETA_MAX:g}$",
    ]

    return axis.text(
        x,
        y,
        "\n".join(lines),
        transform=axis.transAxes,
        ha=horizontal_alignment,
        va=vertical_alignment,
        fontsize=FONT_SIZE,
    )


def add_info_box_below_legend(
    figure,
    axis,
    legend,
    vertical_gap=0.03,
):
    """Place the standard information box directly below a legend."""
    figure.canvas.draw()

    legend_box = legend.get_window_extent(
        renderer=figure.canvas.get_renderer()
    ).transformed(
        axis.transAxes.inverted()
    )

    return add_info_box(
        axis=axis,
        x=legend_box.x0,
        y=legend_box.y0 - vertical_gap,
        horizontal_alignment="left",
        vertical_alignment="top",
    )


def configure_mass_axis(axis, mass_min, mass_max):
    axis.set_xticks(
        np.arange(
            math.ceil(mass_min),
            math.floor(mass_max) + 1,
            1,
        )
    )
    configure_axis(axis)



def plot_mass_fractions(
    accumulator,
    dr_scale,
    frame,
    process,
    output_root,
):
    frac_a, frac_b = accumulator.fractions()
    label = rf"$\Lambda_3 = {dr_scale:.1f}\,\mathrm{{TeV}}$"

    fig_b, ax_b = plt.subplots(figsize=(8, 6))
    fig_a, ax_a = plt.subplots(figsize=(8, 6))

    for axis in (ax_b, ax_a):
        configure_mass_axis(
            axis,
            accumulator.mass_edges[0],
            accumulator.mass_edges[-1],
        )

    ax_b.scatter(
        accumulator.mass_centres,
        frac_b,
        label=label,
    )
    ax_a.scatter(
        accumulator.mass_centres,
        frac_a,
        label=label,
    )

    ax_b.set_xlabel(
        r"Four-jet invariant mass, $M_{jjjj}$ [TeV]",
        fontsize=FONT_SIZE,
    )
    ax_b.set_ylabel(
        f"Fraction (B > {accumulator.b_cut:g})",
        fontsize=FONT_SIZE,
    )
    ax_b.set_ylim(0, 1.05)
    ax_b.grid(True)

    legend_b = ax_b.legend(
        loc="upper left",
        fontsize=FONT_SIZE,
    )

    add_info_box_below_legend(
        figure=fig_b,
        axis=ax_b,
        legend=legend_b,
    )

    ax_a.set_xlabel(
        r"Four-jet invariant mass, $M_{jjjj}$ [TeV]",
        fontsize=FONT_SIZE,
    )
    ax_a.set_ylabel(
        f"Fraction (A < {accumulator.a_cut:g})",
        fontsize=FONT_SIZE,
    )
    ax_a.set_ylim(0, 1.05)
    ax_a.grid(True)

    legend_a = ax_a.legend(
        loc="upper left",
        fontsize=FONT_SIZE,
    )

    add_info_box_below_legend(
        figure=fig_a,
        axis=ax_a,
        legend=legend_a,
    )

    fig_b.tight_layout()
    fig_a.tight_layout()

    b_path = os.path.join(
        output_root,
        f"B_vs_M_{frame}.png",
    )
    a_path = os.path.join(
        output_root,
        f"A_vs_M_{frame}.png",
    )

    fig_b.savefig(
        b_path,
        dpi=300,
        bbox_inches="tight",
    )
    fig_a.savefig(
        a_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig_b)
    plt.close(fig_a)

    print(f"Saved -> {b_path}")
    print(f"Saved -> {a_path}")



def plot_correlations(
    accumulator,
    frame,
    process,
    dr_scale,
    output_root,
):
    correlation_dir = os.path.join(
        output_root,
        "AB_cor",
    )
    os.makedirs(
        correlation_dir,
        exist_ok=True,
    )

    pearson_r, pearson_p = accumulator.pearson()
    sample_a = accumulator.sample.a
    sample_b = accumulator.sample.b

    if len(sample_a) >= 2:
        spearman_rho, spearman_p = spearmanr(
            sample_a,
            sample_b,
        )
    else:
        spearman_rho, spearman_p = np.nan, np.nan

    print("=== A vs B Correlations ===")
    print(
        f"Pearson r (exact, N={accumulator.n}) = "
        f"{pearson_r:.4f} (p = {pearson_p:.3e})"
    )
    print(
        f"Spearman rho (sample N={len(sample_a)}) = "
        f"{spearman_rho:.4f} (p = {spearman_p:.3e})"
    )

    total = np.sum(accumulator.ab_counts)
    bin_area = (
        np.diff(accumulator.a2d_edges)[:, None]
        * np.diff(accumulator.b2d_edges)[None, :]
    )

    density = np.divide(
        accumulator.ab_counts,
        total * bin_area,
        out=np.zeros_like(
            accumulator.ab_counts,
            dtype=float,
        ),
        where=(total > 0) & (bin_area > 0),
    )

    # --------------------------------------------------------
    # Joint-density heatmap
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    mesh = ax.pcolormesh(
        accumulator.a2d_edges,
        accumulator.b2d_edges,
        density.T,
        shading="auto",
    )

    cbar = fig.colorbar(
        mesh,
        ax=ax,
    )
    cbar.set_label(
        "Density",
        fontsize=FONT_SIZE,
    )
    configure_colorbar(cbar)

    ax.set_xlabel(
        "Aplanarity",
        fontsize=FONT_SIZE,
    )
    ax.set_ylabel(
        "Biplanarity",
        fontsize=FONT_SIZE,
    )
    ax.set_title(
        rf"$r={pearson_r:.3f},\ "
        rf"\rho_{{sample}}={spearman_rho:.3f}$",
        fontsize=FONT_SIZE,
    )

    configure_axis(ax)

    add_info_box(
        axis=ax,
    )

    fig.tight_layout()

    heatmap_path = os.path.join(
        correlation_dir,
        f"A_vs_B_2D_{frame}.png",
    )
    fig.savefig(
        heatmap_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Sampled scatter plot
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        sample_a,
        sample_b,
        s=2,
        alpha=0.2,
    )

    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)

    ax.set_xlabel(
        "Aplanarity",
        fontsize=FONT_SIZE,
    )
    ax.set_ylabel(
        "Biplanarity",
        fontsize=FONT_SIZE,
    )
    ax.set_title(
        rf"$r={pearson_r:.3f},\ "
        rf"\rho_{{sample}}={spearman_rho:.3f}$",
        fontsize=FONT_SIZE,
    )

    configure_axis(ax)

    add_info_box(
        axis=ax,
    )

    fig.tight_layout()

    scatter_path = os.path.join(
        correlation_dir,
        f"A_vs_B_scatter_{frame}.png",
    )
    fig.savefig(
        scatter_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Mean B versus A
    # --------------------------------------------------------
    a_centres = 0.5 * (
        accumulator.mean_a_edges[:-1]
        + accumulator.mean_a_edges[1:]
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        a_centres,
        accumulator.mean_b(),
        marker="o",
    )

    ax.set_xlabel(
        "Aplanarity",
        fontsize=FONT_SIZE,
    )
    ax.set_ylabel(
        r"$\langle B \rangle$",
        fontsize=FONT_SIZE,
    )

    ax.grid(True)
    configure_axis(ax)

    add_info_box(
        axis=ax,
    )

    fig.tight_layout()

    mean_path = os.path.join(
        correlation_dir,
        f"MeanB_vs_A_{frame}.png",
    )
    fig.savefig(
        mean_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved A vs B heatmap -> {heatmap_path}")
    print(f"Saved sampled A vs B scatter -> {scatter_path}")
    print(f"Saved <B>(A) -> {mean_path}")


def histogram_quantile(counts, centres, quantile):
    total = np.sum(counts)
    if total == 0:
        return np.nan

    cumulative = np.cumsum(counts)
    index = np.searchsorted(cumulative, quantile * total, side="left")
    index = min(index, len(centres) - 1)
    return centres[index]



def plot_conditional_b_given_a(
    accumulator,
    frame,
    process,
    dr_scale,
    output_root,
    min_events_per_bin=100,
):
    correlation_dir = os.path.join(
        output_root,
        "AB_cor",
    )
    os.makedirs(
        correlation_dir,
        exist_ok=True,
    )

    counts = accumulator.ab_counts.astype(float)

    a_edges = accumulator.a2d_edges
    b_edges = accumulator.b2d_edges
    a_centres = 0.5 * (
        a_edges[:-1] + a_edges[1:]
    )
    b_centres = 0.5 * (
        b_edges[:-1] + b_edges[1:]
    )

    # Number of events in each A bin.
    counts_per_a = np.sum(
        counts,
        axis=1,
    )

    # P(B | A): normalize each A bin separately.
    conditional = np.divide(
        counts,
        counts_per_a[:, None],
        out=np.zeros_like(counts),
        where=counts_per_a[:, None] > 0,
    )

    q16 = np.full(
        len(a_centres),
        np.nan,
    )
    median = np.full(
        len(a_centres),
        np.nan,
    )
    q84 = np.full(
        len(a_centres),
        np.nan,
    )

    for index, row in enumerate(counts):
        if counts_per_a[index] < min_events_per_bin:
            continue

        q16[index] = histogram_quantile(
            row,
            b_centres,
            0.16,
        )
        median[index] = histogram_quantile(
            row,
            b_centres,
            0.50,
        )
        q84[index] = histogram_quantile(
            row,
            b_centres,
            0.84,
        )

    positive = conditional[
        conditional > 0
    ]
    conditional_masked = np.ma.masked_less_equal(
        conditional.T,
        0,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    if len(positive):
        mesh = ax.pcolormesh(
            a_edges,
            b_edges,
            conditional_masked,
            shading="auto",
            norm=LogNorm(
                vmin=max(
                    float(np.min(positive)),
                    1.0e-4,
                ),
                vmax=float(np.max(positive)),
            ),
        )
    else:
        mesh = ax.pcolormesh(
            a_edges,
            b_edges,
            conditional.T,
            shading="auto",
        )

    cbar = fig.colorbar(
        mesh,
        ax=ax,
    )
    cbar.set_label(
        r"Conditional probability, $P(B\mid A)$",
        fontsize=FONT_SIZE,
    )
    configure_colorbar(cbar)

    valid = np.isfinite(median)

    ax.plot(
        a_centres[valid],
        q16[valid],
        color="red",
        linestyle="--",
        linewidth=2,
        label="16th percentile",
        zorder=4,
    )

    ax.plot(
        a_centres[valid],
        q84[valid],
        color="red",
        linestyle="--",
        linewidth=2,
        label="84th percentile",
        zorder=4,
    )

    ax.plot(
        a_centres[valid],
        median[valid],
        color="black",
        marker="o",
        linewidth=2,
        label="Median B",
        zorder=5,
    )

    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)

    ax.set_xlabel(
        "Aplanarity",
        fontsize=FONT_SIZE,
    )
    ax.set_ylabel(
        "Biplanarity",
        fontsize=FONT_SIZE,
    )

    configure_axis(ax)

    # Legend in the upper-right corner.
    legend = ax.legend(
        loc="upper right",
        fontsize=FONT_SIZE,
        frameon=True,
        framealpha=0.85,
    )

    # Draw the figure so Matplotlib can determine the legend position.
    fig.canvas.draw()

    # Get the legend position in axis coordinates.
    legend_box = legend.get_window_extent(
        renderer=fig.canvas.get_renderer()
    ).transformed(
        ax.transAxes.inverted()
    )

    # Information box immediately below the legend.
    add_info_box(
        axis=ax,
        x=legend_box.x1,
        y=legend_box.y0 - 0.03,
        horizontal_alignment="right",
        vertical_alignment="top",
    )

    fig.tight_layout()

    output_path = os.path.join(
        correlation_dir,
        f"B_given_A_{frame}.png",
    )
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(
        "Saved conditional B distribution "
        f"-> {output_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Low-memory planarity plotting from paired partial pickles"
    )
    parser.add_argument(
        "--project-root",
        default=os.environ.get("DRTOM_PROJECT_ROOT", DEFAULT_PROJECT_ROOT),
    )
    parser.add_argument(
        "--raid-storage",
        default=os.environ.get("RAID_STORAGE", DEFAULT_RAID_STORAGE),
    )
    parser.add_argument(
        "--energy-folder",
        default=os.environ.get("ENERGY_FOLDER", DEFAULT_ENERGY_FOLDER),
    )
    parser.add_argument(
        "--process",
        default=os.environ.get("WHAT_PROCESS", DEFAULT_PROCESS),
    )
    parser.add_argument(
        "--dr",
        type=float,
        default=float(os.environ.get("DR_SCALE", DEFAULT_DR_SCALE)),
    )
    parser.add_argument("--frame", choices=["lab", "CoM"], default="CoM")
    parser.add_argument(
        "--plots-root",
        default=os.environ.get("PLOTS_ROOT", DEFAULT_PLOTS_ROOT),
    )
    parser.add_argument("--mass-min", type=float, default=2.0)
    parser.add_argument("--mass-max", type=float, default=11.0)
    parser.add_argument("--mass-bins", type=int, default=50)
    parser.add_argument("--a-cut", type=float, default=0.01)
    parser.add_argument("--b-cut", type=float, default=0.98)
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--max-pairs", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mass_bins < 1:
        raise RuntimeError("--mass-bins must be at least 1")
    if args.mass_max <= args.mass_min:
        raise RuntimeError("--mass-max must be greater than --mass-min")
    if args.progress_every < 1:
        raise RuntimeError("--progress-every must be at least 1")
    if args.sample_size < 0:
        raise RuntimeError("--sample-size cannot be negative")

    sys.path.insert(0, args.project_root)
    import Analysis.DimensionalReduction as dr_module

    partial_root, pairs = discover_pairs(
        args.raid_storage,
        args.energy_folder,
        args.process,
        args.dr,
    )

    if args.max_pairs is not None:
        if args.max_pairs < 1:
            raise RuntimeError("--max-pairs must be at least 1")
        pairs = pairs[: args.max_pairs]

    output_root = os.path.join(
        args.plots_root,
        args.energy_folder,
        args.process,
        f"DR_{args.dr}",
    )
    os.makedirs(output_root, exist_ok=True)

    print("=" * 72)
    print("Low-memory planarity plotting")
    print(f"Partial root: {partial_root}")
    print(f"Paired partials: {len(pairs)}")
    print(f"Frame: {args.frame}")
    print(f"Cuts: A < {args.a_cut}, B > {args.b_cut}")
    print(f"Scatter/Spearman sample size: {args.sample_size}")
    print(f"Output: {output_root}")
    print("=" * 72)

    accumulator = Accumulator(
        args.mass_min,
        args.mass_max,
        args.mass_bins,
        args.a_cut,
        args.b_cut,
        args.sample_size,
    )

    start = time.perf_counter()

    for index, (task_id, shape_path, mass_path) in enumerate(pairs, start=1):
        process_pair(
            task_id,
            shape_path,
            mass_path,
            args.process,
            args.dr,
            args.frame,
            dr_module,
            accumulator,
        )

        if index == 1 or index == len(pairs) or index % args.progress_every == 0:
            elapsed = time.perf_counter() - start
            print(
                f"[INFO] Processed {index}/{len(pairs)} paired partials in "
                f"{elapsed / 60.0:.2f} min; events={accumulator.n}; "
                f"max RSS={max_rss_gib():.2f} GiB"
            )

        plot_mass_fractions(
            accumulator,
            args.dr,
            args.frame,
            args.process,
            output_root,
        )

        # Joint-density heatmap, Sampled scatter plot, 3. Mean B versus A
        plot_correlations(
            accumulator,
            args.frame,
            args.process,
            args.dr,
            output_root,
        )

        # P(B | A), median B, and 16th–84th percentile band
        plot_conditional_b_given_a(
            accumulator,
            args.frame,
            args.process,
            args.dr,
            output_root,
            min_events_per_bin=100,
        )

    elapsed = time.perf_counter() - start
    print("=" * 72)
    print(f"Completed in {elapsed / 60.0:.2f} min")
    print(f"Events processed: {accumulator.n}")
    print(f"Maximum RSS: {max_rss_gib():.2f} GiB")
    print("=" * 72)


if __name__ == "__main__":
    main()
