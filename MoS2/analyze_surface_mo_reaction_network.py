#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended surface Mo coordination / sulfurization analysis for LAMMPS trajectories.

Designed for MoOx + Sx / alpha-Al2O3 surface-reaction trajectories.

Main analyses in ONE streaming pass
-----------------------------------
1. Per-Mo Mo-S, Mo-O, and Mo-Mo coordination numbers.
2. Local MoOxSy state assignment for every surface Mo.
3. MoOxSy population versus simulation time.
4. Global sulfurization index:
       chi_S = sum(CN_Mo-S) / [sum(CN_Mo-S) + sum(CN_Mo-O)]
5. State-to-state transitions and O->S ligand-exchange events.
6. Transition count and row-normalized probability matrices.
7. MoOxSy residence/dwell intervals for individual Mo atoms.
8. Mo height above the instantaneous top Al layer, including species-resolved
   height statistics.
9. Mo-Mo coordination as a descriptor of precursor association/oligomerization.
10. Mo-O-Mo and Mo-S-Mo bridging-ligand statistics.
11. Optional Nature-style figures.

Important physical notes
------------------------
* Mo-O currently includes ALL O atoms mapped as element O. If your trajectory
  contains both substrate O and precursor O under the same atom type, they are
  not separated. To distinguish them, preserve a separate type/molecule/group
  label in the trajectory and extend get_elements()/mask construction.
* Transition and residence statistics are evaluated on ANALYZED frames. Thus
  --stride controls kinetic time resolution. Do not compare kinetic rates from
  different --stride values without checking convergence.
* Coordination cutoffs should ideally be chosen from the first minimum of the
  corresponding RDF / distance distribution.

Default atom-type mapping
-------------------------
    type 1 = Al
    type 2 = Mo
    type 3 = O
    type 4 = S
VERIFY against your LAMMPS data file.

Example
-------
python analyze_surface_mo_reaction_network.py surface.lammpstrj \
    --dt-fs 1.0 --stride 10 \
    --cut-mo-s 2.8 --cut-mo-o 2.5 --cut-mo-mo 3.8 \
    --surface-zmin -1.0 --surface-zmax 5.0 \
    --pbc xy --prefix surface_Mo

Outputs
-------
<prefix>_atoms.csv
<prefix>_time_series.csv
<prefix>_species_population.csv
<prefix>_transitions.csv
<prefix>_transition_counts.csv
<prefix>_transition_probabilities.csv
<prefix>_residence_times.csv
<prefix>_residence_summary.csv
<prefix>_species_height_summary.csv
<prefix>_evolution.[png|pdf|svg]
<prefix>_mechanism.[png|pdf|svg]
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

State = Tuple[int, int]  # (CN_Mo_O, CN_Mo_S)


# -----------------------------------------------------------------------------
# Basic trajectory data structures / reader
# -----------------------------------------------------------------------------
@dataclass
class Box:
    origin: np.ndarray
    h: np.ndarray  # columns are lattice vectors a,b,c
    boundary: Tuple[str, str, str]


@dataclass
class Frame:
    timestep: int
    box: Box
    columns: List[str]
    data: Dict[str, np.ndarray]


def _parse_box(header: str, lines: Sequence[str]) -> Box:
    """Parse LAMMPS orthorhombic or restricted-triclinic BOX BOUNDS."""
    tokens = header.strip().split()[3:]
    triclinic = all(t in tokens for t in ("xy", "xz", "yz"))
    vals = [list(map(float, line.split())) for line in lines]

    boundary_tokens = [t for t in tokens if t not in ("xy", "xz", "yz")]
    if len(boundary_tokens) >= 3:
        boundary = tuple(boundary_tokens[-3:])
    else:
        boundary = ("pp", "pp", "pp")

    if triclinic:
        xlo_b, xhi_b, xy = vals[0][:3]
        ylo_b, yhi_b, xz = vals[1][:3]
        zlo_b, zhi_b, yz = vals[2][:3]

        xlo = xlo_b - min(0.0, xy, xz, xy + xz)
        xhi = xhi_b - max(0.0, xy, xz, xy + xz)
        ylo = ylo_b - min(0.0, yz)
        yhi = yhi_b - max(0.0, yz)
        zlo, zhi = zlo_b, zhi_b

        lx, ly, lz = xhi - xlo, yhi - ylo, zhi - zlo
        h = np.array(
            [[lx, xy, xz], [0.0, ly, yz], [0.0, 0.0, lz]], dtype=float
        )
        origin = np.array([xlo, ylo, zlo], dtype=float)
    else:
        xlo, xhi = vals[0][:2]
        ylo, yhi = vals[1][:2]
        zlo, zhi = vals[2][:2]
        h = np.diag([xhi - xlo, yhi - ylo, zhi - zlo]).astype(float)
        origin = np.array([xlo, ylo, zlo], dtype=float)

    return Box(origin=origin, h=h, boundary=boundary)  # type: ignore[arg-type]


def read_lammps_dump(path: Path) -> Iterator[Frame]:
    """Streaming reader for LAMMPS custom dump files."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            ts_line = f.readline()
            if not ts_line:
                break
            timestep = int(float(ts_line.strip()))

            line = f.readline()
            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(
                    f"Expected 'ITEM: NUMBER OF ATOMS' after timestep {timestep}"
                )
            natoms = int(f.readline().strip())

            box_header = f.readline()
            if not box_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Expected 'ITEM: BOX BOUNDS' at timestep {timestep}")
            box = _parse_box(
                box_header, [f.readline(), f.readline(), f.readline()]
            )

            atom_header = f.readline().strip()
            if not atom_header.startswith("ITEM: ATOMS"):
                raise ValueError(f"Expected 'ITEM: ATOMS' at timestep {timestep}")
            columns = atom_header.split()[2:]
            ncol = len(columns)

            raw_rows: List[List[str]] = []
            for _ in range(natoms):
                row = f.readline().split()
                if len(row) != ncol:
                    raise ValueError(
                        f"Malformed atom row at timestep {timestep}: "
                        f"expected {ncol} columns, got {len(row)}"
                    )
                raw_rows.append(row)

            arr = np.asarray(raw_rows, dtype=object)
            data: Dict[str, np.ndarray] = {}
            string_cols = {"element"}
            int_cols = {"id", "type", "mol", "proc", "procp1"}
            for j, col in enumerate(columns):
                if col in string_cols:
                    data[col] = arr[:, j].astype(str)
                elif col in int_cols:
                    data[col] = arr[:, j].astype(np.int64)
                else:
                    try:
                        data[col] = arr[:, j].astype(float)
                    except ValueError:
                        data[col] = arr[:, j].astype(str)

            yield Frame(timestep=timestep, box=box, columns=columns, data=data)


def get_cartesian_positions(frame: Frame) -> np.ndarray:
    d = frame.data
    origin = frame.box.origin
    h = frame.box.h

    # Cartesian coordinates, wrapped or unwrapped.
    for names in (("x", "y", "z"), ("xu", "yu", "zu")):
        if all(k in d for k in names):
            return np.column_stack([d[k].astype(float) for k in names])

    # Scaled coordinates, wrapped or unwrapped.
    for names in (("xs", "ys", "zs"), ("xsu", "ysu", "zsu")):
        if all(k in d for k in names):
            s = np.column_stack([d[k].astype(float) for k in names])
            return origin[None, :] + s @ h.T

    raise ValueError(
        "No supported coordinates. Need x/y/z, xu/yu/zu, xs/ys/zs, "
        "or xsu/ysu/zsu."
    )


def get_elements(frame: Frame, type_map: Dict[int, str]) -> np.ndarray:
    d = frame.data
    if "element" in d:
        return np.char.capitalize(d["element"].astype(str))
    if "type" not in d:
        raise ValueError("Trajectory has neither 'element' nor 'type' column.")

    types = d["type"].astype(int)
    elems = np.empty(types.shape[0], dtype="U4")
    unknown = set()
    for i, t in enumerate(types):
        ti = int(t)
        if ti not in type_map:
            unknown.add(ti)
            elems[i] = f"T{ti}"
        else:
            elems[i] = type_map[ti]
    if unknown:
        raise ValueError(
            f"Unmapped atom types found: {sorted(unknown)}. "
            "Set --type-Al/--type-Mo/--type-O/--type-S correctly."
        )
    return elems


def get_ids(frame: Frame) -> np.ndarray:
    if "id" in frame.data:
        return frame.data["id"].astype(np.int64)
    n = len(next(iter(frame.data.values())))
    return np.arange(1, n + 1, dtype=np.int64)


# -----------------------------------------------------------------------------
# Geometry / coordination utilities
# -----------------------------------------------------------------------------
def pbc_mask_from_arg(box: Box, pbc_arg: str) -> np.ndarray:
    if pbc_arg == "xy":
        return np.array([True, True, False], dtype=bool)
    if pbc_arg == "xyz":
        return np.array([True, True, True], dtype=bool)
    if pbc_arg == "none":
        return np.array([False, False, False], dtype=bool)
    if pbc_arg == "auto":
        return np.array(
            [b.lower().startswith("p") for b in box.boundary], dtype=bool
        )
    raise ValueError(f"Unknown PBC mode: {pbc_arg}")


def minimum_image_deltas(
    centers: np.ndarray,
    neighbors: np.ndarray,
    box: Box,
    pbc_mask: np.ndarray,
    chunk: int = 128,
) -> Iterator[Tuple[slice, np.ndarray]]:
    """Yield neighbor-center minimum-image displacement arrays in chunks."""
    if len(centers) == 0 or len(neighbors) == 0:
        return

    hinv_t = np.linalg.inv(box.h).T
    h_t = box.h.T

    for i0 in range(0, len(centers), chunk):
        i1 = min(i0 + chunk, len(centers))
        delta = neighbors[None, :, :] - centers[i0:i1, None, :]
        frac = delta @ hinv_t
        for ax in range(3):
            if pbc_mask[ax]:
                frac[..., ax] -= np.rint(frac[..., ax])
        yield slice(i0, i1), frac @ h_t


def coordination_details(
    centers: np.ndarray,
    center_ids: np.ndarray,
    neighbors: np.ndarray,
    neighbor_ids: np.ndarray,
    cutoff: float,
    box: Box,
    pbc_mask: np.ndarray,
    exclude_same_id: bool = False,
    return_neighbor_lists: bool = False,
    return_neighbor_degree: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[np.ndarray]], Optional[np.ndarray]]:
    """
    Coordination statistics within cutoff.

    Returns
    -------
    counts : coordination number of each center
    nearest : nearest valid neighbor distance (not restricted to cutoff)
    neighbor_lists : optional neighbor atom IDs within cutoff for each center
    neighbor_degree : optional number of centers bonded to each neighbor
    """
    ncent = len(centers)
    nnei = len(neighbors)
    counts = np.zeros(ncent, dtype=np.int32)
    nearest = np.full(ncent, np.nan, dtype=float)
    lists: Optional[List[np.ndarray]] = (
        [np.empty(0, dtype=np.int64) for _ in range(ncent)]
        if return_neighbor_lists
        else None
    )
    degree: Optional[np.ndarray] = (
        np.zeros(nnei, dtype=np.int32) if return_neighbor_degree else None
    )

    if ncent == 0 or nnei == 0:
        return counts, nearest, lists, degree

    cutoff2 = cutoff * cutoff
    for sl, delta in minimum_image_deltas(centers, neighbors, box, pbc_mask):
        d2 = np.einsum("ijk,ijk->ij", delta, delta)
        valid_for_nearest = np.ones_like(d2, dtype=bool)

        if exclude_same_id:
            same = center_ids[sl][:, None] == neighbor_ids[None, :]
            valid_for_nearest &= ~same
        else:
            same = None

        # nearest distance excluding self if requested
        d2_near = np.where(valid_for_nearest, d2, np.inf)
        mins = np.min(d2_near, axis=1)
        good = np.isfinite(mins)
        tmp = np.full(len(mins), np.nan, dtype=float)
        tmp[good] = np.sqrt(mins[good])
        nearest[sl] = tmp

        bonded = d2 <= cutoff2
        if same is not None:
            bonded &= ~same

        counts[sl] = np.sum(bonded, axis=1)

        if lists is not None:
            offset = sl.start or 0
            for ii in range(bonded.shape[0]):
                lists[offset + ii] = neighbor_ids[bonded[ii]].astype(np.int64)

        if degree is not None:
            degree += np.sum(bonded, axis=0).astype(np.int32)

    return counts, nearest, lists, degree


def top_al_surface_z(al_z: np.ndarray, layer_tol: float) -> float:
    """Mean z of Al atoms within layer_tol below the uppermost Al atom."""
    if len(al_z) == 0:
        raise ValueError("No Al atoms found in this frame.")
    zmax = float(np.max(al_z))
    top = al_z[al_z >= zmax - layer_tol]
    return float(np.mean(top))


def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) else math.nan


def safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if len(x) else math.nan


def safe_median(x: np.ndarray) -> float:
    return float(np.median(x)) if len(x) else math.nan


def species_name(state: State) -> str:
    """Compact human-readable MoOxSy label."""
    o, s = state
    text = "Mo"
    if o > 0:
        text += "O" + (str(o) if o != 1 else "")
    if s > 0:
        text += "S" + (str(s) if s != 1 else "")
    if o == 0 and s == 0:
        text += "_uncoord"
    return text


def state_sort_key(state: State) -> Tuple[int, int, int]:
    """O-rich -> mixed -> S-rich ordering."""
    o, s = state
    return (-o, s, -(o + s))


def classify_transition(old: State, new: State) -> str:
    do = new[0] - old[0]
    ds = new[1] - old[1]
    if do < 0 and ds > 0:
        return "O_to_S_exchange"
    if do > 0 and ds < 0:
        return "S_to_O_exchange"
    if do < 0 and ds == 0:
        return "O_loss"
    if do > 0 and ds == 0:
        return "O_gain"
    if ds > 0 and do == 0:
        return "S_gain"
    if ds < 0 and do == 0:
        return "S_loss"
    if do == 0 and ds == 0:
        return "unchanged"
    return "concerted_other"


def update_online_stats(stats: Dict[State, Dict[str, float]], state: State, values: np.ndarray) -> None:
    if len(values) == 0:
        return
    rec = stats.setdefault(
        state,
        {"n": 0.0, "sum": 0.0, "sumsq": 0.0, "min": math.inf, "max": -math.inf},
    )
    rec["n"] += float(len(values))
    rec["sum"] += float(np.sum(values))
    rec["sumsq"] += float(np.sum(values * values))
    rec["min"] = min(rec["min"], float(np.min(values)))
    rec["max"] = max(rec["max"], float(np.max(values)))


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def configure_nature_rc() -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    font_name = (
        "Arial"
        if "Arial" in available_fonts
        else ("Helvetica" if "Helvetica" in available_fonts else "DejaVu Sans")
    )
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": 7,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def make_evolution_plot(summary_csv: Path, atom_csv: Path, outbase: Path) -> None:
    """Mean CN + atom-resolved CN heatmaps."""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warning] Evolution plot skipped: {exc}")
        return

    summary = pd.read_csv(summary_csv)
    atoms = pd.read_csv(atom_csv)
    if summary.empty or atoms.empty:
        return

    configure_nature_rc()
    col_o = "#4C78A8"
    col_s = "#E58A52"

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.15, 1.15], hspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    t = summary["time_ns"].to_numpy()
    ax.plot(t, summary["mean_CN_Mo_O"], lw=1.4, color=col_o, label="Mo–O")
    ax.plot(t, summary["mean_CN_Mo_S"], lw=1.4, color=col_s, label="Mo–S")
    ax.set_ylabel("Mean coordination number")
    ax.legend(frameon=False, ncol=2)
    ax.tick_params(top=True, right=True)
    ax.text(-0.065, 1.05, "a", transform=ax.transAxes, fontweight="bold", fontsize=9)

    frame_order = summary["frame_index"].to_numpy()
    mo_ids = np.sort(atoms["Mo_id"].unique())
    id_to_row = {mid: i for i, mid in enumerate(mo_ids)}
    fr_to_col = {fr: i for i, fr in enumerate(frame_order)}
    mat_o = np.full((len(mo_ids), len(frame_order)), np.nan)
    mat_s = np.full_like(mat_o, np.nan)

    for row in atoms.itertuples(index=False):
        if row.frame_index in fr_to_col and row.Mo_id in id_to_row:
            i = id_to_row[row.Mo_id]
            j = fr_to_col[row.frame_index]
            mat_o[i, j] = row.CN_Mo_O
            mat_s[i, j] = row.CN_Mo_S

    max_o = int(np.nanmax(mat_o)) if np.isfinite(mat_o).any() else 1
    max_s = int(np.nanmax(mat_s)) if np.isfinite(mat_s).any() else 1
    t0, t1 = float(t[0]), float(t[-1])
    if t1 == t0:
        t1 = t0 + 1e-9

    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(
        mat_o,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[t0, t1, 0.5, len(mo_ids) + 0.5],
        cmap=plt.get_cmap("Blues", max_o + 1),
        vmin=-0.5,
        vmax=max_o + 0.5,
    )
    ax2.set_ylabel("Surface Mo index")
    ax2.text(-0.065, 1.04, "b", transform=ax2.transAxes, fontweight="bold", fontsize=9)
    cb2 = fig.colorbar(im2, ax=ax2, pad=0.015, fraction=0.025)
    cb2.set_label(r"$N_{\mathrm{Mo-O}}$")
    cb2.set_ticks(np.arange(max_o + 1))

    ax3 = fig.add_subplot(gs[2, 0])
    im3 = ax3.imshow(
        mat_s,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[t0, t1, 0.5, len(mo_ids) + 0.5],
        cmap=plt.get_cmap("Oranges", max_s + 1),
        vmin=-0.5,
        vmax=max_s + 0.5,
    )
    ax3.set_xlabel("Simulation time (ns)")
    ax3.set_ylabel("Surface Mo index")
    ax3.text(-0.065, 1.04, "c", transform=ax3.transAxes, fontweight="bold", fontsize=9)
    cb3 = fig.colorbar(im3, ax=ax3, pad=0.015, fraction=0.025)
    cb3.set_label(r"$N_{\mathrm{Mo-S}}$")
    cb3.set_ticks(np.arange(max_s + 1))

    for a in (ax, ax2, ax3):
        a.grid(False)
        for sp in a.spines.values():
            sp.set_linewidth(0.8)

    fig.subplots_adjust(left=0.10, right=0.92, bottom=0.09, top=0.98)
    for ext, kwargs in (
        ("png", {"dpi": 600}),
        ("pdf", {}),
        ("svg", {}),
    ):
        fig.savefig(
            f"{outbase}_evolution.{ext}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def make_mechanism_plot(summary_csv: Path, population_csv: Path, outbase: Path) -> None:
    """Compact mechanistic figure: CN, sulfurization, bridges, species map."""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except Exception as exc:
        print(f"[warning] Mechanism plot skipped: {exc}")
        return

    s = pd.read_csv(summary_csv)
    p = pd.read_csv(population_csv)
    if s.empty or p.empty:
        return

    configure_nature_rc()
    col_o = "#4C78A8"
    col_s = "#E58A52"
    col_m = "#555555"

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.30)
    t = s["time_ns"].to_numpy()

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, s["mean_CN_Mo_O"], color=col_o, lw=1.4, label="Mo–O")
    ax1.plot(t, s["mean_CN_Mo_S"], color=col_s, lw=1.4, label="Mo–S")
    ax1.plot(t, s["mean_CN_Mo_Mo"], color=col_m, lw=1.1, label="Mo–Mo")
    ax1.set_ylabel("Mean coordination number")
    ax1.legend(frameon=False, ncol=3, handlelength=1.5)
    ax1.text(-0.14, 1.05, "a", transform=ax1.transAxes, fontweight="bold", fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, s["sulfurization_index"], color=col_s, lw=1.5)
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_ylabel(r"Sulfurization index, $\chi_S$")
    ax2.text(-0.14, 1.05, "b", transform=ax2.transAxes, fontweight="bold", fontsize=9)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, s["n_Mo_O_Mo_pairs"], color=col_o, lw=1.3, label="Mo–O–Mo")
    ax3.plot(t, s["n_Mo_S_Mo_pairs"], color=col_s, lw=1.3, label="Mo–S–Mo")
    ax3.set_xlabel("Simulation time (ns)")
    ax3.set_ylabel("Bridge-pair count")
    ax3.legend(frameon=False)
    ax3.text(-0.14, 1.05, "c", transform=ax3.transAxes, fontweight="bold", fontsize=9)

    # Species fractions: use fraction columns, rank by integrated population.
    frac_cols = [c for c in p.columns if c.startswith("f_")]
    if frac_cols:
        means = p[frac_cols].mean().sort_values(ascending=False)
        keep = list(means.index[: min(12, len(means))])
        mat = p[keep].to_numpy().T
        labels = [c[2:] for c in keep]
        cmap = LinearSegmentedColormap.from_list(
            "white_orange", ["#FFFFFF", "#F2D2B6", "#D97945"]
        )
        ax4 = fig.add_subplot(gs[1, 1])
        t0, t1 = float(p["time_ns"].iloc[0]), float(p["time_ns"].iloc[-1])
        if t1 == t0:
            t1 = t0 + 1e-9
        im = ax4.imshow(
            mat,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[t0, t1, -0.5, len(labels) - 0.5],
            cmap=cmap,
            vmin=0,
            vmax=max(0.01, float(np.nanmax(mat))),
        )
        ax4.set_yticks(np.arange(len(labels)))
        ax4.set_yticklabels(labels)
        ax4.set_xlabel("Simulation time (ns)")
        ax4.set_ylabel(r"MoO$_x$S$_y$ state")
        cb = fig.colorbar(im, ax=ax4, pad=0.02, fraction=0.045)
        cb.set_label("Population fraction")
        ax4.text(-0.14, 1.05, "d", transform=ax4.transAxes, fontweight="bold", fontsize=9)

    for ax in fig.axes:
        if hasattr(ax, "tick_params"):
            ax.tick_params(direction="in", top=True, right=True)
        if hasattr(ax, "grid"):
            ax.grid(False)

    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.98)
    for ext, kwargs in (
        ("png", {"dpi": 600}),
        ("pdf", {}),
        ("svg", {}),
    ):
        fig.savefig(
            f"{outbase}_mechanism.{ext}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Analyze surface Mo coordination, MoOxSy populations, ligand-exchange "
            "transitions, residence times, and bridge motifs from a LAMMPS trajectory."
        ),
    )
    p.add_argument("trajectory", type=Path, help="LAMMPS custom trajectory")
    p.add_argument(
        "--dt-fs",
        type=float,
        default=1.0,
        help="LAMMPS integration timestep in fs; physical time = TIMESTEP * dt-fs",
    )
    p.add_argument("--stride", type=int, default=1, help="Analyze every N-th dumped frame")
    p.add_argument("--start-frame", type=int, default=0, help="First dump frame index, 0-based")
    p.add_argument("--stop-frame", type=int, default=None, help="Stop before this dump frame")

    p.add_argument("--cut-mo-s", type=float, default=2.8, help="Mo-S cutoff (angstrom)")
    p.add_argument("--cut-mo-o", type=float, default=2.5, help="Mo-O cutoff (angstrom)")
    p.add_argument("--cut-mo-mo", type=float, default=3.8, help="Mo-Mo cutoff (angstrom)")

    p.add_argument(
        "--surface-zmin",
        type=float,
        default=-1.0,
        help="Minimum Mo height relative to top Al surface (angstrom)",
    )
    p.add_argument(
        "--surface-zmax",
        type=float,
        default=5.0,
        help="Maximum Mo height relative to top Al surface (angstrom)",
    )
    p.add_argument(
        "--al-top-layer-tol",
        type=float,
        default=0.15,
        help="Al atoms within this distance below max Al-z define top Al layer",
    )
    p.add_argument(
        "--pbc",
        choices=["xy", "xyz", "none", "auto"],
        default="xy",
        help="Periodic directions for distance calculations",
    )

    p.add_argument("--type-Al", dest="type_Al", type=int, default=1)
    p.add_argument("--type-Mo", dest="type_Mo", type=int, default=2)
    p.add_argument("--type-O", dest="type_O", type=int, default=3)
    p.add_argument("--type-S", dest="type_S", type=int, default=4)

    p.add_argument("--prefix", type=Path, default=Path("surface_Mo"))
    p.add_argument(
        "--absolute-time",
        action="store_true",
        help="Report absolute LAMMPS time instead of elapsed time from first analyzed frame",
    )
    p.add_argument("--no-plot", action="store_true", help="Skip matplotlib figures")
    args = p.parse_args()

    if args.stride < 1:
        p.error("--stride must be >= 1")
    if args.dt_fs <= 0:
        p.error("--dt-fs must be > 0")
    if min(args.cut_mo_s, args.cut_mo_o, args.cut_mo_mo) <= 0:
        p.error("All coordination cutoffs must be positive")
    if args.surface_zmax < args.surface_zmin:
        p.error("--surface-zmax must be >= --surface-zmin")
    if not args.trajectory.exists():
        p.error(f"Trajectory not found: {args.trajectory}")

    type_map = {
        args.type_Al: "Al",
        args.type_Mo: "Mo",
        args.type_O: "O",
        args.type_S: "S",
    }
    if len(type_map) != 4:
        p.error("Al/Mo/O/S atom type IDs must be distinct")

    prefix = args.prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)

    atom_csv = Path(str(prefix) + "_atoms.csv")
    summary_csv = Path(str(prefix) + "_time_series.csv")
    population_csv = Path(str(prefix) + "_species_population.csv")
    transition_csv = Path(str(prefix) + "_transitions.csv")
    transition_count_csv = Path(str(prefix) + "_transition_counts.csv")
    transition_prob_csv = Path(str(prefix) + "_transition_probabilities.csv")
    residence_csv = Path(str(prefix) + "_residence_times.csv")
    residence_summary_csv = Path(str(prefix) + "_residence_summary.csv")
    height_summary_csv = Path(str(prefix) + "_species_height_summary.csv")

    atom_fields = [
        "frame_index", "timestep", "time_ps", "time_ns",
        "Mo_id", "x_A", "y_A", "z_A", "Al_top_z_A", "height_above_Al_top_A",
        "CN_Mo_S", "CN_Mo_O", "CN_Mo_Mo", "species", "S_fraction",
        "nearest_S_A", "nearest_O_A", "nearest_Mo_A",
    ]
    summary_fields = [
        "frame_index", "timestep", "time_ps", "time_ns", "Al_top_z_A",
        "n_surface_Mo",
        "mean_CN_Mo_S", "std_CN_Mo_S", "median_CN_Mo_S",
        "mean_CN_Mo_O", "std_CN_Mo_O", "median_CN_Mo_O",
        "mean_CN_Mo_Mo", "std_CN_Mo_Mo", "median_CN_Mo_Mo",
        "sum_CN_Mo_S", "sum_CN_Mo_O", "sum_CN_Mo_Mo",
        "sulfurization_index", "oxygen_coord_fraction",
        "n_bridging_O", "n_bridging_S",
        "n_Mo_O_Mo_pairs", "n_Mo_S_Mo_pairs",
        "mean_surface_Mo_height_A", "std_surface_Mo_height_A",
        "n_state_changes", "n_O_to_S_events", "n_S_to_O_events",
    ]
    transition_fields = [
        "frame_index", "timestep", "time_ps", "time_ns", "Mo_id",
        "old_CN_O", "old_CN_S", "new_CN_O", "new_CN_S",
        "old_species", "new_species", "delta_O", "delta_S", "event_type",
    ]
    residence_fields = [
        "Mo_id", "species", "CN_O", "CN_S",
        "start_frame", "end_frame", "start_time_ns", "end_time_ns",
        "duration_ns", "n_observations", "start_reason", "end_reason",
        "right_censored",
    ]

    # State/population data small enough to keep frame-wise in memory.
    population_records: List[Dict[str, object]] = []
    all_states: set[State] = set()

    # Transition bookkeeping.
    transition_counts: Counter[Tuple[State, State]] = Counter()
    prev_states: Dict[int, State] = {}

    # Residence bookkeeping.
    active_runs: Dict[int, Dict[str, object]] = {}
    residence_stats: Dict[State, Dict[str, float]] = defaultdict(
        lambda: {"n": 0.0, "sum": 0.0, "sumsq": 0.0, "min": math.inf, "max": -math.inf}
    )

    # Height statistics by chemical state.
    height_stats: Dict[State, Dict[str, float]] = {}

    n_read = 0
    n_analyzed = 0
    first_analyzed_ts: Optional[int] = None
    last_time_ns = 0.0
    last_dump_index = -1
    last_timestep = 0

    def close_run(
        wr: csv.DictWriter,
        mo_id: int,
        run: Dict[str, object],
        end_frame: int,
        end_time_ns: float,
        end_reason: str,
        right_censored: int = 0,
    ) -> None:
        state = run["state"]  # type: ignore[assignment]
        assert isinstance(state, tuple)
        duration = max(0.0, end_time_ns - float(run["start_time_ns"]))
        wr.writerow(
            {
                "Mo_id": mo_id,
                "species": species_name(state),
                "CN_O": state[0],
                "CN_S": state[1],
                "start_frame": int(run["start_frame"]),
                "end_frame": end_frame,
                "start_time_ns": f"{float(run['start_time_ns']):.9g}",
                "end_time_ns": f"{end_time_ns:.9g}",
                "duration_ns": f"{duration:.9g}",
                "n_observations": int(run["n_obs"]),
                "start_reason": str(run["start_reason"]),
                "end_reason": end_reason,
                "right_censored": right_censored,
            }
        )
        st = residence_stats[state]
        st["n"] += 1.0
        st["sum"] += duration
        st["sumsq"] += duration * duration
        st["min"] = min(st["min"], duration)
        st["max"] = max(st["max"], duration)

    with atom_csv.open("w", newline="", encoding="utf-8") as fa, \
         summary_csv.open("w", newline="", encoding="utf-8") as fs, \
         transition_csv.open("w", newline="", encoding="utf-8") as ft, \
         residence_csv.open("w", newline="", encoding="utf-8") as fr:

        wa = csv.DictWriter(fa, fieldnames=atom_fields)
        ws = csv.DictWriter(fs, fieldnames=summary_fields)
        wt = csv.DictWriter(ft, fieldnames=transition_fields)
        wr = csv.DictWriter(fr, fieldnames=residence_fields)
        wa.writeheader()
        ws.writeheader()
        wt.writeheader()
        wr.writeheader()

        for dump_index, frame in enumerate(read_lammps_dump(args.trajectory)):
            n_read += 1
            if dump_index < args.start_frame:
                continue
            if args.stop_frame is not None and dump_index >= args.stop_frame:
                break
            if (dump_index - args.start_frame) % args.stride != 0:
                continue

            if first_analyzed_ts is None:
                first_analyzed_ts = frame.timestep

            ts_out = frame.timestep if args.absolute_time else frame.timestep - first_analyzed_ts
            time_ps = ts_out * args.dt_fs / 1000.0
            time_ns = time_ps / 1000.0
            last_time_ns = time_ns
            last_dump_index = dump_index
            last_timestep = frame.timestep

            pos = get_cartesian_positions(frame)
            elems = get_elements(frame, type_map)
            ids = get_ids(frame)

            al_mask = elems == "Al"
            mo_mask = elems == "Mo"
            o_mask = elems == "O"
            s_mask = elems == "S"

            if not np.any(al_mask):
                raise ValueError(
                    f"No Al atoms found at timestep {frame.timestep}; check type mapping."
                )
            if not np.any(mo_mask):
                raise ValueError(
                    f"No Mo atoms found at timestep {frame.timestep}; check type mapping."
                )

            zsurf = top_al_surface_z(pos[al_mask, 2], args.al_top_layer_tol)

            mo_all_pos = pos[mo_mask]
            mo_all_ids = ids[mo_mask]
            heights_all = mo_all_pos[:, 2] - zsurf
            surf_sel = (
                (heights_all >= args.surface_zmin)
                & (heights_all <= args.surface_zmax)
            )

            mo_pos = mo_all_pos[surf_sel]
            mo_ids = mo_all_ids[surf_sel]
            mo_h = heights_all[surf_sel]
            o_pos, o_ids = pos[o_mask], ids[o_mask]
            s_pos, s_ids = pos[s_mask], ids[s_mask]

            pbc_mask = pbc_mask_from_arg(frame.box, args.pbc)

            cn_s, near_s, s_neighbors, s_degree = coordination_details(
                mo_pos, mo_ids, s_pos, s_ids, args.cut_mo_s,
                frame.box, pbc_mask,
                return_neighbor_lists=True,
                return_neighbor_degree=True,
            )
            cn_o, near_o, o_neighbors, o_degree = coordination_details(
                mo_pos, mo_ids, o_pos, o_ids, args.cut_mo_o,
                frame.box, pbc_mask,
                return_neighbor_lists=True,
                return_neighbor_degree=True,
            )
            cn_mo, near_mo, _, _ = coordination_details(
                mo_pos, mo_ids, mo_pos, mo_ids, args.cut_mo_mo,
                frame.box, pbc_mask,
                exclude_same_id=True,
            )

            # Stable atom order for outputs and state tracking.
            order = np.argsort(mo_ids)
            mo_ids = mo_ids[order]
            mo_pos = mo_pos[order]
            mo_h = mo_h[order]
            cn_s, near_s = cn_s[order], near_s[order]
            cn_o, near_o = cn_o[order], near_o[order]
            cn_mo, near_mo = cn_mo[order], near_mo[order]

            states: List[State] = [(int(o), int(s)) for o, s in zip(cn_o, cn_s)]
            curr_states: Dict[int, State] = {
                int(mid): st for mid, st in zip(mo_ids, states)
            }
            all_states.update(states)

            # Per-atom output + species-resolved height accumulation.
            for j, (mid, st) in enumerate(zip(mo_ids, states)):
                denom = st[0] + st[1]
                s_fraction = st[1] / denom if denom > 0 else math.nan
                wa.writerow(
                    {
                        "frame_index": dump_index,
                        "timestep": frame.timestep,
                        "time_ps": f"{time_ps:.9g}",
                        "time_ns": f"{time_ns:.9g}",
                        "Mo_id": int(mid),
                        "x_A": f"{mo_pos[j, 0]:.8f}",
                        "y_A": f"{mo_pos[j, 1]:.8f}",
                        "z_A": f"{mo_pos[j, 2]:.8f}",
                        "Al_top_z_A": f"{zsurf:.8f}",
                        "height_above_Al_top_A": f"{mo_h[j]:.8f}",
                        "CN_Mo_S": st[1],
                        "CN_Mo_O": st[0],
                        "CN_Mo_Mo": int(cn_mo[j]),
                        "species": species_name(st),
                        "S_fraction": "" if not np.isfinite(s_fraction) else f"{s_fraction:.8g}",
                        "nearest_S_A": "" if np.isnan(near_s[j]) else f"{near_s[j]:.8f}",
                        "nearest_O_A": "" if np.isnan(near_o[j]) else f"{near_o[j]:.8f}",
                        "nearest_Mo_A": "" if np.isnan(near_mo[j]) else f"{near_mo[j]:.8f}",
                    }
                )

            # Group heights by state without keeping all coordinates in RAM.
            if len(states):
                state_arr = np.array(states, dtype=int)
                for st in set(states):
                    mask_st = (state_arr[:, 0] == st[0]) & (state_arr[:, 1] == st[1])
                    update_online_stats(height_stats, st, mo_h[mask_st])

            # MoOxSy population for this frame.
            pop = Counter(states)
            population_records.append(
                {
                    "frame_index": dump_index,
                    "timestep": frame.timestep,
                    "time_ps": time_ps,
                    "time_ns": time_ns,
                    "n_surface_Mo": len(mo_ids),
                    "population": pop,
                }
            )

            # Bridges: degree = number of selected surface Mo atoms bound to each ligand.
            assert o_degree is not None and s_degree is not None
            n_bridging_o = int(np.sum(o_degree >= 2))
            n_bridging_s = int(np.sum(s_degree >= 2))
            n_mo_o_mo_pairs = int(np.sum(o_degree * (o_degree - 1) // 2))
            n_mo_s_mo_pairs = int(np.sum(s_degree * (s_degree - 1) // 2))

            # Transition statistics between consecutive ANALYZED frames.
            n_state_changes = 0
            n_o_to_s = 0
            n_s_to_o = 0
            if n_analyzed > 0:
                common = set(prev_states).intersection(curr_states)
                for mid in sorted(common):
                    old = prev_states[mid]
                    new = curr_states[mid]
                    transition_counts[(old, new)] += 1
                    if old != new:
                        n_state_changes += 1
                        etype = classify_transition(old, new)
                        if etype == "O_to_S_exchange":
                            n_o_to_s += 1
                        elif etype == "S_to_O_exchange":
                            n_s_to_o += 1
                        wt.writerow(
                            {
                                "frame_index": dump_index,
                                "timestep": frame.timestep,
                                "time_ps": f"{time_ps:.9g}",
                                "time_ns": f"{time_ns:.9g}",
                                "Mo_id": mid,
                                "old_CN_O": old[0],
                                "old_CN_S": old[1],
                                "new_CN_O": new[0],
                                "new_CN_S": new[1],
                                "old_species": species_name(old),
                                "new_species": species_name(new),
                                "delta_O": new[0] - old[0],
                                "delta_S": new[1] - old[1],
                                "event_type": etype,
                            }
                        )

            # Residence intervals.
            curr_id_set = set(curr_states)
            active_id_set = set(active_runs)

            # Mo left surface-selection window.
            for mid in sorted(active_id_set - curr_id_set):
                close_run(
                    wr, mid, active_runs[mid], dump_index, time_ns,
                    end_reason="left_surface", right_censored=0,
                )
                del active_runs[mid]

            for mid in sorted(curr_id_set):
                st = curr_states[mid]
                if mid not in active_runs:
                    start_reason = "initial" if n_analyzed == 0 else "entered_surface"
                    active_runs[mid] = {
                        "state": st,
                        "start_frame": dump_index,
                        "start_time_ns": time_ns,
                        "n_obs": 1,
                        "start_reason": start_reason,
                    }
                else:
                    run = active_runs[mid]
                    old_state = run["state"]
                    if old_state == st:
                        run["n_obs"] = int(run["n_obs"]) + 1
                    else:
                        close_run(
                            wr, mid, run, dump_index, time_ns,
                            end_reason="transition", right_censored=0,
                        )
                        active_runs[mid] = {
                            "state": st,
                            "start_frame": dump_index,
                            "start_time_ns": time_ns,
                            "n_obs": 1,
                            "start_reason": "transition",
                        }

            # Global sulfurization coordinate.
            total_s = int(np.sum(cn_s))
            total_o = int(np.sum(cn_o))
            coord_total = total_s + total_o
            chi_s = total_s / coord_total if coord_total > 0 else math.nan
            chi_o = total_o / coord_total if coord_total > 0 else math.nan

            ws.writerow(
                {
                    "frame_index": dump_index,
                    "timestep": frame.timestep,
                    "time_ps": f"{time_ps:.9g}",
                    "time_ns": f"{time_ns:.9g}",
                    "Al_top_z_A": f"{zsurf:.8f}",
                    "n_surface_Mo": int(len(mo_ids)),
                    "mean_CN_Mo_S": f"{safe_mean(cn_s):.8g}",
                    "std_CN_Mo_S": f"{safe_std(cn_s):.8g}",
                    "median_CN_Mo_S": f"{safe_median(cn_s):.8g}",
                    "mean_CN_Mo_O": f"{safe_mean(cn_o):.8g}",
                    "std_CN_Mo_O": f"{safe_std(cn_o):.8g}",
                    "median_CN_Mo_O": f"{safe_median(cn_o):.8g}",
                    "mean_CN_Mo_Mo": f"{safe_mean(cn_mo):.8g}",
                    "std_CN_Mo_Mo": f"{safe_std(cn_mo):.8g}",
                    "median_CN_Mo_Mo": f"{safe_median(cn_mo):.8g}",
                    "sum_CN_Mo_S": total_s,
                    "sum_CN_Mo_O": total_o,
                    "sum_CN_Mo_Mo": int(np.sum(cn_mo)),
                    "sulfurization_index": "" if not np.isfinite(chi_s) else f"{chi_s:.8g}",
                    "oxygen_coord_fraction": "" if not np.isfinite(chi_o) else f"{chi_o:.8g}",
                    "n_bridging_O": n_bridging_o,
                    "n_bridging_S": n_bridging_s,
                    "n_Mo_O_Mo_pairs": n_mo_o_mo_pairs,
                    "n_Mo_S_Mo_pairs": n_mo_s_mo_pairs,
                    "mean_surface_Mo_height_A": f"{safe_mean(mo_h):.8g}",
                    "std_surface_Mo_height_A": f"{safe_std(mo_h):.8g}",
                    "n_state_changes": n_state_changes,
                    "n_O_to_S_events": n_o_to_s,
                    "n_S_to_O_events": n_s_to_o,
                }
            )

            prev_states = curr_states
            n_analyzed += 1

            if n_analyzed == 1 or n_analyzed % 100 == 0:
                print(
                    f"Analyzed {n_analyzed} frame(s): dump_index={dump_index}, "
                    f"timestep={frame.timestep}, surface Mo={len(mo_ids)}, "
                    f"<CN Mo-S>={safe_mean(cn_s):.3f}, "
                    f"<CN Mo-O>={safe_mean(cn_o):.3f}, chi_S={chi_s:.3f}"
                    if np.isfinite(chi_s)
                    else f"Analyzed {n_analyzed} frame(s): dump_index={dump_index}"
                )

        # Close right-censored runs at trajectory end.
        if n_analyzed > 0:
            for mid in sorted(active_runs):
                close_run(
                    wr,
                    mid,
                    active_runs[mid],
                    last_dump_index,
                    last_time_ns,
                    end_reason="trajectory_end",
                    right_censored=1,
                )

    # ------------------------------------------------------------------
    # Post-pass compact outputs
    # ------------------------------------------------------------------
    states_sorted = sorted(all_states, key=state_sort_key)

    # Species populations, both counts and fractions.
    pop_fields = ["frame_index", "timestep", "time_ps", "time_ns", "n_surface_Mo"]
    pop_fields += [f"N_{species_name(st)}" for st in states_sorted]
    pop_fields += [f"f_{species_name(st)}" for st in states_sorted]
    with population_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pop_fields)
        w.writeheader()
        for rec in population_records:
            pop: Counter[State] = rec["population"]  # type: ignore[assignment]
            nmo = int(rec["n_surface_Mo"])
            row: Dict[str, object] = {
                "frame_index": rec["frame_index"],
                "timestep": rec["timestep"],
                "time_ps": f"{float(rec['time_ps']):.9g}",
                "time_ns": f"{float(rec['time_ns']):.9g}",
                "n_surface_Mo": nmo,
            }
            for st in states_sorted:
                n = int(pop.get(st, 0))
                row[f"N_{species_name(st)}"] = n
                row[f"f_{species_name(st)}"] = f"{(n / nmo if nmo else 0.0):.8g}"
            w.writerow(row)

    # Transition count and probability matrices. Include diagonal persistence.
    matrix_states = sorted(
        set(states_sorted)
        | {a for a, _ in transition_counts}
        | {b for _, b in transition_counts},
        key=state_sort_key,
    )
    names = [species_name(st) for st in matrix_states]
    row_totals = {
        a: sum(transition_counts[(a, b)] for b in matrix_states)
        for a in matrix_states
    }

    with transition_count_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["from\\to"] + names)
        for a in matrix_states:
            w.writerow([species_name(a)] + [transition_counts[(a, b)] for b in matrix_states])

    with transition_prob_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["from\\to"] + names)
        for a in matrix_states:
            total = row_totals[a]
            vals = [transition_counts[(a, b)] / total if total else 0.0 for b in matrix_states]
            w.writerow([species_name(a)] + [f"{x:.8g}" for x in vals])

    # Residence summary by state.
    with residence_summary_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "species", "CN_O", "CN_S", "n_intervals",
            "mean_duration_ns", "std_duration_ns", "min_duration_ns", "max_duration_ns",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for st in sorted(residence_stats, key=state_sort_key):
            rec = residence_stats[st]
            n = int(rec["n"])
            mean = rec["sum"] / n if n else math.nan
            var = max(0.0, rec["sumsq"] / n - mean * mean) if n else math.nan
            w.writerow(
                {
                    "species": species_name(st),
                    "CN_O": st[0],
                    "CN_S": st[1],
                    "n_intervals": n,
                    "mean_duration_ns": f"{mean:.9g}" if n else "",
                    "std_duration_ns": f"{math.sqrt(var):.9g}" if n else "",
                    "min_duration_ns": f"{rec['min']:.9g}" if n else "",
                    "max_duration_ns": f"{rec['max']:.9g}" if n else "",
                }
            )

    # Height summary by local state.
    with height_summary_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "species", "CN_O", "CN_S", "n_samples",
            "mean_height_A", "std_height_A", "min_height_A", "max_height_A",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for st in sorted(height_stats, key=state_sort_key):
            rec = height_stats[st]
            n = int(rec["n"])
            mean = rec["sum"] / n if n else math.nan
            var = max(0.0, rec["sumsq"] / n - mean * mean) if n else math.nan
            w.writerow(
                {
                    "species": species_name(st),
                    "CN_O": st[0],
                    "CN_S": st[1],
                    "n_samples": n,
                    "mean_height_A": f"{mean:.9g}" if n else "",
                    "std_height_A": f"{math.sqrt(var):.9g}" if n else "",
                    "min_height_A": f"{rec['min']:.9g}" if n else "",
                    "max_height_A": f"{rec['max']:.9g}" if n else "",
                }
            )

    print("\nDone.")
    print(f"Dump frames read              : {n_read}")
    print(f"Frames analyzed               : {n_analyzed}")
    print(f"Atom-resolved                 : {atom_csv}")
    print(f"Frame time series             : {summary_csv}")
    print(f"MoOxSy populations            : {population_csv}")
    print(f"State-change events           : {transition_csv}")
    print(f"Transition counts             : {transition_count_csv}")
    print(f"Transition probabilities      : {transition_prob_csv}")
    print(f"Residence intervals           : {residence_csv}")
    print(f"Residence summary             : {residence_summary_csv}")
    print(f"Species-height summary        : {height_summary_csv}")

    if n_analyzed > 0 and not args.no_plot:
        make_evolution_plot(summary_csv, atom_csv, prefix)
        make_mechanism_plot(summary_csv, population_csv, prefix)
        print(f"Plots                         : {prefix}_evolution.[png|pdf|svg]")
        print(f"                                {prefix}_mechanism.[png|pdf|svg]")


if __name__ == "__main__":
    main()
