# ChemAL: Chemistry-Aware Active Learning Sampler

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#requirements)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20optional-ee4c2c.svg)](#requirements)
[![ASE](https://img.shields.io/badge/ASE-trajectory%20IO-green.svg)](#requirements)
[![Status](https://img.shields.io/badge/status-research%20code-orange.svg)](#)

**ChemAL** is a chemistry-aware active-learning frame selector for atomistic trajectories. It is designed for computational chemistry, catalysis, surface reactions, molecular dynamics, and reactive sampling workflows where the goal is to extract a small set of chemically informative and diverse structures from a long trajectory.

The sampler combines:

- chemically weighted neighbor graphs;
- covalent-radius-based soft bond gates;
- optional surface/reactive-region masking;
- e3nn spherical-harmonic geometric descriptors;
- topology-aware message passing;
- entropy-deficit anomaly scoring;
- Wasserstein-like distribution-shift scoring;
- bond-network-change scoring;
- temporal non-maximum suppression, NMS;
- farthest-point sampling, FPS, in frame-embedding space.

The output structures can be used as candidates for **DFT labeling**, **machine-learning potential retraining**, **active learning**, or **dataset enrichment**.

---

## Table of Contents

- [Motivation](#motivation)
- [Method Overview](#method-overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Programmatic Usage](#programmatic-usage)
- [Input Format](#input-format)
- [Output Files](#output-files)
- [Key Parameters](#key-parameters)
- [Recommended Settings](#recommended-settings)
- [Workflow Example](#workflow-example)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)

---

## Motivation

Conventional active-learning selection often relies on model uncertainty, energy variance, or geometric RMSD. These criteria can miss chemically important events such as:

- bond formation and bond breaking;
- transient intermediates;
- local coordination changes;
- surface reconstruction;
- adsorption/desorption;
- rare reactive environments;
- transition-state-like configurations.

**ChemAL** addresses this by constructing a chemically meaningful local environment representation and selecting frames that are both **novel** and **diverse**.

---

## Method Overview

ChemAL performs a two-stage selection procedure.

### Stage 1: Chemistry-aware novelty pre-screening

For each frame, ChemAL builds a neighbor graph from atomic positions using ASE. Edge weights combine a smooth distance cutoff and a soft chemistry gate based on covalent radii:

```text
edge_weight = cosine_cutoff(r / rc) × soft_bond_gate(r, r_cov_i, r_cov_j)
```

The active region is defined using the following priority:

1. non-zero ASE atom tags, if present;
2. atoms above `z_surface_threshold`, if provided;
3. optional filtering by `reactive_species`.

For each atom, ChemAL constructs geometry-aware descriptors using first- and second-order e3nn spherical harmonics. These descriptors are propagated through the local topology by message passing.

The novelty score combines four terms:

```text
S = w_local       × local_anomaly
  + w_wasserstein × distribution_shift
  + w_deviation   × fingerprint_deviation
  + w_bond        × bond_network_change
```

Temporal non-maximum suppression is then applied to avoid repeatedly selecting nearly identical neighboring frames from the same local event.

### Stage 2: Diversity selection with FPS

The highest-scoring candidate pool is embedded in a frame-level chemical fingerprint space. ChemAL then applies farthest-point sampling, FPS, to select a final set of chemically diverse frames.

---

## Repository Structure

```text
.
├── ChemAL.py        # Main sampler implementation
├── README.md        # Project documentation
└── surface.extxyz   # Example input trajectory, not included by default
```

The current implementation is a single-file research script. It can be run directly or imported into another workflow.

---

## Requirements

ChemAL requires Python and the following scientific Python packages:

```text
python >= 3.9
numpy
matplotlib
torch
ase
e3nn
```

CUDA is optional. If a CUDA-capable GPU is available, the script automatically uses it through PyTorch.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

Create a clean environment:

```bash
conda create -n chemal python=3.10 -y
conda activate chemal
```

Install dependencies:

```bash
pip install numpy matplotlib ase e3nn torch
```

For GPU-enabled PyTorch, install the appropriate wheel from the official PyTorch installation selector for your CUDA version.

---

## Quick Start

Place an ASE-readable trajectory named `surface.extxyz` in the repository root:

```text
surface.extxyz
```

Run:

```bash
python ChemAL.py
```

By default, the script looks for:

```python
traj_path = "surface.extxyz"
```

and writes selected active-learning candidates to:

```text
active_learning_candidates.xyz
```

If `surface.extxyz` is not found, the script exits with:

```text
Trajectory file not found: surface.extxyz
```

---

## Programmatic Usage

For most production workflows, it is better to import the selection function and call it explicitly:

```python
from ChemAL import extract_novel_frames_from_trajectory

extract_novel_frames_from_trajectory(
    traj_file="surface.extxyz",
    out_file="active_learning_candidates.xyz",
    file_format="extxyz",
    top_k=20,
    candidate_multiplier=5,
    rc=3.5,
    z_surface_threshold=10.0,
    reactive_species=["B", "N", "Ni"],
    nms_tau=50,
    w_local=1.0,
    w_wasserstein=0.2,
    w_deviation=10.0,
    w_bond=3.0,
    mp_steps=3,
    mp_weight=0.5,
    bond_scale=1.25,
    bond_sharpness=0.10,
)
```

---

## Input Format

ChemAL reads trajectories through `ase.io.iread`, so any ASE-supported trajectory format can be used if the corresponding `file_format` is set correctly.

Typical formats include:

```text
extxyz
xyz
traj
vasp
lammps-dump-text
```

For the default script configuration, use:

```python
file_format="extxyz"
```

### Important input constraint

The trajectory must have a fixed number of atoms in every frame. ChemAL checks that each frame has the same atom count as the reference frame.

This is suitable for:

- NVT molecular dynamics;
- NPT molecular dynamics with fixed composition;
- surface reaction trajectories without atom insertion/deletion;
- AIMD or MLMD trajectories with consistent atom indexing.

It is not suitable, without modification, for trajectories where atoms are inserted, deleted, or reordered between frames.

---

## Output Files

Given:

```python
out_file="active_learning_candidates.xyz"
```

ChemAL generates the following files:

| File | Description |
|---|---|
| `active_learning_candidates.xyz` | Selected chemically novel and diverse structures, written in `extxyz` format. |
| `active_learning_candidates_selection_meta.json` | Metadata for the final selected frames, including rank, frame index, score components, raw score, and NMS score. |
| `active_learning_candidates_score_history.json` | Full trajectory scoring history for diagnostics and post-analysis. |
| `active_learning_candidates_scoring_timeline.png` | Diagnostic plot showing raw novelty, NMS-suppressed novelty, component scores, and selected frames. |

---

## Key Parameters

### Selection size

| Parameter | Meaning | Typical value |
|---|---|---|
| `top_k` | Number of final frames selected for active learning. | `5–100` |
| `candidate_multiplier` | Candidate-pool multiplier before FPS. Pool size is `top_k × candidate_multiplier`. | `3–10` |

### Local chemical graph

| Parameter | Meaning | Typical value |
|---|---|---|
| `rc` | Neighbor cutoff radius in Å. | `3.0–4.5` for surfaces, `2.0–3.5` for molecules |
| `bond_scale` | Covalent-radius scaling factor for the soft bond gate. | `1.10–1.35` |
| `bond_sharpness` | Smoothness of the soft bond transition. | `0.05–0.20` |

### Reactive-region control

| Parameter | Meaning | Example |
|---|---|---|
| `z_surface_threshold` | Emphasizes atoms above a z-coordinate threshold. Useful for slab/surface systems. | `10.0` |
| `reactive_species` | Restricts active-learning focus to selected elements. | `["B", "N", "Ni"]` |

The active mask follows this logic:

```text
if non-zero ASE tags exist:
    active atoms = tagged atoms
elif z_surface_threshold is not None:
    active atoms = atoms with z >= z_surface_threshold
else:
    active atoms = all atoms

if reactive_species is not None:
    active atoms = active atoms ∩ selected species
```

### Message passing

| Parameter | Meaning | Typical value |
|---|---|---|
| `mp_steps` | Number of topology-aware message-passing steps. | `1–2` for molecules, `2–3` for surfaces |
| `mp_weight` | Mixing ratio between local descriptors and neighbor-aggregated descriptors. | `0.3–0.6` |

### Novelty-score weights

| Parameter | Meaning | Typical value |
|---|---|---|
| `w_local` | Emphasizes local symmetry breaking and rare local environments. | `0.5–2.0` |
| `w_wasserstein` | Emphasizes distribution-level evolution of the active region. | `0.1–2.0` |
| `w_deviation` | Emphasizes deviation from the reference chemical fingerprint. | `3–10` |
| `w_bond` | Emphasizes coordination and bond-network change. | `1–5` |

### Temporal diversity

| Parameter | Meaning | Typical value |
|---|---|---|
| `nms_tau` | Temporal non-maximum-suppression timescale. Larger values enforce stronger temporal separation. | `10–30` short trajectories, `30–100` medium trajectories, `100–300` long trajectories |

---

## Recommended Settings

### Surface catalysis / reactive slab systems

```python
extract_novel_frames_from_trajectory(
    traj_file="surface.extxyz",
    out_file="al_candidates.xyz",
    rc=3.5,
    z_surface_threshold=10.0,
    reactive_species=["C", "O", "Pt"],
    top_k=20,
    candidate_multiplier=5,
    nms_tau=50,
    w_local=1.0,
    w_wasserstein=0.2,
    w_deviation=8.0,
    w_bond=3.0,
    mp_steps=3,
    mp_weight=0.5,
)
```

### Molecular reaction trajectories

```python
extract_novel_frames_from_trajectory(
    traj_file="reaction.extxyz",
    out_file="al_candidates.xyz",
    rc=2.8,
    z_surface_threshold=None,
    reactive_species=None,
    top_k=10,
    candidate_multiplier=5,
    nms_tau=30,
    w_local=1.5,
    w_wasserstein=0.5,
    w_deviation=6.0,
    w_bond=4.0,
    mp_steps=2,
    mp_weight=0.4,
)
```

### Adsorbate-focused active learning

```python
extract_novel_frames_from_trajectory(
    traj_file="adsorption.extxyz",
    out_file="adsorbate_al_candidates.xyz",
    rc=3.2,
    z_surface_threshold=None,
    reactive_species=["H", "C", "O"],
    top_k=15,
    candidate_multiplier=8,
    nms_tau=80,
    w_local=1.0,
    w_wasserstein=0.3,
    w_deviation=10.0,
    w_bond=5.0,
    mp_steps=2,
    mp_weight=0.5,
)
```

---

## Workflow Example

A typical active-learning loop is:

1. Run molecular dynamics or AIMD to generate a trajectory.
2. Use ChemAL to select chemically novel and diverse frames.
3. Label selected frames with DFT or another high-level electronic-structure method.
4. Add labeled structures to the training set.
5. Retrain the machine-learning potential.
6. Repeat until the target chemical events are well sampled.

Example:

```bash
python ChemAL.py
```

Then inspect:

```bash
active_learning_candidates_selection_meta.json
active_learning_candidates_score_history.json
active_learning_candidates_scoring_timeline.png
```

The selected structures can be passed to DFT workflows such as VASP, Quantum ESPRESSO, CP2K, ORCA, Gaussian, or other electronic-structure codes after conversion with ASE.

---

## Troubleshooting

### `Trajectory file not found: surface.extxyz`

The default script expects `surface.extxyz` in the current directory. Either rename your trajectory or call `extract_novel_frames_from_trajectory()` directly with your file path.

### `Frame X has N atoms, but reference has M atoms`

Your trajectory changes atom count between frames. ChemAL currently assumes fixed composition and fixed atom indexing.

### CUDA memory issues

Try reducing:

```python
top_k
candidate_multiplier
rc
mp_steps
```

You can also force CPU execution by running PyTorch without CUDA or editing the device selection logic.

### Selected frames are too temporally clustered

Increase:

```python
nms_tau
candidate_multiplier
```

### Selected frames are chemically uninteresting

Consider increasing:

```python
w_deviation
w_bond
```

and define a better active region using:

```python
z_surface_threshold
reactive_species
ASE atom tags
```

### Too much bulk/slab noise

Set `z_surface_threshold` to focus only on the surface or reactive region. Deep-slab edges are automatically attenuated when this threshold is used.

---

## Limitations

- The current implementation assumes fixed atom count and consistent atom ordering.
- The method is descriptor-based and does not directly use model uncertainty from an ensemble potential.
- Hyperparameters should be tuned for the chemical system, trajectory length, temperature, and target reaction mechanism.
- The script is currently designed as research code rather than a packaged command-line application.

---

## Citation

If you use this code in academic work, please cite the repository and describe the method as a chemistry-aware active-learning sampler based on equivariant local geometric fingerprints, bond-network novelty scoring, temporal non-maximum suppression, and FPS diversity selection.

Suggested citation format:

```bibtex
@software{chemal_sampler,
  title  = {ChemAL: Chemistry-Aware Active Learning Sampler},
  author = {Your Name},
  year   = {2026},
  url    = {https://github.com/<your-username>/<your-repository>}
}
```

---

## License

No license file is included by default. Before making the repository public, add a `LICENSE` file, for example MIT, BSD-3-Clause, Apache-2.0, or another license appropriate for your project and institution.
