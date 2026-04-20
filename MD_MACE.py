import time
import numpy as np
from ase import units
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.andersen import Andersen
from ase.md.verlet import VelocityVerlet
from ase.md.nose_hoover_chain import NoseHooverChainNVT  # <-- Correct imported module
from ase.constraints import FixAtoms
from mace.calculators import MACECalculator
from mace.calculators import mace_mp

# ============================================================
#  Simulation Parameters
# ============================================================
TOTAL_STEPS     = 1000000
OUTPUT_INTERVAL = 50
LOG_INTERVAL    = 50
TEMPERATURE_K   = 1600
TIMESTEP_FS     = 1.0    # fs

# --- Thermostat Settings ---
THERMOSTAT      = "NVE"  # Options: "Langevin", "NoseHooverChain", "Berendsen", "Andersen", "NVE"

# Thermostat-specific parameters
FRICTION          = 0.01      # [Langevin] Friction coefficient (1/fs)
NHC_TDAMP         = 100.0     # [NoseHooverChain] Characteristic time scale for thermostat (fs). Rec: 100 * timestep
NHC_TCHAIN        = 3         # [NoseHooverChain] Number of thermostat variables in the chain
BERENDSEN_TAUT    = 100.0     # [Berendsen] Relaxation time constant (fs)
ANDERSEN_PROB     = 1e-3      # [Andersen] Collision probability

# ============================================================
#  Fixed Layers Settings (for surface slabs)
# ============================================================
FIX_N_BOTTOM_LAYERS = 0       # Number of bottom layers to fix (counting from bottom to top)
LAYER_TOL           = 0.2     # Tolerance for z-coordinate to group atoms into the same layer (Å)

# ============================================================

start_time = time.time()

# ── Load Model (Mod 1: float32 for faster computation) ──────
calc = mace_mp(
    model="mh-1",
    default_dtype="float32",  
    device="cuda",
    head="omat_pbe",
)

# ── Read Structure ───────────────────────────────────────────
atoms = read("Al2O3.cif")
atoms.calc = calc

# ============================================================
#  Layer Detection: Group atoms by Z-coordinate
# ============================================================
def detect_layers(atoms, tol=0.5):
    """
    Groups atoms into layers based on their z-coordinates.
    Returns a list of lists, where each sublist contains the atomic indices of a layer, ordered from bottom to top.
    """
    z_positions = atoms.get_positions()[:, 2]
    sorted_idx  = np.argsort(z_positions)
    sorted_z    = z_positions[sorted_idx]

    layers = []
    current_layer = [sorted_idx[0]]

    for i in range(1, len(sorted_idx)):
        if sorted_z[i] - sorted_z[i - 1] <= tol:
            current_layer.append(sorted_idx[i])
        else:
            layers.append(current_layer)
            current_layer = [sorted_idx[i]]
    layers.append(current_layer)

    return layers

def get_fixed_indices_by_layer(atoms, n_bottom_layers, tol=0.5):
    layers   = detect_layers(atoms, tol=tol)
    n_layers = len(layers)

    print(f"\n{'─'*55}")
    print(f"  Detected total layers: {n_layers}")
    for i, layer in enumerate(layers):
        z_vals  = atoms.get_positions()[layer, 2]
        syms    = [atoms.get_chemical_symbols()[j] for j in layer]
        sym_str = " ".join(sorted(set(syms)))
        label   = "  ← FIXED" if i < n_bottom_layers else ""
        print(f"  Layer {i+1:2d}: z={z_vals.mean():.3f} Å  "
              f"({len(layer):3d} atoms, {sym_str}){label}")
    print(f"{'─'*55}\n")

    if n_bottom_layers > n_layers:
        raise ValueError(
            f"Requested to fix {n_bottom_layers} layers, but the structure only has {n_layers} layers!"
        )

    fixed = []
    for i in range(n_bottom_layers):
        fixed.extend(layers[i])
    return fixed, n_layers


# ── Detect Layers & Determine Fixed Atoms ────────────────────
fixed_indices, total_layers = get_fixed_indices_by_layer(
    atoms, FIX_N_BOTTOM_LAYERS, tol=LAYER_TOL
)
free_indices = [i for i in range(len(atoms)) if i not in fixed_indices]

# ── Apply Constraints ────────────────────────────────────────
if fixed_indices:
    atoms.set_constraint(FixAtoms(indices=fixed_indices))

print(f"Total atoms: {len(atoms)}")
print(f"Fixed atoms: {len(fixed_indices)} (Bottom {FIX_N_BOTTOM_LAYERS}/{total_layers} layers)")
print(f"Free atoms:  {len(free_indices)}")

# ── Initialize Velocities (Mod 3: Strict scaling) ────────────
MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K)
velocities = atoms.get_velocities()

if fixed_indices:
    # Zero out velocities of fixed atoms
    for i in fixed_indices:
        velocities[i] = [0.0, 0.0, 0.0]
    atoms.set_velocities(velocities)

# Rescale velocities of free atoms to match the exact target temperature
if free_indices:
    ke = atoms.get_kinetic_energy()
    # Current temperature calculated only from the degrees of freedom of free atoms
    current_temp = ke / (1.5 * units.kB * len(free_indices))
    if current_temp > 0:
        scale_factor = (TEMPERATURE_K / current_temp) ** 0.5
        for i in free_indices:
            velocities[i] *= scale_factor
        atoms.set_velocities(velocities)

# ── Initialize MD Engine ─────────────────────────────────────
print(f"\nInitializing MD engine...")
print(f"  Selected Thermostat: {THERMOSTAT}")

if THERMOSTAT == "Langevin":
    dyn = Langevin(
        atoms,
        timestep=TIMESTEP_FS * units.fs,
        temperature_K=TEMPERATURE_K,
        friction=FRICTION / units.fs,
    )
elif THERMOSTAT == "NoseHooverChain":
    # Using the strict Nose-Hoover Chain NVT ensemble
    dyn = NoseHooverChainNVT(
        atoms,
        timestep=TIMESTEP_FS * units.fs,
        temperature_K=TEMPERATURE_K,
        tdamp=NHC_TDAMP * units.fs,
        tchain=NHC_TCHAIN
    )
elif THERMOSTAT == "Berendsen":
    dyn = NVTBerendsen(
        atoms,
        timestep=TIMESTEP_FS * units.fs,
        temperature_K=TEMPERATURE_K,
        taut=BERENDSEN_TAUT * units.fs,
    )
elif THERMOSTAT == "Andersen":
    dyn = Andersen(
        atoms,
        timestep=TIMESTEP_FS * units.fs,
        temperature_K=TEMPERATURE_K,
        andersen_prob=ANDERSEN_PROB,
    )
elif THERMOSTAT == "NVE":
    # Microcanonical ensemble (no thermostat), only integrating equations of motion
    dyn = VelocityVerlet(
        atoms,
        timestep=TIMESTEP_FS * units.fs,
    )
else:
    raise ValueError(f"Unsupported thermostat type: {THERMOSTAT}")


# ============================================================
#  Trajectory & Logging
# ============================================================
def write_trajectory():
    write("traj.extxyz", atoms, append=True)

def write_log():
    elapsed = time.time() - start_time
    ke      = atoms.get_kinetic_energy()
    pe      = atoms.get_potential_energy()
    # Temperature calculation based only on free atoms
    temp    = ke / (1.5 * units.kB * len(free_indices)) if free_indices else 0.0
    with open("md.log", "a") as f:
        f.write(f"Step: {dyn.nsteps:6d}, "
                f"Time: {dyn.get_time()/units.fs:8.1f} fs, "
                f"T: {temp:6.1f} K, "
                f"Epot: {pe:10.4f} eV, "
                f"Ekin: {ke:10.4f} eV, "
                f"Etot: {pe+ke:10.4f} eV, "
                f"Runtime: {elapsed:8.1f} s\n")

# Initialize log file
with open("md.log", "w") as f:
    f.write("# Molecular Dynamics Simulation Log\n")
    f.write(f"# Thermostat: {THERMOSTAT} at {TEMPERATURE_K} K\n")
    f.write(f"# Timestep: {TIMESTEP_FS} fs\n")
    f.write(f"# Slab: {total_layers} layers total | "
            f"Fixed: bottom {FIX_N_BOTTOM_LAYERS} layers "
            f"({len(fixed_indices)} atoms) | "
            f"Free: {len(free_indices)} atoms\n")
    f.write("# Step      Time(fs)    T(K)     Epot(eV)      Ekin(eV)      Etot(eV)   Runtime(s)\n")

dyn.attach(write_trajectory, interval=OUTPUT_INTERVAL)
dyn.attach(write_log,        interval=LOG_INTERVAL)

# ============================================================
#  Run Simulation
# ============================================================
print(f"\nStarting MD simulation...")
print(f"  Temperature: {TEMPERATURE_K} K")
print(f"  Timestep:    {TIMESTEP_FS} fs")
print(f"  Total steps: {TOTAL_STEPS}  ({TOTAL_STEPS * TIMESTEP_FS / 1000:.1f} ps)")
print(f"  Output:      every {OUTPUT_INTERVAL} steps")
print(f"  Log:         every {LOG_INTERVAL} steps")
print("-" * 60)

dyn.run(steps=TOTAL_STEPS)

end_time      = time.time()
total_runtime = end_time - start_time

print("-" * 60)
print("MD simulation completed!")
print(f"  Total runtime:         {total_runtime:.2f} s")
print(f"  Average time per step: {total_runtime/TOTAL_STEPS:.4f} s")
print(f"  Simulation rate:       {TOTAL_STEPS/total_runtime:.1f} steps/s")
print(f"  Trajectory:            traj.extxyz ({TOTAL_STEPS//OUTPUT_INTERVAL} frames)")
print(f"  Log file:              md.log")

with open("md.log", "a") as f:
    f.write(f"# Simulation completed\n")
    f.write(f"# Total steps: {TOTAL_STEPS}\n")
    f.write(f"# Total runtime: {total_runtime:.2f} s\n")
    f.write(f"# Average time per step: {total_runtime/TOTAL_STEPS:.4f} s\n")
    f.write(f"# Simulation rate: {TOTAL_STEPS/total_runtime:.1f} steps/s\n")
