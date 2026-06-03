import time
from ase.io import read, write

# Import the NEP calculator
try:
    from pynep.calculate import NEP
except ImportError:
    print("Please install pynep first: pip install pynep")
    # If you want to use GPU, you can change this to: 
    # from calorine.calculators import GPUNEP as NEP
    exit()

# 1. Load the NEP model
model_path = "nep.txt"
print(f"Loading NEP model: {model_path} ...")
calc = NEP(model_path)

# 2. Read the extxyz trajectory to be calculated (using the previous mace.extxyz as an example)
input_file = "dft.extxyz"
output_file = "nep_recalculated.extxyz"

print(f"Reading trajectory file: {input_file} ...")
frames = read(input_file, index=":")
total_frames = len(frames)
print(f"Detected {total_frames} frames in total.")

# 3. Loop through to calculate the energy and forces for each frame
print("Starting to calculate Energy & Forces ...")
start_time = time.time()

# Clear or create the output file in advance
with open(output_file, "w") as f:
    pass

for i, atoms in enumerate(frames):
    frame_start = time.time()
    
    # Bind the NEP calculator
    atoms.calc = calc
    
    # Trigger the actual calculation (ASE will automatically update atoms.info['energy'] and atoms.arrays['forces'])
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    
    # Append the current frame with updated energy and forces to the new file
    write(output_file, atoms, format="extxyz", append=True)
    
    # Print progress
    if (i + 1) % max(1, total_frames // 10) == 0 or (i + 1) == total_frames:
        elapsed = time.time() - start_time
        print(f"Progress: [{i+1}/{total_frames}] | Total elapsed time: {elapsed:.2f}s")

print(f"\nCalculation complete! The NEP predicted trajectory has been saved to: {output_file}")
print(f"Total time taken: {time.time() - start_time:.2f} seconds.")
