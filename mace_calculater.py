import torch
from ase.io import read, write
from mace.calculators import MACECalculator
import time

# 1. Detect the hardware environment, preferring GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Current computing device: {device.upper()}")

# 2. Load the custom MACE model
model_path = "MACE.model"
print(f"Loading MACE model: {model_path} ...")
calc = MACECalculator(model_paths=model_path, device=device, default_dtype="float64") 
# Note: If you used float32 during training, you can change default_dtype to "float32" to improve speed

# 3. Read the extxyz trajectory to be calculated
input_file = "dft.extxyz"
output_file = "dft_recalculated.extxyz"

print(f"Reading trajectory file: {input_file} ...")
frames = read(input_file, index=":")
total_frames = len(frames)
print(f"Detected {total_frames} frames in total.")

# 4. Loop through to calculate the energy and forces for each frame, writing to the new file in real-time
print("Starting to calculate Energy & Forces ...")
start_time = time.time()

# Clear or create the output file in advance
with open(output_file, "w") as f:
    pass

for i, atoms in enumerate(frames):
    frame_start = time.time()
    
    # Bind the MACE calculator
    atoms.calc = calc
    
    # Trigger the actual calculation (ASE will automatically store the results in atoms.info['energy'] and atoms.arrays['forces'])
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    
    # Append the current frame with updated energy and forces to the new file
    write(output_file, atoms, format="extxyz", append=True)
    
    # Print progress
    if (i + 1) % max(1, total_frames // 10) == 0 or (i + 1) == total_frames:
        elapsed = time.time() - start_time
        print(f"Progress: [{i+1}/{total_frames}] | Current frame time: {time.time() - frame_start:.2f}s | Total elapsed time: {elapsed:.1f}s")

print(f"\nCalculation complete! The new trajectory has been successfully saved to: {output_file}")
print(f"Total time taken: {time.time() - start_time:.2f} seconds.")
