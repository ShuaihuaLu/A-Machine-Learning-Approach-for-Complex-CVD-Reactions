import torch
from ase.io import read, write
from mace.calculators import MACECalculator
import time

# 1. 检测硬件环境，优先使用 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前使用的计算设备: {device.upper()}")

# 2. 加载自定义 MACE 模型
model_path = "BN_Ni_stagetwo.model"
print(f"正在加载 MACE 模型: {model_path} ...")
calc = MACECalculator(model_paths=model_path, device=device, default_dtype="float64") 
# 注：如果你训练时用的是 float32，可以把 default_dtype 改为 "float32" 以提升速度

# 3. 读取待计算的 extxyz 轨迹
input_file = "dft.extxyz"
output_file = "dft_recalculated.extxyz"

print(f"正在读取轨迹文件: {input_file} ...")
frames = read(input_file, index=":")
total_frames = len(frames)
print(f"共检测到 {total_frames} 帧结构。")

# 4. 循环计算每一帧的能量和力，并实时写入新文件
print("开始计算能量与力 (Energy & Forces) ...")
start_time = time.time()

# 提前清空或创建输出文件
with open(output_file, "w") as f:
    pass

for i, atoms in enumerate(frames):
    frame_start = time.time()
    
    # 绑定 MACE 计算器
    atoms.calc = calc
    
    # 触发实际计算 (ASE 会自动将结果存入 atoms.info['energy'] 和 atoms.arrays['forces'])
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    
    # 将更新了能量和力的当前帧追加写入到新文件中
    write(output_file, atoms, format="extxyz", append=True)
    
    # 打印进度条
    if (i + 1) % max(1, total_frames // 10) == 0 or (i + 1) == total_frames:
        elapsed = time.time() - start_time
        print(f"进度: [{i+1}/{total_frames}] | 当前帧耗时: {time.time() - frame_start:.2f}s | 总累计耗时: {elapsed:.1f}s")

print(f"\n计算完成！新轨迹已成功保存至: {output_file}")
print(f"总共耗时: {time.time() - start_time:.2f} 秒。")
