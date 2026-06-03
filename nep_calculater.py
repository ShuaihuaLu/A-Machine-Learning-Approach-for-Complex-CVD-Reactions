import time
from ase.io import read, write

# 导入 NEP 计算器
try:
    from pynep.calculate import NEP
except ImportError:
    print("请先安装 pynep: pip install pynep")
    # 如果你想用 GPU，可以改为: 
    # from calorine.calculators import GPUNEP as NEP
    exit()

# 1. 加载 NEP 模型
model_path = "nep.txt"
print(f"正在加载 NEP 模型: {model_path} ...")
calc = NEP(model_path)

# 2. 读取待计算的 extxyz 轨迹 (这里以之前的 mace.extxyz 为例)
input_file = "dft.extxyz"
output_file = "nep_recalculated.extxyz"

print(f"正在读取轨迹文件: {input_file} ...")
frames = read(input_file, index=":")
total_frames = len(frames)
print(f"共检测到 {total_frames} 帧结构。")

# 3. 循环计算每一帧的能量和力
print("开始计算能量与力 (Energy & Forces) ...")
start_time = time.time()

# 提前清空或创建输出文件
with open(output_file, "w") as f:
    pass

for i, atoms in enumerate(frames):
    frame_start = time.time()
    
    # 绑定 NEP 计算器
    atoms.calc = calc
    
    # 触发实际计算 (ASE 会自动更新 atoms.info['energy'] 和 atoms.arrays['forces'])
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    
    # 将包含新能量和力的当前帧追加写入到新文件中
    write(output_file, atoms, format="extxyz", append=True)
    
    # 打印进度条
    if (i + 1) % max(1, total_frames // 10) == 0 or (i + 1) == total_frames:
        elapsed = time.time() - start_time
        print(f"进度: [{i+1}/{total_frames}] | 累计耗时: {elapsed:.2f}s")

print(f"\n计算完成！NEP 预测轨迹已保存至: {output_file}")
print(f"总共耗时: {time.time() - start_time:.2f} 秒。")
