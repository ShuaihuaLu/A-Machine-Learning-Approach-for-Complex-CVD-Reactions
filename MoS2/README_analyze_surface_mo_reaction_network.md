# `analyze_surface_mo_reaction_network.py` 使用说明与代码解析

## 1. 程序用途

`analyze_surface_mo_reaction_network.py` 用于分析 **MoO\(_x\) + S\(_x\) / α-Al\(_2\)O\(_3\)** 表面反应体系的 LAMMPS 分子动力学轨迹。

程序采用 **逐帧流式读取（streaming）**，不会一次性把整个 `surface.lammpstrj` 读入内存，因此适合较长的 MD 轨迹。

它的核心目标是从原子轨迹中提取 Mo 在 Al\(_2\)O\(_3\) 表面的局域化学环境及其随时间的变化，包括：

1. 每个表面 Mo 的 Mo–S、Mo–O、Mo–Mo 配位数；
2. 每个 Mo 的局域 MoO\(_x\)S\(_y\) 状态；
3. MoO\(_x\)S\(_y\) 物种 population 随时间的演化；
4. 全局 sulfurization index；
5. O → S ligand exchange 与其他状态转换事件；
6. 状态 transition count matrix 和 transition probability matrix；
7. 每种 MoO\(_x\)S\(_y\) 状态的 residence time；
8. Mo 相对于瞬时 Al\(_2\)O\(_3\) 顶表面的高度；
9. Mo–Mo 配位，用于描述 precursor association / oligomerization；
10. Mo–O–Mo 与 Mo–S–Mo bridging motifs；
11. 自动输出适合进一步论文作图的 CSV 文件，并可生成 PNG/PDF/SVG 图。

---

# 2. 最基本的运行方法

假设轨迹文件名为：

```text
surface.lammpstrj
```

最简单运行：

```bash
python analyze_surface_mo_reaction_network.py surface.lammpstrj
```

推荐用于当前 MoO\(_x\)+S\(_x\)/Al\(_2\)O\(_3\) 体系的命令：

```bash
python analyze_surface_mo_reaction_network.py surface.lammpstrj \
    --dt-fs 1.0 \
    --stride 10 \
    --cut-mo-s 2.8 \
    --cut-mo-o 2.5 \
    --cut-mo-mo 3.8 \
    --surface-zmin -1.0 \
    --surface-zmax 5.0 \
    --pbc xy \
    --prefix surface_Mo
```

如果暂时只需要 CSV，不需要自动绘图：

```bash
python analyze_surface_mo_reaction_network.py surface.lammpstrj \
    --dt-fs 1.0 \
    --stride 10 \
    --no-plot
```

查看所有参数：

```bash
python analyze_surface_mo_reaction_network.py -h
```

---

# 3. Python 环境与依赖

建议：

```text
Python >= 3.9
```

必须依赖：

```bash
pip install numpy
```

如果需要自动绘图：

```bash
pip install pandas matplotlib
```

程序的核心轨迹分析只依赖 `numpy`。`pandas` 与 `matplotlib` 仅在绘图阶段使用。

---

# 4. LAMMPS 轨迹格式要求

程序读取标准 LAMMPS custom dump，例如：

```text
ITEM: TIMESTEP
5000
ITEM: NUMBER OF ATOMS
2360
ITEM: BOX BOUNDS pp pp ff
0.0 60.0
0.0 60.0
0.0 60.0
ITEM: ATOMS id type x y z
1 1 1.234 2.345 4.567
2 1 2.345 3.456 4.678
...
```

推荐 LAMMPS 输出：

```lammps
dump            traj all custom 100 surface.lammpstrj id type x y z
dump_modify     traj sort id
```

至少必须能够得到：

- 原子种类：`type` 或 `element`
- 原子坐标

程序支持以下坐标形式：

```text
x   y   z
xu  yu  zu
xs  ys  zs
xsu ysu zsu
```

其中：

- `x y z`：wrapped Cartesian coordinates；
- `xu yu zu`：unwrapped Cartesian coordinates；
- `xs ys zs`：scaled coordinates；
- `xsu ysu zsu`：unwrapped scaled coordinates。

程序同时支持：

- orthorhombic box；
- restricted triclinic LAMMPS box。

---

# 5. 默认 atom type 对应关系

默认设置为：

```text
type 1 = Al
type 2 = Mo
type 3 = O
type 4 = S
```

这与之前结构文件中的元素顺序：

```text
Al Mo O S
```

保持一致。

如果你的 LAMMPS data 文件不是这个映射，必须修改。例如：

```text
1 = Al
2 = O
3 = Mo
4 = S
```

运行：

```bash
python analyze_surface_mo_reaction_network.py surface.lammpstrj \
    --type-Al 1 \
    --type-O 2 \
    --type-Mo 3 \
    --type-S 4
```

**务必检查 atom type。错误的 type mapping 会导致所有后续分析失去物理意义。**

---

# 6. 时间是如何计算的？

LAMMPS dump 中的：

```text
ITEM: TIMESTEP
5000
```

表示的是 LAMMPS step number，而不是直接的 ps 或 ns。

程序使用：

\[
t = N_{\mathrm{step}} \times \Delta t
\]

其中 `--dt-fs` 是真实 MD integration timestep，单位为 fs。

默认：

```bash
--dt-fs 1.0
```

例如：

```text
TIMESTEP = 5000
MD timestep = 1 fs
```

则：

\[
t=5000\times1~\mathrm{fs}=5~\mathrm{ps}=0.005~\mathrm{ns}
\]

默认情况下，程序把**第一帧被分析的 frame 定义为 t = 0**。

即：

```python
ts_out = frame.timestep - first_analyzed_ts
```

如果希望保留绝对 LAMMPS 时间，则加入：

```bash
--absolute-time
```

---

# 7. `--stride` 的含义

例如：

```bash
--stride 1
```

表示每一个 dump frame 都分析。

```bash
--stride 10
```

表示每 10 个 dump frame 分析一次。

注意：这里的 stride 是 **dump frame stride**，不是 LAMMPS MD timestep。

假设 LAMMPS：

```lammps
timestep 0.001
dump traj all custom 100 surface.lammpstrj id type x y z
```

在 `units metal` 下：

```text
MD timestep = 1 fs
一个 dump frame = 100 fs = 0.1 ps
```

如果再设置：

```bash
--stride 10
```

那么实际分析的时间间隔为：

\[
100\times10\times1~\mathrm{fs}=1~\mathrm{ps}
\]

## 非常重要

`--stride` 会直接影响：

- transition detection；
- O → S exchange event；
- residence time；
- state lifetime；
- 短寿命 intermediate 的识别。

因此如果分析动力学，不建议一开始使用过大的 stride。

推荐先测试：

```text
stride = 1
stride = 5
stride = 10
```

检查结果是否收敛。

---

# 8. 如何定义 Al₂O₃ 顶表面？

每一个 frame 都会重新计算表面位置。

代码：

```python
def top_al_surface_z(al_z, layer_tol):
    zmax = np.max(al_z)
    top = al_z[al_z >= zmax - layer_tol]
    return np.mean(top)
```

首先找到最高 Al 原子的 z：

\[
z_{\max}^{\mathrm{Al}}
\]

然后选取：

\[
z_{\mathrm{Al}}\ge z_{\max}^{\mathrm{Al}}-\Delta z
\]

的 Al 原子，并使用这些 Al 的平均 z 定义：

\[
z_{\mathrm{surf}}
\]

默认：

```bash
--al-top-layer-tol 0.15
```

即最高 Al 以下 0.15 Å 范围内的 Al 被认为属于顶层。

这样做比直接使用单个最高 Al 更稳定，可以降低热振动造成的表面高度噪声。

---

# 9. 如何定义“表面 Mo”？

对于每一个 Mo：

\[
h_{\mathrm{Mo}}=z_{\mathrm{Mo}}-z_{\mathrm{surf}}
\]

默认只有满足：

\[
-1.0 \le h_{\mathrm{Mo}} \le 5.0~\text{Å}
\]

的 Mo 被纳入分析。

对应参数：

```bash
--surface-zmin -1.0
--surface-zmax 5.0
```

例如如果希望只看表面上方 0–4 Å 的 Mo：

```bash
--surface-zmin 0.0 \
--surface-zmax 4.0
```

这一步非常重要，因为它可以排除远离表面的气相 Mo-containing species。

---

# 10. 配位数定义

## 10.1 Mo–S coordination

默认：

\[
r_{\mathrm{Mo-S}}\le2.8~\text{Å}
\]

参数：

```bash
--cut-mo-s 2.8
```

得到：

\[
CN_{\mathrm{Mo-S}}
\]

---

## 10.2 Mo–O coordination

默认：

\[
r_{\mathrm{Mo-O}}\le2.5~\text{Å}
\]

参数：

```bash
--cut-mo-o 2.5
```

得到：

\[
CN_{\mathrm{Mo-O}}
\]

---

## 10.3 Mo–Mo coordination

默认：

\[
r_{\mathrm{Mo-Mo}}\le3.8~\text{Å}
\]

参数：

```bash
--cut-mo-mo 3.8
```

得到：

\[
CN_{\mathrm{Mo-Mo}}
\]

Mo–Mo CN 可以用作：

- precursor association；
- oligomerization；
- Mo-containing network connectivity

的简单结构 descriptor。

## 推荐做法

正式论文分析中，建议根据 RDF / pair-distance distribution 的第一谷值选择 cutoff，而不是只依赖经验值。

---

# 11. 周期性边界条件

默认：

```bash
--pbc xy
```

即：

```text
x = periodic
y = periodic
z = non-periodic
```

这通常适用于 surface slab。

可选：

```text
--pbc xy
--pbc xyz
--pbc none
--pbc auto
```

其中：

### `xy`

只对 x/y 使用 minimum image convention。

### `xyz`

三个方向都使用 PBC。

### `none`

完全不使用 PBC。

### `auto`

读取 LAMMPS `BOX BOUNDS` 中的 boundary flag，例如：

```text
pp pp ff
```

自动判断。

对典型 Al\(_2\)O\(_3\) slab，推荐：

```bash
--pbc xy
```

---

# 12. minimum-image 距离是如何计算的？

程序不是简单地直接计算 Cartesian distance，而是先把位移转换成 fractional coordinates：

```python
frac = delta @ hinv_t
```

对于周期方向：

```python
frac[..., ax] -= np.rint(frac[..., ax])
```

然后再返回 Cartesian space：

```python
frac @ h_t
```

因此能够正确处理：

- orthorhombic cells；
- restricted triclinic cells；
- 跨周期边界的 Mo–S / Mo–O / Mo–Mo 键。

为了避免一次生成过大的距离矩阵，Mo center 会被分块：

```python
chunk = 128
```

因此内存需求显著低于一次性计算全部 pair distances。

---

# 13. MoOₓSᵧ 状态如何定义？

程序定义：

```python
State = (CN_Mo_O, CN_Mo_S)
```

例如：

```text
(CN_O, CN_S) = (6,0) -> MoO6
(CN_O, CN_S) = (5,1) -> MoO5S
(CN_O, CN_S) = (4,2) -> MoO4S2
(CN_O, CN_S) = (1,4) -> MoOS4
(CN_O, CN_S) = (0,5) -> MoS5
```

如果：

```text
CN_O = 0
CN_S = 0
```

则记为：

```text
Mo_uncoord
```

注意：这里的 `MoO4S2` 是**局域配位标签**，不一定代表一个独立、具有整数化学计量比的自由分子。

它更准确的含义是：

> 当前 frame 中，该 Mo 周围 cutoff 内存在 4 个 O 和 2 个 S。

---

# 14. Sulfurization index

程序定义：

\[
\chi_S(t)=
\frac{\sum_i CN^{(i)}_{\mathrm{Mo-S}}}
{\sum_i CN^{(i)}_{\mathrm{Mo-S}}+\sum_i CN^{(i)}_{\mathrm{Mo-O}}}
\]

代码：

```python
total_s = np.sum(cn_s)
total_o = np.sum(cn_o)
chi_s = total_s / (total_s + total_o)
```

其物理意义：

```text
χS → 0 : O-rich coordination environment
χS → 1 : S-rich coordination environment
```

对应的 oxygen coordination fraction：

\[
\chi_O=1-\chi_S
\]

只要总 Mo–O + Mo–S 配位数非零。

`χS` 是一个很适合比较不同温度或不同前处理条件的整体 reaction coordinate。

---

# 15. 单个 Mo 的 S fraction

每个 Mo 还会计算：

\[
f_S^{(i)}=
\frac{CN^{(i)}_{\mathrm{Mo-S}}}
{CN^{(i)}_{\mathrm{Mo-S}}+CN^{(i)}_{\mathrm{Mo-O}}}
\]

输出字段：

```text
S_fraction
```

例如：

```text
MoO6      -> 0.000
MoO5S     -> 1/6
MoO4S2    -> 2/6
MoOS4     -> 4/5
MoS5      -> 1.000
```

这个值适合用于：

- 单个 Mo 的 sulfurization degree；
- 空间着色；
- `height vs S_fraction` correlation；
- 局域环境 clustering。

---

# 16. Transition 是如何定义的？

程序使用稳定的 `Mo_id` 在连续**被分析的 frame**之间追踪同一个 Mo。

例如：

```text
frame n:     Mo 769 = MoO5
frame n+1:   Mo 769 = MoO4S
```

则：

```text
delta_O = -1
delta_S = +1
```

分类为：

```text
O_to_S_exchange
```

程序的 transition 分类包括：

```text
O_to_S_exchange
S_to_O_exchange
O_loss
O_gain
S_loss
S_gain
concerted_other
unchanged
```

对应代码：

```python
if do < 0 and ds > 0:
    return "O_to_S_exchange"
```

这里的定义比较宽松。例如：

```text
MoO5 -> MoO3S2
```

因为：

```text
delta_O = -2
delta_S = +2
```

也会被归类为 `O_to_S_exchange`。

它代表的是“净 O 减少同时净 S 增加”，不意味着程序已经解析出严格的一步 elementary reaction。

---

# 17. Transition matrix

程序会统计连续分析 frame 中：

\[
i\rightarrow j
\]

的次数。

例如：

\[
\mathrm{MoO_6}\rightarrow\mathrm{MoO_5S}
\]

出现 100 次，则 matrix 中：

```text
MoO6 -> MoO5S = 100
```

程序同时包含 diagonal persistence，例如：

\[
\mathrm{MoO_6}\rightarrow\mathrm{MoO_6}
\]

因此 transition count matrix 既包含状态转换，也包含状态保持。

概率矩阵进行 row normalization：

\[
P_{ij}=\frac{N_{i\rightarrow j}}
{\sum_jN_{i\rightarrow j}}
\]

所以每一行总和约为 1。

注意：这是基于离散采样 frame 的 empirical transition probability，不应在没有进一步 Markov-state 验证的情况下直接等同于严格动力学速率常数。

---

# 18. Residence time 是如何定义的？

程序对每一个 `Mo_id` 维护一个 active state run。

例如：

```text
0.00 ns  MoO6
0.01 ns  MoO6
0.02 ns  MoO6
0.03 ns  MoO5S
```

则 MoO6 residence interval 约为：

\[
0.03-0.00=0.03~\mathrm{ns}
\]

结束原因包括：

```text
transition
left_surface
trajectory_end
```

起始原因包括：

```text
initial
entered_surface
transition
```

如果状态一直持续到轨迹结束，则：

```text
right_censored = 1
```

说明这个 residence time 只是下限，因为真正的终止时间未知。

对于严谨的 lifetime 分析，建议在进一步统计时区分：

```text
right_censored = 0
right_censored = 1
```

而不是无条件把所有 interval 当作完整寿命。

---

# 19. Mo–O–Mo 与 Mo–S–Mo bridge 如何识别？

程序计算每个 O 或 S 同时与多少个“被选中的表面 Mo”配位。

例如一个 O：

```text
Mo1 - O - Mo2
```

则该 O 的 degree：

```text
degree = 2
```

因此被定义为 bridging O。

程序输出：

```text
n_bridging_O
n_bridging_S
```

即具有：

\[
degree\ge2
\]

的 O/S 原子数量。

另外还输出：

```text
n_Mo_O_Mo_pairs
n_Mo_S_Mo_pairs
```

对于一个 ligand 如果连接 `d` 个 Mo，则产生：

\[
\binom{d}{2}=\frac{d(d-1)}{2}
\]

个 Mo–X–Mo pair。

所以代码使用：

```python
degree * (degree - 1) // 2
```

例如一个 S 同时连接 3 个 Mo：

\[
\binom{3}{2}=3
\]

会贡献 3 个 Mo–S–Mo pair。

因此：

```text
n_bridging_S
```

和：

```text
n_Mo_S_Mo_pairs
```

不是同一个物理量。

---

# 20. 输出文件总览

假设：

```bash
--prefix surface_Mo
```

程序会生成：

```text
surface_Mo_atoms.csv
surface_Mo_time_series.csv
surface_Mo_species_population.csv
surface_Mo_transitions.csv
surface_Mo_transition_counts.csv
surface_Mo_transition_probabilities.csv
surface_Mo_residence_times.csv
surface_Mo_residence_summary.csv
surface_Mo_species_height_summary.csv
```

如果没有使用 `--no-plot`，还会生成：

```text
surface_Mo_evolution.png
surface_Mo_evolution.pdf
surface_Mo_evolution.svg

surface_Mo_mechanism.png
surface_Mo_mechanism.pdf
surface_Mo_mechanism.svg
```

---

# 21. `surface_Mo_atoms.csv`

这是最详细的逐原子输出。

每一行代表：

> 一个 analyzed frame 中的一个 surface Mo。

主要字段：

| 字段 | 含义 |
|---|---|
| `frame_index` | dump frame 的 0-based index |
| `timestep` | LAMMPS timestep |
| `time_ps` | 物理时间 / ps |
| `time_ns` | 物理时间 / ns |
| `Mo_id` | LAMMPS atom ID |
| `x_A,y_A,z_A` | Mo Cartesian coordinates / Å |
| `Al_top_z_A` | 当前 frame 的 Al 顶表面高度 |
| `height_above_Al_top_A` | Mo 相对表面高度 |
| `CN_Mo_S` | Mo–S CN |
| `CN_Mo_O` | Mo–O CN |
| `CN_Mo_Mo` | Mo–Mo CN |
| `species` | MoO\(_x\)S\(_y\) label |
| `S_fraction` | 单个 Mo 的 sulfurization fraction |
| `nearest_S_A` | 最近 S 距离 |
| `nearest_O_A` | 最近 O 距离 |
| `nearest_Mo_A` | 最近其他 Mo 距离 |

这个文件最适合做：

- atom-resolved trajectory tracking；
- Mo-specific heatmap；
- height vs species；
- S fraction vs position；
- individual reaction pathway。

---

# 22. `surface_Mo_time_series.csv`

每一行代表一个 analyzed frame 的整体统计。

主要包含：

```text
n_surface_Mo

mean_CN_Mo_S
std_CN_Mo_S
median_CN_Mo_S

mean_CN_Mo_O
std_CN_Mo_O
median_CN_Mo_O

mean_CN_Mo_Mo
std_CN_Mo_Mo
median_CN_Mo_Mo

sum_CN_Mo_S
sum_CN_Mo_O
sum_CN_Mo_Mo

sulfurization_index
oxygen_coord_fraction

n_bridging_O
n_bridging_S
n_Mo_O_Mo_pairs
n_Mo_S_Mo_pairs

mean_surface_Mo_height_A
std_surface_Mo_height_A

n_state_changes
n_O_to_S_events
n_S_to_O_events
```

这个文件最适合画：

\[
\langle CN_{Mo-S}\rangle(t)
\]

\[
\langle CN_{Mo-O}\rangle(t)
\]

\[
\chi_S(t)
\]

\[
N_{Mo-O-Mo}(t)
\]

\[
N_{Mo-S-Mo}(t)
\]

等全局演化曲线。

---

# 23. `surface_Mo_species_population.csv`

每个 frame 统计所有观测到的 MoO\(_x\)S\(_y\) 状态。

对于每种 species 同时输出：

```text
N_MoO6
N_MoO5S
N_MoO4S2
...
```

以及 fraction：

```text
f_MoO6
f_MoO5S
f_MoO4S2
...
```

其中：

\[
f_i=\frac{N_i}{N_{\mathrm{surface~Mo}}}
\]

这个文件非常适合画：

\[
P(\mathrm{MoO_xS_y},t)
\]

二维 heatmap。

---

# 24. `surface_Mo_transitions.csv`

只记录发生状态变化的事件。

例如：

```text
Mo_id       = 769
old_species = MoO5
new_species = MoO4S
delta_O     = -1
delta_S     = +1
event_type  = O_to_S_exchange
```

适合分析：

- O → S substitution；
- O loss；
- S gain；
- reversible exchange；
- 单个 Mo 的 sequential reaction pathway。

---

# 25. Transition matrices

## `surface_Mo_transition_counts.csv`

原始 transition count matrix。

## `surface_Mo_transition_probabilities.csv`

row-normalized transition probability matrix。

适合进一步画：

- heatmap；
- network graph；
- Sankey-like reaction network；
- dominant pathway analysis。

---

# 26. Residence-time 文件

## `surface_Mo_residence_times.csv`

保存每一个独立 residence interval。

主要字段：

```text
Mo_id
species
CN_O
CN_S
start_frame
end_frame
start_time_ns
end_time_ns
duration_ns
n_observations
start_reason
end_reason
right_censored
```

## `surface_Mo_residence_summary.csv`

按 species 汇总：

```text
n_intervals
mean_duration_ns
std_duration_ns
min_duration_ns
max_duration_ns
```

适合判断：

- stable precursor；
- metastable intermediate；
- short-lived transient state。

---

# 27. `surface_Mo_species_height_summary.csv`

按 MoO\(_x\)S\(_y\) state 汇总 Mo 相对 Al\(_2\)O\(_3\) 表面的高度。

输出：

```text
species
CN_O
CN_S
n_samples
mean_height_A
std_height_A
min_height_A
max_height_A
```

可以分析：

\[
\mathrm{sulfurization}\quad\leftrightarrow\quad\mathrm{surface~binding}
\]

例如检验：

\[
f_S\uparrow\quad\Rightarrow\quad h_{Mo}\uparrow
\]

是否成立。

如果成立，可能意味着 sulfurization 伴随 precursor 与 sapphire 表面锚定减弱。

---

# 28. 自动生成的图

## `surface_Mo_evolution.*`

包含三个 panel：

### a

平均 Mo–O 和 Mo–S coordination number 随时间变化。

### b

atom-resolved Mo–O CN heatmap。

### c

atom-resolved Mo–S CN heatmap。

---

## `surface_Mo_mechanism.*`

包含：

### a

\[
\langle CN_{Mo-O}\rangle,
\langle CN_{Mo-S}\rangle,
\langle CN_{Mo-Mo}\rangle
\]

随时间变化。

### b

Sulfurization index：

\[
\chi_S(t)
\]

### c

Mo–O–Mo / Mo–S–Mo bridge-pair evolution。

### d

主要 MoO\(_x\)S\(_y\) states 的 population fraction heatmap。

PNG 默认 600 dpi，同时提供 PDF/SVG 矢量输出。

---

# 29. 代码整体结构

程序大致分为 4 个层级。

## 第一层：轨迹读取

主要函数：

```python
read_lammps_dump()
_parse_box()
get_cartesian_positions()
get_elements()
get_ids()
```

作用：

> 把每一帧 LAMMPS dump 转换成统一的 `Frame` 对象。

---

## 第二层：几何与配位分析

主要函数：

```python
minimum_image_deltas()
coordination_details()
top_al_surface_z()
```

作用：

- PBC minimum-image distance；
- coordination number；
- nearest-neighbor distance；
- ligand neighbor list；
- ligand bridging degree；
- 表面高度。

---

## 第三层：化学状态与动力学

主要函数：

```python
species_name()
classify_transition()
update_online_stats()
```

以及 `main()` 中的：

```text
population tracking
transition tracking
residence tracking
bridge counting
sulfurization coordinate
```

---

## 第四层：输出和绘图

主要函数：

```python
make_evolution_plot()
make_mechanism_plot()
```

以及 CSV 输出部分。

---

# 30. 为什么使用 streaming reader？

代码：

```python
def read_lammps_dump(path):
    ...
    yield Frame(...)
```

这里使用 Python generator。

意味着：

```text
读 frame 1
→ 分析
→ 丢弃大部分坐标
→ 读 frame 2
→ 分析
→ ...
```

而不是：

```text
先把几十 GB 或几百 GB trajectory 全部读进 RAM
```

因此对于长轨迹特别重要。

程序只把一些小型结果长期保存在内存中，例如：

- species population；
- transition counter；
- active residence runs；
- online height statistics。

原子级数据则直接逐行写入 CSV。

---

# 31. `start-frame` 与 `stop-frame`

如果只分析轨迹的一部分：

```bash
--start-frame 1000
--stop-frame 5000
```

表示分析 dump frame index：

```text
1000 <= frame < 5000
```

注意 frame index 是 0-based。

例如第一帧：

```text
frame_index = 0
```

---

# 32. 推荐用于生产计算的命令

对于 slab + gas/surface species：

```bash
python analyze_surface_mo_reaction_network.py surface.lammpstrj \
    --dt-fs 1.0 \
    --stride 5 \
    --cut-mo-s 2.8 \
    --cut-mo-o 2.5 \
    --cut-mo-mo 3.8 \
    --surface-zmin -1.0 \
    --surface-zmax 5.0 \
    --al-top-layer-tol 0.15 \
    --pbc xy \
    --type-Al 1 \
    --type-Mo 2 \
    --type-O 3 \
    --type-S 4 \
    --prefix analysis/surface_Mo
```

程序会自动建立 `analysis/` 目录。

---

# 33. 多温度轨迹推荐组织方式

例如：

```text
800K/surface.lammpstrj
950K/surface.lammpstrj
1100K/surface.lammpstrj
1250K/surface.lammpstrj
```

分别运行：

```bash
python analyze_surface_mo_reaction_network.py 800K/surface.lammpstrj \
    --dt-fs 1.0 --stride 10 --prefix 800K/surface_Mo

python analyze_surface_mo_reaction_network.py 950K/surface.lammpstrj \
    --dt-fs 1.0 --stride 10 --prefix 950K/surface_Mo

python analyze_surface_mo_reaction_network.py 1100K/surface.lammpstrj \
    --dt-fs 1.0 --stride 10 --prefix 1100K/surface_Mo

python analyze_surface_mo_reaction_network.py 1250K/surface.lammpstrj \
    --dt-fs 1.0 --stride 10 --prefix 1250K/surface_Mo
```

注意不同温度之间必须尽量保持：

```text
相同 cutoff
相同 surface window
相同 stride
相同 dump interval
相同时间定义
```

否则 transition frequency、residence time 等动力学量不能直接比较。

---

# 34. 当前最重要的物理限制：所有 O 默认混在一起

当前代码：

```python
o_mask = elems == "O"
```

这意味着：

- Al\(_2\)O\(_3\) substrate O；
- MoO\(_x\) precursor O；
- 其他 O-containing species 中的 O

如果它们使用同一个 atom type，都会进入：

\[
CN_{Mo-O}
\]

因此目前的 `CN_Mo_O` 表示：

> Mo 周围所有 O 的总配位数。

它不能自动区分：

\[
CN_{Mo-O_{sub}}
\]

和：

\[
CN_{Mo-O_{precursor}}
\]

## 推荐改进

如果可以重新输出 trajectory，最好在 LAMMPS 中保留：

- 不同 atom type；或
- `mol` ID；或
- 可用于区分来源的 atom ID range。

例如：

```text
type 3 = substrate O
type 4 = precursor O
type 5 = S
```

这样后续可以分别分析：

\[
CN_{Mo-O_{sub}}(t)
\]

与：

\[
CN_{Mo-O_{prec}}(t)
\]

这对于区分：

- MoO\(_x\) 内部 Mo–O bond breaking；
- Mo–O\(_{surface}\) anchoring

非常关键。

---

# 35. 第二个重要限制：CN 是 cutoff-based descriptor

配位数采用 hard cutoff：

```text
r <= cutoff : bonded
r > cutoff  : not bonded
```

因此如果某根键在 cutoff 附近热振动，例如：

```text
2.49 Å
2.51 Å
2.48 Å
2.52 Å
```

可能造成：

```text
bonded
unbonded
bonded
unbonded
```

从而产生人工的高频 state switching。

解决方法包括：

1. 根据 RDF 第一谷值合理选 cutoff；
2. 使用 hysteresis cutoff；
3. 对 state trajectory 进行最短寿命过滤；
4. 要求状态连续存在至少 N 个 analyzed frames；
5. 对关键 transition 返回原始轨迹人工验证。

如果后续要把 transition rate 用于定量动力学，这是非常重要的步骤。

---

# 36. 第三个重要限制：transition 不等于 elementary reaction

例如：

```text
MoO5 -> MoO3S2
```

在两个 sampled frames 之间发生。

程序只能知道：

\[
\Delta O=-2,\quad\Delta S=+2
\]

但中间真实过程可能是：

\[
MoO_5
\rightarrow
MoO_4S
\rightarrow
MoO_3S_2
\]

只是因为输出/分析时间分辨率不够，没有看到中间态。

所以：

> trajectory sampling interval 决定了你能够解析的 reaction temporal resolution。

这也是为什么动力学分析时 `--stride` 不宜过大。

---

# 37. 第四个限制：表面判据假设 slab 顶面没有跨越 z 周期边界

程序通过最高 Al z 识别 top surface。

因此假设：

- slab 顶面在 simulation box 中连续；
- slab 没有整体跨越 z boundary；
- z wrapping 没有把同一个 slab 分裂到 box 顶部和底部。

如果 z 方向为 periodic 且 slab 跨边界，需要先重新 wrap / center slab，或者修改表面识别方法。

---

# 38. 推荐的科学分析顺序

对于 MoO\(_x\)+S\(_x\)/Al\(_2\)O\(_3\) CVD surface chemistry，建议按下面顺序分析。

## Step 1：验证 cutoff

先计算：

```text
Mo–O RDF / pair-distance distribution
Mo–S RDF / pair-distance distribution
Mo–Mo RDF / pair-distance distribution
```

确定第一谷值。

---

## Step 2：看整体 coordination evolution

分析：

\[
\langle CN_{Mo-O}\rangle(t)
\]

\[
\langle CN_{Mo-S}\rangle(t)
\]

以及：

\[
\chi_S(t)
\]

判断是否存在整体 sulfurization。

---

## Step 3：看 MoOₓSᵧ population

画：

\[
P(\mathrm{MoO_xS_y},t)
\]

寻找主要 intermediate states。

---

## Step 4：看 transition network

分析：

\[
MoO_xS_y\rightarrow MoO_{x'}S_{y'}
\]

找到 dominant pathway。

---

## Step 5：看 residence time

判断哪些状态是：

```text
stable precursor
metastable intermediate
short-lived fluctuation
```

---

## Step 6：看 bridge conversion

比较：

\[
N_{Mo-O-Mo}(t)
\]

与：

\[
N_{Mo-S-Mo}(t)
\]

判断是否发生 network chemistry reconstruction。

---

## Step 7：看 substrate coupling

分析：

\[
h_{Mo}(t)
\]

以及：

\[
h_{Mo}\;vs\;MoO_xS_y
\]

如果未来区分 substrate O 和 precursor O，则进一步看：

\[
CN_{Mo-O_{sub}}(t)
\]

---

## Step 8：多温度动力学

对于：

```text
800 K
950 K
1100 K
1250 K
```

比较：

\[
\chi_S(t,T)
\]

\[
P(MoO_xS_y,t,T)
\]

\[
\tau_{MoO_xS_y}(T)
\]

以及 transition frequency。

在定义充分严谨、采样充分的情况下，可进一步讨论 effective sulfurization kinetics 和 temperature dependence。

---

# 39. 建议的论文级核心图

基于本程序输出，可以组织为一组机制图：

### Panel a

\[
\langle CN_{Mo-O}\rangle(t)
\quad\text{and}\quad
\langle CN_{Mo-S}\rangle(t)
\]

### Panel b

\[
\chi_S(t)
\]

### Panel c

MoO\(_x\)S\(_y\) population heatmap。

### Panel d

主要 state transition network：

\[
MoO_x
\rightarrow
MoO_xS_y
\rightarrow
MoS_x
\]

### Panel e

\[
N_{Mo-O-Mo}(t)
\quad vs \quad
N_{Mo-S-Mo}(t)
\]

### Panel f

species-resolved Mo surface height 或 residence time。

这种组织方式比只展示平均 coordination number 更能支撑“surface sulfurization mechanism”。

---

# 40. 常见问题

## Q1. 为什么没有找到 Mo？

通常是 atom type mapping 错误。

检查：

```bash
head -50 surface.lammpstrj
```

以及 LAMMPS data 文件中的：

```text
Masses
```

确认 type ID。

---

## Q2. 为什么 `n_surface_Mo = 0`？

可能是：

- surface window 太窄；
- z 坐标定义不合适；
- top Al surface 识别错误；
- slab 跨 z boundary。

可尝试：

```bash
--surface-zmin -3
--surface-zmax 10
```

先检查，再收紧范围。

---

## Q3. 为什么 transition 太多？

可能原因：

- cutoff 在第一配位壳层边缘；
- trajectory thermal noise 大；
- dump 太密但没有做 hysteresis；
- 某些键频繁跨过 cutoff。

建议检查原始 bond-distance trajectory。

---

## Q4. 为什么 residence time 很短？

先排除：

- cutoff noise；
- stride 设置；
- surface window crossing；
- 临界距离抖动。

特别是 Mo 在 `surface-zmax` 附近来回运动，会产生 `left_surface` / `entered_surface` interval。

---

## Q5. 可以分析所有 Mo，而不仅是 surface Mo 吗？

目前没有单独的 `--all-mo` 参数，但可以人为把窗口设得很宽，例如：

```bash
--surface-zmin -1000 \
--surface-zmax 1000
```

不过对于真正的 gas-phase + surface system，更推荐保留明确的 surface definition。

---

# 41. 推荐下一步扩展

当前版本已经覆盖主要 coordination/state analyses。后续最值得增加的是：

1. **substrate O / precursor O 分离**；
2. **Mo–O / Mo–S individual bond lifetime**；
3. **bond survival correlation function**；
4. **state hysteresis / minimum lifetime filtering**；
5. **Mo lateral MSD**；
6. **species-resolved diffusion**；
7. **temperature-combined analysis script**；
8. **自动生成 transition-network figure**；
9. **RDF-based automatic cutoff determination**。

其中第 1 项对当前 MoO\(_x\)/Al\(_2\)O\(_3\) 表面反应问题尤其重要。

---

# 42. 简短总结

这个程序的逻辑可以概括为：

```text
LAMMPS trajectory
        |
        v
逐帧读取 atomic positions
        |
        v
动态确定 Al2O3 top surface
        |
        v
筛选 surface Mo
        |
        v
计算 Mo-O / Mo-S / Mo-Mo neighbors
        |
        v
定义 MoOxSy local state
        |
        +-----------------------------+
        |                             |
        v                             v
population vs time             atom-resolved CN
        |                             |
        v                             v
transition network             surface height
        |
        v
residence time
        |
        v
O-to-S exchange / sulfurization kinetics
```

因此它不只是一个简单的 coordination-number 脚本，而是一个针对 **MoO\(_x\) → MoO\(_x\)S\(_y\) → S-rich surface species** 演化过程的结构动力学分析框架。

