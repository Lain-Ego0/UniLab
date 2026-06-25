# OpenDoge 四足机器人 UniLab 地形训练手册

uv run scripts/play_viser.py task=opendoge_joystick_rough/mujoco \
  interactive.action_mode=policy viser.port=8080


轻量四足（总重 ~4.80 kg，base 2.24 kg，12 DOF，4 腿 FL/FR/RL/RR，每腿 hip+thigh+calf）。

> 物理约束与机器人几何参数详见 [OpenDoge_flat.md](OpenDoge_flat.md)。

## 资产位置

| 类型 | 路径 |
|------|------|
| Robot XML | `src/unilab/assets/robots/opendoge/opendoge.xml` |
| 地形 Fragment | `src/unilab/assets/robots/opendoge/locomotion_task.xml` |
| Meshes | `src/unilab/assets/robots/opendoge/assets/`（STL） |
| Env 包 | `src/unilab/envs/locomotion/opendoge/` |
| PPO 地形配置 | `conf/ppo/task/opendoge_joystick_rough/mujoco.yaml` |
| Viser 地形渲染 | `src/unilab/visualization/viser_scene.py` (hfield→mesh 适配) |
| 测试 | `tests/test_opendoge.py` |

## 地形配置

地形由 `OpenDogeRoughTerrainCfg` 定义，按 2.2 kg 小狗保守缩放（R2 起用，R1 稍高后被降低）：

| 地形类型 | 比例 | 参数 | 说明 |
|----------|:----:|------|------|
| flat | 25% | 平地 | 保证基础行走不退化 |
| pyramid_stairs | 8% | 台阶 1-3 cm, step_width=0.4m | R1: 1-4cm |
| pyramid_stairs_inv | 8% | 倒台阶 1-3 cm | |
| hf_pyramid_slope | 15% | 斜坡 0-10%（~5.7°） | R1: 0-15% |
| hf_pyramid_slope_inv | 15% | 倒斜坡 0-10% | |
| random_rough | 15% | 粗糙地形 0.2-1.2 cm | R1: 0.3-1.5cm |
| wave_terrain | 14% | 波浪地形 0-3 cm | R1: 0-4cm |

网格：6×6 cells，每格 8×8m，边界 20m。

## 与平地任务的关键架构差异

| 方面 | 平地 (flat) | 地形 (rough) |
|------|-------------|--------------|
| Scene | `scene_flat.xml` | `opendoge.xml` + `locomotion_task.xml` fragment + terrain hfield |
| Actor obs | 49 维 | 49 维（同 flat，全可部署） |
| Critic obs | 52 维 | 239 维（52 + 187 height scan = 17×11 grid） |
| Base height | 平面采样 | `base_height_from_scan()` 地形感知 |
| Spawn | 原点 + 随机 XY | `apply_spawn()` 精确地形表面采样（R4 修复） |
| Termination | 倾斜 >60° | 倾斜 >72°（R5 放宽）+ terrain out-of-bounds |
| Fallen detection | — | base_z < 0.05 持续 1s → truncate（R5 新增） |
| max_episode | 20s | 30s（R5 延长） |
| 地形专有 reward | — | undesired_contacts, feet_slide, feet_height_body, feet_gait, upward 等 |

## 地形专有 Reward

| Reward | Scale | 说明 |
|--------|:-----:|------|
| undesired_contacts | -1.0 | 身体部位（base/hip/thigh/calf）触地惩罚 |
| feet_slide | -0.1 | 支撑相足端滑动速度惩罚 |
| feet_height_body | -5.0 | 摆动相足端离地过高惩罚（防止无效高抬腿） |
| feet_gait | 0.5 | 对角线步态同步/异步奖励 |
| feet_air_time_variance | -1.0 | 各腿飞行相/支撑相时间方差（步态均匀性） |
| feet_contact_without_cmd | 0.1 | 零指令时触地奖励（鼓励站立） |
| upward | 1.5 | 身体朝向重力反方向奖励（R2: 1.0→1.5） |

## 训练成果总览

| Round | Best | Final | Episode | 关键改动 |
|:-----:|:----:|:-----:|:-------:|:---------|
| R1 | 61.5 | 37.9 | 191 | 地形首训，保守参数 + R37 reward |
| R2 | 68.1 | 33.7 | 143 | 降地形难度 + entropy 8e-3（过高致抖动） |
| R3 | 94.6 | 43.9 | 162 | 回退 entropy 5e-3 + curriculum |
| R4 | 90.2 | 28.6 | 121 | spawn bug fix + base_height -100（反效果） |
| **R5** | **168.1** 🔥 | **60.1** 🔥 | **243** 🔥 | **放宽终止 + 延长时间 + 倒地检测** |

### Round 1 (地形首训, 1500 iters) — 保守地形 + R37 reward

```yaml
# 地形: 台阶 1-4cm / 斜坡 0-15% / 粗糙 0.3-1.5cm / 波浪 0-4cm / 平地 20%
# Reward: 继承 R37 平地最优 + 7 地形专有 reward
# Bug: origins_for 而非 apply_spawn（空中出生）
algo.num_envs: 2048
algo.max_iterations: 1500
entropy_coef: 5.0e-3
```

**结果**：best=**61.45**, final=**37.91**, episode=**191**。

**教训**：episode=191 摔倒频繁；reward 早期见顶（iter 166）后持续退化。

### Round 2 (降难度 + 高熵尝试, 2000 iters) — 过犹不及 ❌

```yaml
# 地形: 降为台阶 1-3cm / 斜坡 0-10% / 粗糙 0.2-1.2cm / 波浪 0-3cm / 平地 25%
# Reward: ang_vel_xy -0.7→-1.0, upward 1.0→1.5, tracking_ang_vel 1.5→1.8
algo.max_iterations: 2000
entropy_coef: 8.0e-3          # ← 过高！
```

**结果**：best=**68.1**, final=**33.7**, episode=**143** ❌。action_std 0.81 动作严重抖动。

**教训**：entropy 8e-3 过高 → 策略过于随机 → 存活率反降。

### Round 3 (回退熵 + Curriculum, 2000 iters)

```yaml
# 保持 R2 地形 + R2 reward scales
# 回退: entropy 8e-3→5e-3, action_smooth -0.003→-0.002
algo.max_iterations: 2000
entropy_coef: 5.0e-3
```

**结果**：best=**94.6**, final=**43.9**, episode=**162**。三项追踪全提升，身体摇摆减半。

### Round 4 (spawn fix + base_height 弱化, 2000 iters) ❌

```yaml
# spawn: origins_for → apply_spawn（修复空中出生 bug）
# base_height: -200 → -100（减轻地形高度惩罚）
# push_interval: 300 → 500
algo.max_iterations: 2000
```

**结果**：best=**90.2**, final=**28.6**, episode=**121** ❌。spawn 修复有效但 base_height -100 导致姿态崩坏，反噬全部指标。

**教训**：base_height 惩罚不能弱化，-200 是安全值；spawn fix 单独保留。

### Round 5 (放宽终止 + 延长时间 + 倒地检测, 2000 iters) 🔥🏆

```yaml
# 终止: gravity_z ≤0.5→≤0.3 (60°→72°)，跌倒后给恢复机会
# max_episode: 20s → 30s
# 新增: 倒地检测 (base_z < 0.05 持续 1s → truncate)
# spawn bug fix 保留
algo.max_iterations: 2000
```

**结果**：best=**168.14**, final=**60.08**, episode=**243**。

### 当前最优配置 (R5) 🏆

```yaml
# conf/ppo/task/opendoge_joystick_rough/mujoco.yaml
# @package _global_
training:
  task_name: OpenDogeJoystickRough
  sim_backend: mujoco
algo:
  num_envs: 2048
  max_iterations: 2000
  empirical_normalization: true
  obs_groups:
    actor: [actor]
    critic: [critic]
  policy:
    init_noise_std: 0.5
  algorithm:
    learning_rate: 3.0e-4
    entropy_coef: 5.0e-3
env:
  max_episode_seconds: 30.0
  scene:
    model_file: src/unilab/assets/robots/opendoge/opendoge.xml
    fragment_files:
      - src/unilab/assets/robots/opendoge/locomotion_task.xml
    terrain:
      hfield_name: terrain_hfield
      geom_name: floor
  terrain_curriculum:
    enabled: true
    promote_frac: 0.5
    demote_frac: 0.25
  terrain_scan:
    enabled: true
  termination_config:
    terrain_out_of_bounds: true
    terrain_distance_buffer: 3.0
  commands:
    rel_standing_envs: 0.1
    pure_axis_prob: 0.25
    resampling_time: 10.0
    heading_command: true
    vel_limit: [[-0.8, -0.6, -1.5], [0.8, 0.6, 1.5]]
  noise_config:
    level: 1.0
    scale_joint_angle: 0.08
    scale_joint_vel: 0.8
    scale_gyro: 0.35
  domain_rand:
    randomize_base_mass: true
    added_mass_range: [-0.6, 0.6]
    randomize_body_mass: true
    body_mass_multiplier_range: [0.92, 1.08]
    random_com: true
    com_offset_x: [-0.03, 0.03]
    com_offset_y: [-0.02, 0.02]
    com_offset_z: [-0.01, 0.01]
    randomize_ground_friction: true
    ground_friction_multiplier_range: [0.7, 1.3]
    randomize_gravity: true
    gravity_range: [[-0.5, -0.5, -9.81], [0.5, 0.5, -9.81]]
    push_robots: true
    push_interval: 300
    max_force: [0.8, 0.8, 0.5]
reward:
  scales:
    tracking_vx: 1.5
    tracking_vy: 1.8
    tracking_ang_vel: 1.8
    cross_axis_suppression: -0.6
    tracking_vel_linear: -0.4
    lin_vel_z: -5.0
    ang_vel_xy: -1.0
    base_height: -200.0
    action_rate: -0.004
    action_smooth: -0.002
    similar_to_default: -0.03
    dof_acc: -0.0000005
    stand_still: -0.5
    zero_command_stillness: 3.0
    torques: -0.005
    energy: -0.0001
    contact: 0.24
    swing_feet_z: 12.0
    feet_air_time: 0.5
    joint_mirror: -0.05
    hip_pos: -0.25
    undesired_contacts: -1.0
    feet_slide: -0.1
    feet_height_body: -5.0
    feet_gait: 0.5
    feet_air_time_variance: -1.0
    feet_contact_without_cmd: 0.1
    upward: 1.5
  tracking_sigma: 0.22
  base_height_target: 0.15
  contact_threshold: 1.0
  feet_air_time_threshold: 0.5
  feet_height_body_target: -0.1
  max_air_time: 0.25
```

### R5 最终指标

| Reward | 值 | Scale | 归一化 |
|--------|-----|:---:|:---:|
| tracking_vx | 1.236 | 1.5 | 82.4% |
| tracking_vy | 1.527 | 1.8 | 84.8% |
| tracking_ang_vel | 1.143 | 1.8 | 63.5% |
| swing_feet_z | 3.957 | 12.0 | 33.0% |
| feet_gait | 0.412 | 0.5 | 82.4% |
| upward | 5.698 | 6.0 | 95.0% |
| ang_vel_xy | -0.667 | -1.0 | ✅ |
| lin_vel_z | -0.061 | -5.0 | 98.8% |
| base_height | -0.027 | — | 精准 |
| undesired_contacts | -0.014 | -1.0 | 极少碰撞 |
| action_rate | -0.063 | — | ✅ |
| torques | -0.188 | -0.005 | 地形扭矩较高 |

### 五轮对比

| 指标 | R1 | R2 | R3 | R4 | R5 | 趋势 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|
| Best reward | 61.5 | 68.1 | 94.6 | 90.2 | **168.1** | 🔥🔥 |
| Final reward | 37.9 | 33.7 | 43.9 | 28.6 | **60.1** | 🔥 |
| Episode | 191 | 143 | 162 | 121 | **243** | 🔥 |
| tracking_vx | 90.7% | 90.0% | 93.4% | — | 82.4% | — |
| tracking_vy | 91.3% | 92.2% | 94.7% | — | 84.8% | — |
| swing_feet_z | 3.58 | 3.76 | 4.12 | — | 3.96 | — |

### 与平地 R37 对比

| 指标 | 平地 R37 | 地形 R5 |
|------|:------:|:-----:|
| tracking_vx | 1.44 (96%) | 1.24 (82%) |
| tracking_vy | 1.68 (94%) | 1.53 (85%) |
| swing_feet_z | 3.32 | **3.96** 🔥 |
| episode | **1000** | 243 |
| best reward | **182** | 168.1 |

**R5 首次让地形 best reward 接近平地水平**（168 vs 182），episode 从 162→243 (+50%)。

## 关键教训

1. **地形参数要按机器人尺寸缩放，且宁小勿大**：OpenDoge 2.2kg/0.17m 站高 → 台阶 ≤3cm、斜坡 ≤10%、粗糙 ≤1.2cm。R1→R2 降难度后追踪明显改善。
2. **entropy 在 terrain 上不能过高**：R2 entropy=8e-3 导致 action_std 0.81、动作抖动 + episode 反降。5e-3 是已验证的安全值。
3. **地形 reward 的 scale 需要适应当前难度**：upward 1.0→1.5、ang_vel_xy -0.7→-1.0、tracking_ang_vel 1.5→1.8 在 R3 中全部正面生效。
4. **spawn 必须用 apply_spawn 而非 origins_for**：R1-R3 的 origins_for 只取 cell 原点 Z，导致机器人在随机 XY 偏移后可能悬空或陷入地形。R4 修复后保留至今。
5. **放宽终止条件显著提升 episode**：gravity_z 阈值 0.5→0.3（60°→72°）+ max_episode 20→30s → episode +50%（162→243）。
6. **base_height 惩罚不能弱化**：R4 降至 -100 导致姿态崩坏，-200 是安全值。
7. **倒地检测是宽松终止的必要配套**：放宽终止后机器人可能躺着不动，base_z < 0.05 持续 1s 截断清理僵尸 env。
8. **terrain curriculum 需要配合足够的 episode length**：五轮中 curriculum 从未激活（episode 不够走远）。更长的 episode 或更低的 promote 阈值才能使 curriculum 真正生效。
9. **所有轮次都在 iter ~124-166 早期见顶**：策略在早期找到局部最优后持续退化。这是地形训练的系统性问题，curriculum 理论上应该缓解但尚未生效。
10. **reward 数值不可与平地对齐**：地形天然低压。R5「best~168 / final~60」是当前合理基线，best 首次接近平地水平。

## Viser 地形可视化修复

viser 原本不支持 MuJoCo hfield 渲染。修复方式：

| 文件 | 修改 |
|------|------|
| `src/unilab/visualization/viser_scene.py` | `MujocoViserScene` 新增 `terrain_cfg` 参数 + `_build_terrain_mesh()` — 用 `TerrainGenerator` 直接生成三角网格 |
| `scripts/play_viser.py` | `_build_scene_entries` 提取 `env.cfg.scene.terrain.generator` 传入场景 |

## 工作流

### 训练

```bash
# 标准训练
uv run train --algo ppo --task opendoge_joystick_rough --sim mujoco

# 快速验证
uv run train --algo ppo --task opendoge_joystick_rough --sim mujoco \
  algo.num_envs=256 algo.max_iterations=20
```

### 可视化

```bash
# Viser（浏览器操控方向，支持地形渲染）
uv run python scripts/play_viser.py task=opendoge_joystick_rough/mujoco \
  interactive.action_mode=policy viser.port=8080
```

打开 `http://localhost:8080`，右侧面板拖 `vx`/`vy`/`vyaw` 滑块操控方向。

### 日志位置

```
logs/rsl_rl_ppo/OpenDogeJoystickRough/<run>/
├── model_1999.pt      # checkpoint
├── policy.onnx         # ONNX 导出
├── policy.pt           # TorchScript
├── play_video.mp4      # 训练录像
├── run_summary.json    # 训练摘要
└── events.out.*        # TensorBoard
```
