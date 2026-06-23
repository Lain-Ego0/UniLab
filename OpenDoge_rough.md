# OpenDoge 四足机器人 UniLab 地形训练手册

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
| Spawn | 原点 + 随机 XY | terrain cell 原点 + 随机 yaw + 高度采样 |
| Termination | 倾斜 >60° | 倾斜 >60° + terrain out-of-bounds |
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
| **R3** | **94.6** 🔥 | **43.9** 🔥 | **162** | **回退 entropy 5e-3 + curriculum** |

### Round 1 (地形首训, 1500 iters) — 保守地形 + R37 reward

```yaml
# 地形: 台阶 1-4cm / 斜坡 0-15% / 粗糙 0.3-1.5cm / 波浪 0-4cm / 平地 20%
# Reward: 继承 R37 平地最优 + 7 地形专有 reward
algo.num_envs: 2048
algo.max_iterations: 1500
entropy_coef: 5.0e-3
```

**结果**：best=**61.45**, final=**37.91**, episode=**191**。

| 指标 | 值 | 归一化 | 评估 |
|------|-----|:---:|------|
| tracking_vx | 1.361 | 90.7% | ✅ |
| tracking_vy | 1.643 | 91.3% | ✅ |
| tracking_ang_vel | 1.105 | 73.7% | ⚠️ |
| swing_feet_z | 3.581 | 29.8% | ⚠️ |
| ang_vel_xy | -0.672 | — | ⚠️ 地形摇摆 |
| action_rate/smooth | -0.09/-0.04 | — | ✅ |

**教训**：episode=191 摔倒频繁；reward 早期见顶（iter 166）后持续退化。

### Round 2 (降难度 + 高熵尝试, 2000 iters) — 过犹不及 ❌

```yaml
# 地形: 台阶 1-3cm / 斜坡 0-10% / 粗糙 0.2-1.2cm / 波浪 0-3cm / 平地 25%
# Reward: ang_vel_xy -0.7→-1.0, upward 1.0→1.5, tracking_ang_vel 1.5→1.8
#         lin_vel_z -4.0→-5.0, action_smooth -0.002→-0.003
algo.max_iterations: 2000
entropy_coef: 8.0e-3          # ← 过高！
```

**结果**：best=**68.1**, final=**33.7**, episode=**143** ❌。

| 指标 | R1 | R2 | 判定 |
|------|:--:|:--:|:--:|
| action_std | 0.57 | **0.81** | ❌ 过于随机 |
| action_rate | -0.09 | **-0.16** | ❌ +80% |
| action_smooth | -0.04 | **-0.12** | ❌ +173% |
| episode | 191 | **143** | ❌ 反降 |

**教训**：entropy 8e-3 过高 → 策略过于随机 → 动作抖动严重 + 存活率反降。

### Round 3 (回退熵 + Curriculum, 2000 iters) 🔥🏆

```yaml
# 保持 R2 地形难度 + R2 reward scales
# 回退: entropy 8e-3→5e-3, action_smooth -0.003→-0.002
# 新增: terrain_curriculum enabled=true
algo.max_iterations: 2000
entropy_coef: 5.0e-3          # 回退 R1 值
```

**结果**：best=**94.64**, final=**43.94**, episode=**162**。

### 当前最优配置 (R3) 🏆

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

### R3 最终指标

| Reward | 值 | Scale | 归一化 |
|--------|-----|:---:|:---:|
| tracking_vx | 1.401 | 1.5 | **93.4%** 🔥 |
| tracking_vy | 1.705 | 1.8 | **94.7%** 🔥 |
| tracking_ang_vel | 1.439 | 1.8 | **80.0%** 🔥 |
| tracking_vel_linear | -0.093 | -0.4 | 76.8% |
| swing_feet_z | 4.120 | 12.0 | 34.3% 🔥 |
| feet_gait | 0.441 | 0.5 | **88.3%** |
| upward | 5.994 | 6.0 | 99.9% |
| ang_vel_xy | -0.547 | -1.0 | ✅ 大幅改善 |
| lin_vel_z | -0.040 | -5.0 | 99.2% |
| base_height | -0.010 | — | 精准 |
| undesired_contacts | -0.001 | -1.0 | 几乎无碰撞 |
| action_rate | -0.038 | — | ✅ 平滑 |
| action_smooth | -0.019 | — | ✅ 平滑 |
| torques | -0.117 | -0.005 | ✅ |

### 三轮对比

| 指标 | R1 | R2 | R3 | 趋势 |
|------|:--:|:--:|:--:|:--:|
| Best reward | 61.5 | 68.1 | **94.6** | 🔥 |
| Final reward | 37.9 | 33.7 | **43.9** | 🔥 |
| tracking_vx | 90.7% | 90.0% | **93.4%** | 🔥 |
| tracking_vy | 91.3% | 92.2% | **94.7%** | 🔥 |
| tracking_ang_vel | 73.7% | 70.0% | **80.0%** | 🔥 |
| swing_feet_z | 3.58 | 3.76 | **4.12** | 🔥 |
| ang_vel_xy | -0.67 | -1.13 | **-0.55** | 🔥🔥 |
| action_rate | -0.09 | -0.16 | **-0.04** | 🔥🔥 |
| action_smooth | -0.04 | -0.12 | **-0.02** | 🔥🔥 |
| action_std | 0.57 | 0.81 | **0.38** | ✅ |
| Episode | 191 | 143 | 162 | — |

### 与平地 R37 对比

| 指标 | 平地 R37 | 地形 R3 |
|------|:------:|:-----:|
| tracking_vx | 1.44 (96%) | 1.40 (93%) |
| tracking_vy | 1.68 (94%) | 1.71 (95%) |
| tracking_ang_vel | 1.44 (96%) | 1.44 (80%) |
| swing_feet_z | 3.32 | **4.12** 🔥 |
| ang_vel_xy | -0.267 | -0.547 |
| episode | **1000** | 162 |
| best reward | **182** | 94.6 |

## 关键教训

1. **地形参数要按机器人尺寸缩放，且宁小勿大**：OpenDoge 2.2kg/0.17m 站高 → 台阶 ≤3cm、斜坡 ≤10%、粗糙 ≤1.2cm。R1→R2 降难度后追踪明显改善。
2. **entropy 在 terrain 上不能过高**：R2 entropy=8e-3 导致 action_std 0.81、动作抖动 + episode 反降。5e-3 是已验证的安全值。
3. **地形 reward 的 scale 需要适应当前难度**：upward 1.0→1.5、ang_vel_xy -0.7→-1.0、tracking_ang_vel 1.5→1.8 在 R3 中全部正面生效。
4. **terrain curriculum 需要配合足够的 episode length**：R3 中 curriculum enabled 但无 promote（episode 太短走不够远）。更长的 episode 或更低的 promote 阈值才能使 curriculum 真正生效。
5. **reward 数值不可与平地对齐**：地形天然低压，「best~95 / final~45」是当前合理基线。不与平地 best=182 对比。
6. **所有三轮都在 iter ~166 早期见顶**：策略在早期找到局部最优后持续退化。可能原因：(a) 初始探索阶段的偶然高分被 mean reward 统计放大；(b) 策略后期在困难地形 cell 上学习时破坏了早期在平坦 cell 上的表现。curriculum 未能解决此问题。
7. **R3 viser 验证步态自然、全向追踪正常**：用户确认视觉效果满足需求。

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
