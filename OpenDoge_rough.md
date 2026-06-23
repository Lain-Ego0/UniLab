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
| 测试 | `tests/test_opendoge.py` |

## 地形配置

地形由 `OpenDogeRoughTerrainCfg` 定义，按 2.2 kg 小狗保守缩放：

| 地形类型 | 比例 | 参数 |
|----------|:----:|------|
| flat | 20% | 平地（保证基础行走不退化） |
| pyramid_stairs | 8% | 台阶 1-4 cm, step_width=0.4m |
| pyramid_stairs_inv | 8% | 倒台阶 1-4 cm |
| hf_pyramid_slope | 15% | 斜坡 0-15%（~8.5°） |
| hf_pyramid_slope_inv | 15% | 倒斜坡 0-15% |
| random_rough | 20% | 粗糙地形 0.3-1.5 cm |
| wave_terrain | 14% | 波浪地形 0-4 cm |

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
| upward | 1.0 | 身体朝向重力反方向的奖励（站直） |

## 训练成果

### Round 1 (地形首训, 1500 iters) — 保守地形参数 + R37 reward

```yaml
# 地形: 台阶 1-4cm / 斜坡 0-15% / 粗糙 0.3-1.5cm / 波浪 0-4cm / 平地 20%
# Reward: 继承 R37 平地最优配置 + 7 个地形专有 reward
algo.num_envs: 2048
algo.max_iterations: 1500
```

**结果**：best=**61.45** (iter 166), final=**37.91**, episode=**191**, FPS=70k。

| Reward | 值 | Scale | 归一化 |
|--------|-----|:---:|:---:|
| tracking_vx | 1.361 | 1.5 | **90.7%** |
| tracking_vy | 1.643 | 1.8 | **91.3%** |
| tracking_ang_vel | 1.105 | 1.5 | 73.7% |
| tracking_vel_linear | -0.094 | -0.4 | 76.4% |
| swing_feet_z | 3.581 | 12.0 | 29.8% |
| feet_gait | 0.450 | 0.5 | **89.9%** |
| upward | 3.966 | 4.0 | 99.1% |
| lin_vel_z | -0.075 | -4.0 | 98.1% |
| base_height | -0.024 | — | 精准 |
| ang_vel_xy | -0.672 | -0.7 | ⚠️ 地形摇摆 |
| torques | -0.169 | -0.005 | ⚠️ 地形扭矩高 |
| undesired_contacts | -0.003 | -1.0 | 几乎无碰撞 |

### 与平地 R37 对比

| 指标 | 平地 R37 | 地形 R1 |
|------|:------:|:-----:|
| tracking_vx | 1.44 (96%) | 1.36 (91%) |
| tracking_vy | 1.68 (94%) | 1.64 (91%) |
| swing_feet_z | 3.32 | **3.58** |
| episode | **1000** | 191 |
| best reward | **182** | 61.5 |

**关键发现**：
1. **速度追踪几乎追平平地** — vx 91%, vy 91%，地形对追踪能力影响可控
2. **地形上 swing 反而更高** — 3.58 vs 3.32，需要跨障碍
3. **episode 大幅缩短** — 1000→191，复杂地形摔倒概率高 5x
4. **reward 数值不可与平地直接对比** — 地形天然更低压，「best 30+ / final 100+」才是目标
5. **身体摇摆显著恶化** — ang_vel_xy -0.672 vs 平地 -0.267，地形必然代价

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
├── model_1499.pt      # checkpoint
├── policy.onnx         # ONNX 导出
├── policy.pt           # TorchScript
├── play_video.mp4      # 训练录像
├── run_summary.json    # 训练摘要
└── events.out.*        # TensorBoard
```
