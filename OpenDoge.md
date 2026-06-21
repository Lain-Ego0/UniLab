# OpenDoge 四足机器人 UniLab 训练手册

轻量四足（~2.2kg base，12 DOF，4 腿 FL/FR/RL/RR，每腿 hip+thigh+calf）。
源 URDF: `/home/lain/OpenDoge/OpenDoge_train/resources/robots/Opendoge/`

## 资产位置

| 类型 | 路径 |
|------|------|
| Robot XML | `src/unilab/assets/robots/opendoge/opendoge.xml` |
| Scene XML | `src/unilab/assets/robots/opendoge/scene_flat.xml` |
| Meshes | `src/unilab/assets/robots/opendoge/assets/`（STL） |
| Env 包 | `src/unilab/envs/locomotion/opendoge/` |
| PPO 配置 | `conf/ppo/task/opendoge_joystick_flat/mujoco.yaml` |
| 测试 | `tests/test_opendoge.py` |

## 物理约束

| 属性 | 值 |
|------|-----|
| Base mass | 2.2 kg |
| DOF | 12（4腿 × 3关节） |
| Hip 范围 | FL/RL: -0.785~0.26, FR/RR: -0.26~0.785 |
| Thigh 范围 | -0.785 ~ 1.134 |
| Calf 范围 | **-2.68 ~ -1.04**（永远弯折 ≥60°） |
| 最大站立高度 | **~0.17m**（受 calf 关节限制） |
| PD 增益 | Kp=20, Kd=0.3（轻量机器人降低增益） |
| 电机额定扭矩 | **1.8 Nm**（连续），峰值 6 Nm |
| 热限 | >2Nm=300s, 3Nm=44s, 4Nm=14s, 5Nm=7s, 6Nm=5s |

**关键**：`base_height_target` 不可高于 0.17，否则策略被持续惩罚无法达成的目标。
新机器人验证方法：
```bash
uv run python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('src/unilab/assets/robots/opendoge/scene_flat.xml')
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
for _ in range(50): mujoco.mj_step(m, d)
print(f'Steady-state base z: {d.qpos[2]:.4f}')
"
```

## 奖励调参历史

### Round 1 (初始, 150 iters) — 照搬 Go2 配置
```yaml
base_height_target: 0.25     # 物理不可达！
tracking_lin_vel: 1.0
ang_vel_xy: -0.1
action_rate: -0.005
similar_to_default: -0.1
keyframe: base z=0.20, thigh=0.8/1.0, calf=-1.5
```
**结果**：reward=24.1, episode=980, base_height=-0.41（被持续惩罚），姿态深蹲匍匐。

### Round 2 (扭振修复, 500 iters) — 加大惩罚抑制扭转
```yaml
base_height_target: 0.25     # 仍未修复根因
ang_vel_xy: -0.5             # 从 -0.1 加大
base_height: -200            # 从 -100 加大
action_rate: -0.01           # 从 -0.005 加大
similar_to_default: -0.05    # 从 -0.1 减小
keyframe: 不变
```
**结果**：episode=1000（从不摔倒），但 reward=22.1（反而下降），base_height=-0.55（更差），姿态猥琐扭动。

### Round 3 (姿态修复, 500 iters) — 修复 keyframe + 可达高度目标
```yaml
base_height_target: 0.15     # 核心修复：物理可达
tracking_lin_vel: 1.5        # 从 1.0 提高，增强速度跟踪抑制侧摆
action_rate: -0.02           # 从 -0.01 加大
similar_to_default: -0.03    # 从 -0.05 减小
keyframe: base z=0.17, thigh=0.5/0.7, calf=-1.3
```
**结果**：reward=**47.6**（翻倍），base_height=**-0.03**（改善 18x），tracking_lin_vel=**1.27**（改善 2.5x），action_std=**0.07**。

### Round 4 (平稳优雅, 500 iters) — jerk 平滑 + 高抬腿 + 转向增强
```yaml
tracking_ang_vel: 0.3        # 从 0.2 提高（转向更灵敏）
lin_vel_z: -10.0             # 从 -5.0 加大（抑制垂直弹跳）
action_rate: -0.01           # 从 -0.02 减小（让位给 action_smooth）
action_smooth: -0.005        # NEW: 二阶动作平滑（jerk 惩罚）
dof_acc: -5.0e-7             # NEW: 关节加速度惩罚（qacc 未计算，无效）
swing_feet_z: 6.0            # 从 4.0 提高（更高抬腿，步态更优雅）
```
**结果**：reward=**67.1**（+41%），best_reward=**72.9**，tracking_ang_vel=**0.25**（+107%），swing_feet_z=**1.66**（+80%），action_smooth=-0.013（jerk 平滑生效）。

### Round 5 (去生硬 + 全向增强, 800 iters) — 提高熵 + 扩展指令范围
```yaml
entropy_coef: 3.0e-3          # 从 1e-3 提高（减少过拟合，软化步态）
action_smooth: -0.003         # 从 -0.005 减小（减少 jerk 惩罚 = 更自然）
action_rate: -0.008           # 从 -0.01 减小
lin_vel_z: -5.0               # 从 -10 减小（不过度惩罚弹跳 = 步态更舒展）
stand_still: -0.2             # NEW: 停止时保持姿态稳定
vel_limit vy: [-0.6, 0.6]     # 从 [-0.4, 0.4] 扩展（更强侧移训练）
vel_limit vyaw: [-1.5, 1.5]   # 从 [-0.8, 0.8] 扩展（更强转向训练）
max_iterations: 800           # 从 500 延长
```
**结果**：best_reward=**74.8**（新纪录），action_std=**0.084**（+15%，步态更自然舒展），swing_feet_z=**1.80**。entropy 从 -14.9→-13.2（减少过拟合）。

### Round 5 (去生硬, 800 iters) — 提高熵 + 扩展指令范围
```yaml
entropy_coef: 3.0e-3          # 从 1e-3 提高
action_smooth: -0.003         # 从 -0.005 减小
lin_vel_z: -5.0               # 从 -10 减小
stand_still: -0.2             # NEW
vel_limit vy: [-0.6, 0.6]     # 扩展
vel_limit vyaw: [-1.5, 1.5]   # 扩展
max_iterations: 800           # 延长
```
**结果**：best=**74.8**，action_std=**0.084**（+15%，去生硬生效）。

### Round 6 (节能, 1000 iters) — 扭矩约束 + 自适应步频
```yaml
torques: -0.005              # NEW: PD扭矩L1惩罚（目标<1.8Nm）
energy: -0.0001              # NEW: 机械功率约束
max_iterations: 1000
# 代码新增: PD扭矩估算 + 自适应步频(1.2→2.5Hz随速度)
```
**结果**：best=**76.2**，torques≈1.6Nm/关节（低于额定1.8Nm），swing_feet_z=2.13。

### Round 7 (步高自适应尝试, 1000 iters) — 权重太低崩溃
```yaml
swing_feet_z: 3.0            # 从 6.0 骤降
# 步骤目标: 0.02→0.06m（太保守）
```
**结果**：best=**57.9**，崩了。

### Round 7a (步高恢复, 1000 iters) — 调整权重范围
```yaml
swing_feet_z: 5.0            # 从 3.0 回升
# 步骤目标: 0.03→0.08m
```
**结果**：best=**72.9**，恢复。

### Round 8 (步高含vyaw, 1000 iters) — 3D指令范数驱动步高 🔥
```yaml
# 代码关键: cmd_mag = norm([vx, vy, vyaw])  # 3D范数含旋转
# 步骤目标: 0.04 + 0.08 * cmd_mag → [0.04, 0.12]
```
**结果**：best=**88.7**, final=**75.1**（新纪录！+16%）。

### Round 12 (零速指令注入, 1000 iters) — DR provider 注入 10% 站立环境
```yaml
# 核心修改: _sample_commands 注入 10% 零速指令 + 清零微小指令
# env.commands.rel_standing_envs: 0.1
# zero_small: cmd_mag < 0.03 → 0
```
**动机**：R11 奖励门控正确但训练时 cmd_mag<0.03 概率仅 0.002%，门控永不激活。现在 10% 环境持续收到零指令，策略必须学会四脚贴地站立。
**结果**：best=**94.51**🔥🔥🔥, final=**75.45**（best 暴涨 +4.0！）。10%零速指令使策略学会站立，步态整体提升。

### 当前最优配置 (Round 12)
# conf/ppo/task/opendoge_joystick_flat/mujoco.yaml
algo:
  num_envs: 1024
  max_iterations: 800
  empirical_normalization: true
  policy:
    init_noise_std: 0.5
  algorithm:
    learning_rate: 3.0e-4
    entropy_coef: 3.0e-3
env:
  commands:
    vel_limit:
      - [-0.6, -0.6, -1.5]
      - [1.0, 0.6, 1.5]
reward:
  scales:
    tracking_lin_vel: 1.5
    tracking_ang_vel: 0.3
    lin_vel_z: -5.0
    ang_vel_xy: -0.5
    base_height: -200.0
    action_rate: -0.008
    action_smooth: -0.003
    similar_to_default: -0.03
    dof_acc: -0.0000005
    stand_still: -0.2
    contact: 0.24
    swing_feet_z: 6.0
  tracking_sigma: 0.25
  base_height_target: 0.15
```

## 训练成果总览

| Round | Best Reward | Final | 关键改动 |
|:-----:|:-----------:|:-----:|:---------|
| R1 | 33.2 | 24.1 | 照搬Go2，深蹲匍匐 |
| R2 | 32.7 | 22.1 | 扭振修复，reward反降 |
| R3 | 54.0 | 47.6 | **keyframe修复**，翻倍 |
| R4 | 72.9 | 67.1 | jerk平滑+高抬腿 |
| R5 | 74.8 | 66.6 | 熵提高，去生硬 |
| R6 | 76.2 | 69.8 | 扭矩约束+自适应步频 |
| R7 | 57.9 | 50.3 | 步高权重太低（崩） |
| R7a | 72.9 | 64.5 | 步高恢复 |
| **R8** | **88.7** | **75.1** | 🔥 步高含vyaw 3D范数 |
| **R9** | **91.82** 🔥 | **79.38** | 降低最低步高 4cm→2cm |
| **R10** | **90.57** | **79.97** | 零速贴地步高 (=脚球半径) |
| **R11** | **90.56** | **80.25** 🔥 | 零速 gait 坍缩 (swing→0 + contact全stance) |
| **R12** | **94.51** 🔥🔥 | **75.45** | 零速指令注入 10% standing envs |

### 最终已实现功能
- ✅ 扭矩 ~1.6 Nm/关节（<1.8 额定，安全持续运行）
- ✅ 能量（功率）约束
- ✅ 步频自适应：1.2→2.5 Hz（速度越快步频越高）
- ✅ 步高自适应：0.015→0.12m（零速贴地、高速有力，线性过渡）
- ✅ 步高关联 vyaw（旋转时也抬腿）
- ✅ 动作平滑（action_smooth jerk惩罚）
- ✅ 物理可达站高（base_height_target=0.15）
- ✅ viser GUI 实时方向控制滑块
- ✅ viser 完整 mesh 渲染

## 关键教训

1. **keyframe 是训练质量的第一因**：default_angles 决定策略的"中立姿态"，错误 keyframe 导致策略始终对抗物理约束
2. **base_height_target 必须经验验证可达性**：先加载 scene settle 50 步，测稳态高度，target 设在 ±10% 范围
3. **新机器人先调 keyframe 再调 reward**：姿态问题先怀疑 keyframe，不要急于加惩罚项
4. **平滑性靠 jerk 惩罚而非纯 action_rate**：`action_smooth`（二阶）比 `action_rate`（一阶）更有效抑制抖动，两者配合使用
5. **足端抬升对步态美观度影响巨大**：`swing_feet_z` 从 4.0→6.0 让抬腿从"拖地"变"踏步"
6. **步态生硬 = 过拟合**：action_std < 0.08 通常意味着策略过于确定性。提高 `entropy_coef`（1e-3→3e-3），降低 `action_smooth`、`action_rate` 惩罚权重可恢复自然度
7. **扭矩约束需自行估算**：MuJoCo 后端不自动提供 `torques`。用 PD 公式 `tau = Kp*(target-q) - Kd*qdot` 在 `update_state` 中估算并写入 `info["torques"]`，然后在 `RewardContext` 中传 `dof_vel` 以支持 `energy` 奖励
8. **步高目标用 3D 指令范数**：仅 vx+vy 范数会忽略旋转指令（vyaw），导致转弯时拖地。用 `norm([vx,vy,vyaw])` 驱动 `target_height`（0.02→0.12m），旋转越大抬腿越高。最低步高需匹配腿长：腿长 0.2m → 最低 0.02m（10% 腿长），过高则低速步态笨重
9. **奖励权重调参法则**：每轮只改 1-2 项，保持其它不变；若 reward 骤降 >20% 即为失败，回滚后微调
10. 每次修改 reward 需在本文件中追加记录轮次、数值、效果

## viser 可视化修复

viser 默认只显示碰撞体（后端用 `create_discardvisual_xml` 剥离 visual mesh）。
新增 `get_playback_visual_model()` 方法链：

| 文件 | 修改 |
|------|------|
| `src/unilab/base/backend/base.py` | SimBackend 抽象方法（默认 fallback 到 get_playback_model） |
| `src/unilab/base/backend/mujoco/backend.py` | MuJoCo 实现（加载 scene_visual_model_file 并缓存） |
| `src/unilab/base/np_env.py` | env 层透传 |
| `scripts/play_viser.py` | `_load_env_playback_model` 优先调用 visual model |

## 工作流

### 环境准备

```bash
# 安装依赖（MuJoCo + viser）
uv sync --extra mujoco --extra viser
```

### 训练

```bash
# 标准训练（1000 轮，1024 envs，CUDA）
uv run train --algo ppo --task opendoge_joystick_flat --sim mujoco

# 快速验证（20 轮，256 envs）
uv run train --algo ppo --task opendoge_joystick_flat --sim mujoco \
  algo.num_envs=256 algo.max_iterations=20

# 覆盖参数（不改配置文件）
uv run train --algo ppo --task opendoge_joystick_flat --sim mujoco \
  algo.max_iterations=2000 reward.scales.torques=-0.01
```

### 评估（渲染录像）

```bash
# 加载最新模型自动评估
uv run eval --algo ppo --task opendoge_joystick_flat --sim mujoco --load-run -1
```

### Web 可视化（viser，浏览器操控方向）

```bash
# 启动 viser（默认 8080 端口，GUI 滑块实时调方向）
uv run python scripts/play_viser.py task=opendoge_joystick_flat/mujoco \
  interactive.action_mode=policy viser.port=8080
```

打开 `http://localhost:8080`，右侧面板：
- **Controls**: 暂停 / 调速 / 切视角
- **Command**: `vx`/`vy`/`vyaw` 滑块实时调方向，`Override command` 勾选框切换手动/随机

### 交互式 MuJoCo 原生 viewer（需要显示器）

```bash
uv run python scripts/play_interactive.py \
  --algo ppo --task opendoge_joystick_flat --sim mujoco +load_run=-1
```

### 测试与检查

```bash
uv run pytest tests/test_opendoge.py -xvs   # 单元测试（5 项）
make check                                    # lint + type check
tensorboard --logdir logs/rsl_rl_ppo/OpenDogeJoystickFlat/  # 训练曲线
```

### 日志位置

```
logs/rsl_rl_ppo/OpenDogeJoystickFlat/<run>/
├── model_499.pt       # checkpoint
├── policy.onnx         # ONNX 导出
├── policy.pt           # TorchScript
├── play_video.mp4      # 训练录像
├── run_summary.json    # 训练摘要
└── events.out.*        # TensorBoard
```
