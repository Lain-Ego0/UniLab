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

### 当前最优配置 (Round 5)
```yaml
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

## 关键教训

1. **keyframe 是训练质量的第一因**：default_angles 决定策略的"中立姿态"，错误 keyframe 导致策略始终对抗物理约束
2. **base_height_target 必须经验验证可达性**：先加载 scene settle 50 步，测稳态高度，target 设在 ±10% 范围
3. **新机器人先调 keyframe 再调 reward**：姿态问题先怀疑 keyframe，不要急于加惩罚项
4. **平滑性靠 jerk 惩罚而非纯 action_rate**：`action_smooth`（二阶）比 `action_rate`（一阶）更有效抑制抖动，两者配合使用
5. **足端抬升对步态美观度影响巨大**：`swing_feet_z` 从 4.0→6.0 让抬腿从"拖地"变"踏步"
6. **步态生硬 = 过拟合**：action_std < 0.08 通常意味着策略过于确定性。提高 `entropy_coef`（1e-3→3e-3），降低 `action_smooth`、`action_rate` 惩罚权重可恢复自然度
7. 每次修改 reward 需在本文件中追加记录轮次、数值、效果

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
# 标准训练（800 轮，1024 envs，CUDA）
uv run train --algo ppo --task opendoge_joystick_flat --sim mujoco

# 快速验证（20 轮，256 envs）
uv run train --algo ppo --task opendoge_joystick_flat --sim mujoco \
  algo.num_envs=256 algo.max_iterations=20
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

## 工作流
