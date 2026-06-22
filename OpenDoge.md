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

### Round 13 (速度追踪优化, 1000 iters) — per-axis 独立追踪 + 零轴泄露抑制
```yaml
# 核心修改 1: tracking_lin_vel 拆分为 tracking_vx + tracking_vy（per-axis 独立问责）
# 核心修改 2: 新增 cross_axis_suppression 惩罚（零命令轴上的速度泄露）
# 核心修改 3: _sample_commands 混入 15% 纯单轴命令（增强训练暴露）
# 代码改动: rewards.py 新增 tracking_vx/tracking_vy/cross_axis_suppression
#           joystick.py 替换 dispatch（用共享 per-axis 函数）
#           mujoco.yaml tracking_vx:1.5, tracking_vy:1.5, cross_axis_suppression:-0.3
```
**动机**：当前 tracking_lin_vel 将 vx+vy 合并为一个标量误差，策略用 vx 好表现"掩盖"vy 漂移。拆分为独立 per-axis 奖励 + cross_axis_suppression 直接惩罚零轴泄露。同时训练中混入 15% 纯单轴命令增强暴露。
**结果**：best=**103.28**🔥🔥🔥🔥, final=**83.42**（best +8.77, final +7.97！）。tracking_vx=1.47, tracking_vy=1.43, tracking_ang_vel=0.157 (+43%), cross_axis_suppression=-0.012 (67%改善), ang_vel_xy=-0.078 (84%改善), episode=1000 全程不摔倒。

### Round 14 (vyaw 转向修复 + 域随机化, 1000 iters) — 步频 Bug 修复 + DR 鲁棒性
```yaml
# 核心修改 1: 步频计算包含 vyaw ([:,:2]→[:,:3]) — CRITICAL BUG FIX
#   纯转向时 cmd_speed 从 0→vyaw 值，步频从 1.2 Hz→最高 2.5 Hz
# 核心修改 2: tracking_ang_vel scale: 0.3→1.0（角速度梯度从 20%→67%）
# 核心修改 3: 新增 domain_rand（mass +-0.3kg, com +-2cm, push 0.3N/10s）
# 核心修改 4: 新增 noise_config.level: 1.0（传感器观测噪声）
# 核心修改 5: Viser vyaw 滑块 [-0.8,0.8]→[-1.5,1.5]
# ground_friction 暂闭（需 backend geom cache，后续实现）
```
**动机**：R13 转向基本不可用，诊断发现步频计算 `[:,:2]` 忽略 vyaw，纯转向时步频锁死 1.2 Hz。同时缺乏域随机化，策略对干扰敏感。
**结果**：best=**117.36**🔥🔥🔥🔥🔥, final=**94.25**（best +14.08, final +10.83！）。tracking_ang_vel=**0.952** (R13: 0.157, **+506%**🔥), tracking_vx=1.45, tracking_vy=1.35, cross_axis=-0.009 (25%改善), episode=1000 零摔倒。vyaw 转向从不可用到流畅稳定。

### Round 15 (零速后退漂移修复, 1500 iters) — actor linvel 观测 + cross_axis 增强
```yaml
# 核心修改 1: actor obs 新增 noisy linvel（49→52 维）
#   策略现在可以直接观察自身速度，零指令时可以感知并纠正漂移
#   NoiseConfig.scale_linvel=0.1（已有参数，此前未使用）
# 核心修改 2: cross_axis_suppression scale: -0.3→-0.6
#   零指令速度惩罚翻倍，强化策略对近零命令轴的速度抑制
# max_iterations: 1500（观测空间变化大，延长训练让策略充分学习 linvel 信息）
```
**动机**：Round 14 在零速指令下机器人轻微缓慢后退。诊断发现：(a) actor 观测不含 linvel，策略无法直接感知自身速度纠正漂移；(b) 默认 keyframe 后腿 thigh=0.7 前腿=0.5，不对称姿态产生后向 CoM 偏移；(c) cross_axis_suppression -0.3 对 0.1m/s 漂移仅贡献 0.03 惩罚，梯度太弱。修复选择：添加 linvel 到 actor obs 使策略获得直接速度反馈；增强 cross_axis 使零速漂移代价翻倍。不修改 keyframe 姿态以避免破坏已有行走性能。
**结果**：best=**116.50**, final=**97.42**（final +3.17！）。tracking_vx=1.47 (+0.02), tracking_vy=1.33, tracking_ang_vel=0.955, cross_axis=-0.0136 (per-unit leak 降低 ~24%), ang_vel_xy=-0.083, episode=1000 零摔倒。best 略低于 R14 (-0.86)，但 **final reward 大幅提升 (+3.17)**，收敛更稳定。zero-command drift 需 viser play 验证。

### Round 16 (指令对称化 + quadratic cross_axis, 1500 iters) — vx 范围对称 + L2 惩罚
```yaml
# 核心修改 1: vx 指令范围 [-0.6,1.0]→[-0.8,0.8]（前后完全对称）
#   消除训练分布的前向偏置（62%/38%→50%/50%），vy 追踪应追平 vx
# 核心修改 2: cross_axis_suppression |v|→v²（二次惩罚）
#   近零宽容（自然微抖不被过度压制）、远零重罚（大漂移非线性增长）
#   新增 rewards.cross_axis_suppression_l2 函数，保留原 L1 版本不动
# 核心修改 3: cross_axis_suppression scale: -0.6→-6.0
#   二次值比线性小 ~10x（|v|<1 时 v² ≪ |v|），scale 同比放大以匹配惩罚力度
```
**动机**：R15 修复了观测盲区但 tracking_vy (1.33) 仍低于 tracking_vx (1.47)，零速漂移方向偏后。根因：(a) vx 指令不对称导致前向经验远多于后向；(b) 线性 cross_axis 在近零处梯度恒定，对微小自然抖动过度惩罚，抑制了策略探索更自然的零速站立姿态。改为对称范围 + 二次惩罚让策略在零指令时"放心站直"而非"紧张消除每一丝速度"。
**结果**：best=**118.98** (+2.48), final=**90.66** (**-6.76** ❌)。tracking_vx=1.34 (-0.13), tracking_vy=1.24 (-0.09), cross_axis=-0.011, stand_still=-0.023。**失败**：L2 cross_axis (-6.0) 梯度过强 (12|v| vs L1 的 0.6)，策略过度压制寄生速度反而牺牲主轴追踪。vx 对称化保留，cross_axis 回退 L1。

### Round 17 (R16 修复, 1500 iters) — 保留对称 vx + 回退 cross_axis L1
```yaml
# 核心修改 1: vx 指令范围 [-0.8,0.8]（保留 R16 的对称化）
# 核心修改 2: cross_axis_suppression L2→L1，scale -6.0→-0.6（回退 R15 验证过的配置）
#   教训：L2 在 v>0.05 时梯度即超过 L1，对策略的"压迫感"太强
```
**动机**：R16 的 vx 对称化是正确方向，但 L2 cross_axis 过强导致速度追踪退步。保留对称范围 + 回退到已验证的 L1 惩罚。
**结果**：best=**118.89**, final=**97.55**。tracking_vx=1.43, tracking_vy=1.34, tracking_ang_vel=0.949, cross_axis=-0.016, episode=1000 零摔倒。vx-vy 差距从 0.14 (R15) 缩至 0.09 (**-36%**)，对称化有效但未完全消除差距。best 略高于 R15 (+2.39)，final 持平。

### Round 18 (零指令强制静止, 1500 iters) — zero_command_stillness 专用奖励
```yaml
# 核心修改: 新增 zero_command_stillness 奖励（scale=3.0, σ=0.01）
#   仅当 command_mag < 0.05 时激活，用极紧指数 exp(-|v|²/0.01) 奖励零速度
#   梯度是 tracking_vx 的 ~15 倍：0.05m/s 漂移 → tracking 丢 0.015, stillness 丢 0.66
```
**动机**：R17 零指令下仍有微动。根因是 tracking_vx/vy 的 σ=0.25 在 v→0 时梯度近零（v=0.05 仅差 0.015），驱动策略到完美静止的力太弱。新增极紧 σ 专用奖励彻底解决。
**结果**：best=**138.80** 🔥🔥🔥🔥🔥🔥, final=**101.02** 🔥（best **+19.91**, final **+3.47**，双双新纪录！）。tracking_vx=1.45, tracking_vy=1.25, tracking_ang_vel=0.949, cross_axis=-0.012, ang_vel_xy=-0.073, zero_command_stillness=0.337 (+12x 从 0.028), episode=1000 零摔倒。静止奖励不仅修复了零指令漂移，更整体提升了策略质量——**首次突破 best 130+ 和 final 100+**。

### Round 19 (踱步抑制 + 转向增强, 1500 iters) — stand_still 增强 + tracking_ang_vel 提权
```yaml
# 核心修改 1: stand_still scale: -0.2→-0.5（2.5x，零指令关节更贴近默认姿态）
#   零指令踱步根因：zero_command_stillness 只约束身体速度，不约束关节运动
#   增强后 per-unit 关节偏差降低 ~16%
# 核心修改 2: tracking_ang_vel scale: 1.0→1.5（50% 增强）
#   左右转向不对称根因：无机械/配置不对称，策略收敛到局部最优
#   增强梯度迫使两个方向都达到高精度
```
**动机**：R18 零指令仍有踱步 + 左右转向不对称。踱步因 stand_still 太弱无法抑制关节 fidgeting。转向不对称经全面诊断（XML、keyframe、vel_limit、reward）无任何机械根源，纯策略局部最优。增强 stand_still 抑制踱步 + 增强 yaw 追踪梯度消解不对称。
**结果**：best=**145.44** 🔥🔥🔥🔥🔥🔥🔥, final=**110.21** 🔥（best **+6.64**, final **+9.19**，双双新纪录！）。tracking_vx=1.42, tracking_vy=1.27, tracking_ang_vel=1.437/1.5 (norm 0.958, +0.009), stand_still=-0.042 (偏差↓16%), zero_command_stillness=0.338, episode=1000 零摔倒。

### 当前最优配置 (Round 19)
# conf/ppo/task/opendoge_joystick_flat/mujoco.yaml
algo:
  num_envs: 1024
  max_iterations: 1000
  empirical_normalization: true
  policy:
    init_noise_std: 0.5
  algorithm:
    learning_rate: 3.0e-4
    entropy_coef: 3.0e-3
env:
  commands:
    rel_standing_envs: 0.1
    pure_axis_prob: 0.15
    vel_limit:
      - [-0.8, -0.6, -1.5]
      - [0.8, 0.6, 1.5]
  noise_config:
    level: 1.0
  domain_rand:
    randomize_base_mass: true
    added_mass_range: [-0.3, 0.3]
    random_com: true
    com_offset_x: [-0.02, 0.02]
    push_robots: true
    push_interval: 500
    max_force: [0.3, 0.3, 0.2]
reward:
  scales:
    tracking_vx: 1.5
    tracking_vy: 1.5
    tracking_ang_vel: 1.5
    cross_axis_suppression: -0.6
    lin_vel_z: -5.0
    ang_vel_xy: -0.5
    base_height: -200.0
    action_rate: -0.008
    action_smooth: -0.003
    similar_to_default: -0.03
    dof_acc: -0.0000005
    stand_still: -0.5
    zero_command_stillness: 3.0
    torques: -0.005
    energy: -0.0001
    contact: 0.24
    swing_feet_z: 5.0
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
| **R13** | **103.28** 🔥🔥🔥🔥 | **83.42** | per-axis 追踪 + 零轴抑制 + 纯轴采样 |
| **R14** | **117.36** 🔥🔥🔥🔥🔥 | **94.25** | vyaw 步频 Bug 修复 + DR + 观测噪声 |
| **R15** | **116.50** | **97.42** 🔥 | actor linvel 观测 + cross_axis 增强 (-0.3→-0.6) |
| **R16** | **118.98** | 90.66 ❌ | vx 对称化 + quadratic cross_axis (L2 过强，失败) |
| **R17** | **118.89** | **97.55** | 保留对称 vx + 回退 L1 cross_axis (R16 修复) |
| **R18** | **138.80** 🔥🔥🔥🔥🔥🔥 | **101.02** 🔥 | zero_command_stillness (σ=0.01, +19.91 best!) |
| **R19** | **145.44** 🔥🔥🔥🔥🔥🔥🔥 | **110.21** 🔥 | stand_still ↑2.5x + tracking_ang_vel ↑1.5x |

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
- ✅ **per-axis 独立速度追踪**（vx/vy/vyaw 各自独立问责）
- ✅ **cross_axis_suppression**（零命令轴速度泄露抑制）
- ✅ **vyaw 步频自适应**（纯转向时步频 1.2→2.5 Hz）
- ✅ **域随机化**（质量/质心/推搡扰动）
- ✅ **观测噪声**（传感器噪声鲁棒）
- ✅ **actor linvel 观测**（策略可直接感知速度，零指令漂移自纠正）

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
11. **Per-axis 速度追踪远优于合并追踪**：tracking_lin_vel 将 vx+vy 合并为一个标量误差，允许策略用 vx 好表现掩盖 vy 漂移。拆分为独立的 tracking_vx + tracking_vy 后每轴独立问责，速度追踪精度大幅提升（tracking_vx: 0.87→1.47）。
12. **cross_axis_suppression 是速度追踪的"刹车"**：当命令只有 vx 时直接惩罚 vy/vyaw 泄露，从源头抑制不期望运动。训练中从 -0.03 降至 -0.012（67% 改善），ang_vel_xy 从 -0.5 降至 -0.078（84% 改善）。
13. **步频必须包含 vyaw — 否则纯转向无异于站立**：`cmd_speed = norm(commands[:,:2])` 忽略 vyaw，纯转向时步频锁死在 1.2 Hz，机器人几乎不迈步。改为 `[:,:3]` 后步频响应 vyaw 指令，tracking_ang_vel 从 0.157→0.952（+506%），转向从不可用到流畅稳定。
14. **轻量机器人的域随机化要按质量比缩放**：OpenDoge 2.2kg vs Go2 12kg（5.5x），mass +-0.3kg（14%）、push 0.3N（Go2 1.0N 按 5.5x 缩放）、com +-2cm（Go2 5cm 减半）。太强的 DR 会直接击倒小机器人。
15. **零速漂移的根因通常是观测盲区而非奖励不对称**：当奖励函数在零指令时对称（tracking_vx 用指数型、cross_axis_suppression 用绝对值），漂移方向取决于默认姿态的不对称性，但策略无法纠正漂移的本质原因是 actor 缺少 linvel 观测——它"看不见"自己在漂移。添加 noisy linvel 到 actor obs 让策略获得直接速度反馈，零指令时可主动抑制漂移。这是比单纯增强惩罚项更根本的修复。
16. **二次惩罚 (L2) 在 cross_axis 场景下实证失败**：理论上 L2 对小漂移宽容、对大漂移严惩更优，但实践中 L2 梯度 `2|v|` 在 |v|>0.05 即超过 L1 的恒定梯度 1.0，配合 scale -6.0 后梯度放大到 `12|v|`（L1 的 `0.6`），策略被迫过度关注寄生速度抑制而牺牲主轴追踪。**L1 对零轴速度抑制已足够有效**。

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
