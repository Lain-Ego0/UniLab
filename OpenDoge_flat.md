# OpenDoge 四足机器人 UniLab 训练手册

轻量四足（总重 ~4.80 kg，base 2.24 kg，12 DOF，4 腿 FL/FR/RL/RR，每腿 hip+thigh+calf）。
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

## 机器人几何与质量参数

### 总质量与分布

| 部件 | 单件质量 (kg) | ×数量 | 小计 (kg) | 占比 |
|------|:---------:|:----:|:------:|:----:|
| Base (机身) | 2.235 | 1 | 2.235 | **46.6%** |
| Hip (髋关节连杆) | 0.273 | 4 | 1.092 | 22.8% |
| Thigh (大腿) | 0.344 | 4 | 1.376 | 28.7% |
| Calf (小腿) | 0.023 | 4 | 0.093 | 1.9% |
| **总计** | — | — | **4.796** | 100% |

> **注意**：单条腿质量 = 0.273 + 0.344 + 0.023 = **0.640 kg**（占总量 13.3%）。
> 四条腿合计 **2.561 kg（53.4%）**，超过机身质量。小腿极轻（23g）——电机集中在大腿/髋部，小腿仅为结构件。

### 机身尺寸

| 属性 | 值 | 说明 |
|------|-----|------|
| 碰撞盒 | 0.17 × 0.09 × 0.10 m | 基础几何（长×宽×高） |
| Base CoM 偏移 | [+0.13, +0.001, +0.77] mm | 几乎在几何中心 |
| Base 惯量 (Ixx/Iyy/Izz) | 0.00542 / 0.00992 / 0.01345 kg·m² | fullinertia 对角元 |

### 腿段长度（运动学链）

以 FL 腿为例（四腿对称，仅符号差异）：

```
Base中心 ──[X: 0.1425, Y: 0.040]──▶ Hip关节(abduction, X轴)
                                      │
                                      │ Y: 0.08615 m (髋外展连杆)
                                      ▼
                                  Thigh关节(pitch, Y轴)
                                      │
                                      │ Z: 0.100 m (大腿长)
                                      ▼
                                  Calf关节(pitch, Y轴)
                                      │
                                      │ Z: 0.0994 m (小腿长)
                                      ▼
                                  Foot球 (r=0.015 m, 触地)
```

| 段 | 长度 (m) | 方向 | 说明 |
|----|:------:|------|------|
| 髋外展连杆 (hip abduction) | 0.086 | Y (侧向) | 从髋关节到 thigh 关节的侧向偏移 |
| 大腿 (thigh) | 0.100 | Z (竖直) | thigh 关节 → calf 关节 |
| 小腿 (calf) | 0.099 | Z (竖直) | calf 关节 → foot 球心 |
| 脚球半径 | 0.015 | — | 触地碰撞球 |
| **最大腿伸直长度 (Z)** | **~0.214** | — | 0.100 + 0.099 + 0.015 (thigh+calf 完全竖直) |

### 髋关节间距

| 方向 | 间距 (m) | 说明 |
|------|:------:|------|
| 前后 (X) | **0.285** | 前髋 FL↔RL、后髋 FR↔RR 距离（= 2 × 0.1425） |
| 左右 (Y) | **0.080** | 左腿 FL↔FR、右腿 RL↔RR 距离（= 2 × 0.040） |

> 前后间距 0.285m 远大于左右 0.080m → 机器人呈**狭长矩形**站姿，侧向稳定性天然弱于前后稳定性。

### 腿段质量分布（单腿）

| 段 | 质量 (kg) | 占单腿比 | 占总量比 |
|----|:------:|:-----:|:-----:|
| Hip | 0.273 | 42.7% | 5.7% |
| Thigh | 0.344 | 53.7% | 7.2% |
| Calf | 0.023 | 3.6% | 0.5% |

> 大腿是最重的单腿段（344g），旋转惯量主要集中在近端关节。小腿几乎无质量——**腿的动力学由大腿主导**。

### 关节范围

| 关节 | 前腿 (FL/FR) | 后腿 (RL/RR) | 轴 |
|------|:-----------:|:-----------:|:--:|
| Hip (abduction) | FL: [-0.785, 0.26], FR: [-0.26, 0.785] | RL: [-0.785, 0.26], RR: [-0.26, 0.785] | X |
| Thigh (pitch) | [-0.785, 1.134] | [-0.785, 1.134] | Y |
| Calf (pitch) | **[-2.68, -1.04]** | **[-2.68, -1.04]** | Y |

> Calf 关节范围 **全为负**（永远弯折 ≥60° = 1.04 rad），膝关节不能伸直——这是此机器人的关键运动学约束。

### 驱动器参数

| 属性 | 值 |
|------|-----|
| PD 增益 | Kp=**20**, Kd=**0.3** |
| Hip/Thigh 力限 | ±6 Nm |
| Calf 力限 | ±9 Nm |
| 电机额定扭矩 | **1.8 Nm**（连续），峰值 6 Nm |
| 关节阻尼 | 0.5 Nm·s/rad |
| 关节摩擦 | 0.2 Nm |
| 电枢惯量 | 0.01 kg·m² |

### 关键物理特性总结（AI 决策参考）

| 特征 | 数值 | 含义 |
|------|------|------|
| 腿身质量比 | 53% : 47% | 过半质量在腿上，腿运动显著影响机身姿态 |
| 大腿/小腿质量比 | 15:1 | 小腿近乎无质量，摆腿动力学 = 大腿旋转 |
| 长宽比 | 3.6:1 | 极狭长机身，侧向稳定弱 → vy 追踪天生难于 vx |
| Calf 弯曲约束 | ≥60° | 膝永不全直，站立高度 ≤0.17m，迈步=膝盖折叠更深 |
| 脚球半径 | 15 mm | 小接触点，对地面摩擦/坡度敏感 |
| 质心高度 (站立) | ~0.10-0.12 m | 腿长 0.2m 的 50-60%，稳定裕度适中 |

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

### Round 20 (lin_vel_z + action_smooth 翻倍, 1000 iters) — 纵向震荡过度压制
```yaml
lin_vel_z: -10.0             # 从 -5.0 翻倍（强压纵向震荡）
action_smooth: -0.005        # 从 -0.003 增强（67% jerk 惩罚）
```
**动机**：进一步压制弹跳和抖动。
**结果**：best=**145.35**, final=**113.23** 🔥。vy 追平 vx（对称化完全生效），但步高过小、步频过快——`lin_vel_z:-10.0` 过强压制纵向震荡，策略选择极小碎步保持身体平板滑行，步态不自然。

### Round 21 (释放纵向震荡, 1000 iters) — lin_vel_z 回退 + action_smooth 微降
```yaml
lin_vel_z: -4.0              # 从 -10.0 回退（低于 R19 的 -5.0，进一步释放自然步态震荡）
action_smooth: -0.004        # 从 -0.005 微降（允许更动态的迈步动作）
```
**动机**：R20 的 `lin_vel_z:-10.0` 与 `swing_feet_z:5.0` 冲突——抬腿需要身体竖向运动，竖向运动却被 `lin_vel_z` 强惩罚。`lin_vel_z` 梯度（`20*vz`）远超 `swing_feet_z` 的指数梯度，策略放弃抬腿、选择小碎步滑行。减小竖向惩罚到比 R19 更低（-4.0 vs -5.0），让自然步态的竖向震荡得以释放。同时微降 `action_smooth` 避免过度平滑抑制动态迈步。
**结果**：best=**145.30**, final=**109.31**, episode=1000 零摔倒。tracking_vx=1.405, tracking_vy=1.191, tracking_ang_vel=1.435 (norm 0.957), cross_axis=-0.012, ang_vel_xy=-0.086, stand_still=-0.034, zero_command_stillness=0.297, swing_feet_z=1.17。

### Round 22 (增强域随机化, 500 iters) — 地形摩擦 + 推搡 2x + 质量范围 +67%
```yaml
# 核心修改 1: 代码层 — DR provider 支持 ground_friction 和 body_mass baseline 缓存
#   OpenDogeJoystickDomainRandomizationProvider 新增 __init__ 接收 base_geom_friction/ground_geom_id
#   env __init__ 从 backend 获取 geom_friction + geom_id 传入 DR provider
# 核心修改 2: randomize_ground_friction: false→true（缺失的 sim2real 因子）
# 核心修改 3: max_force: [0.3,0.3,0.2]→[0.6,0.6,0.4]（推搡力度翻倍）
# 核心修改 4: added_mass_range: [-0.3,0.3]→[-0.5,0.5]（质量范围 +67%，23% 体重）
# 核心修改 5: com_offset_x: [-0.02,0.02]→[-0.03,0.03]（质心偏移 +50%）
```
**动机**：地面摩擦随机化是 sim2real 转移的关键因子，此前缺失。推搡力度 0.3N 太弱（Δv≈0.003 m/s），翻倍到 0.6N 给予有意义的扰动。质量范围适度放大到 23% 体重。三项增强共同提升策略对环境变化的鲁棒性。
**结果**：best=**141.28**, final=**101.31**, episode=1000 零摔倒🔥。tracking_vx=1.166, tracking_vy=1.166, tracking_ang_vel=1.384, action_std=0.07, swing_feet_z=1.20。**vx/vy 首次完全对等**（1.166=1.166, gap=0.00），DR 强化迫使策略学习对称速度追踪。action_std 从 0.05→0.07（更柔软探索）。Best 降 4 分在预期内（DR 越强训练越难，sim2real 泛化越好）。

### Round 23 (追踪权重过高尝试, 500 iters) — DR 强化 + tracking 提权
```yaml
tracking_vx: 1.5→1.8       # +20%
tracking_vy: 1.5→2.0       # +33%
max_force: 0.6→0.8N        # +33%
push_interval: 500→250     # 2x 频率
added_mass: ±0.5→±0.7kg    # +40%
ground_friction: [0.6,1.4] # 2x 宽度
```
**动机**：R22 DR 增强后 tracking 下降，尝试提高追踪权重补偿 + 进一步强化 DR。
**结果**：best=**156.60** 🔥🔥🔥（新纪录！），final=**118.12**。tracking_vx=1.745 (归一化97%), tracking_vy=1.799 (90%), swing_feet_z=1.11。tracking 主导 reward 导致步态质量下降（swing 从 1.20→1.11）。**教训：tracking 权重过高挤压 gait。**

### Round 24 (追踪回退, 500 iters) — 保留 DR、追踪回 1.5
```yaml
tracking_vx: 1.8→1.5       # 回退
tracking_vy: 2.0→1.5       # 回退
# DR 保持 R23 强水平
```
**动机**：R23 tracking 过高挤压步态，回退追踪权重观察 DR 对 gait 的影响。
**结果**：best=**141.09**, final=**103.02**, swing_feet_z=1.12。DR 太强（push 0.8N/250 + friction [0.6,1.4]）本身就在压制步态，即使 tracking 回退也无法恢复 gait quality。**DR 强度与步态质量有直接 tradeoff。**

### Round 25 (步态优先, 500 iters) — 适中 DR + swing_feet_z 提权
```yaml
max_force: 0.8→0.6N, push_interval: 250→350  # DR 回退到适中
added_mass: ±0.7→±0.6, friction: [0.7,1.3]    # 摩擦范围收窄
swing_feet_z: 5.0→6.0                           # +20% 步高奖励
tracking: all 1.5                                # 保持平衡
```
**动机**：R23/R24 证明强 DR 与优质步态不可兼得。找到 DR 适中点（比 R22 稍强、比 R23 弱很多）+ 直接提权 `swing_feet_z` 以信号级优先步态质量。
**结果**：best=**143.40**, final=**109.98**, episode=1000 零摔倒。swing_feet_z=**1.44** 🔥🔥（所有轮次最高！+29% vs R24），tracking_vx=1.305, tracking_vy=1.196, tracking_ang_vel=1.414, action_std=0.08, lin_vel_z=-0.036。**步态质量与鲁棒性的最佳平衡点。**

### Round 26 (消除 linvel sim2real gap, 500 iters) — 49 维 actor obs + 非对称 actor-critic 🔥
```yaml
# 核心修改: actor 观测移除 linvel (52→49 维)，critic 保留 privileged linvel
#   gyro(3) + neg_gravity(3) + dof_pos_diff(12) + dof_vel(12)
#   + last_action(12) + commands(3) + feet_phase(4) = 49 维
# obs_groups_spec: {"obs": 49, "critic": 52}
# 所有 R25 reward 配置保持不变
```
**动机**：linvel 是 sim2real 的根本性 gap——仿真有 ground truth，实机永远无法精确获取。R15 引入 linvel 修复零指令漂移，但同时创建了跨域不可迁移的依赖。正确方案：actor 不用 linvel（49 维，全部可部署），critic 保留 linvel 做 privileged 值估计（非对称 actor-critic）。
**结果**：best=**143.08**, final=**113.03** 🔥🔥（final +3.05 vs R25！），swing_feet_z=**1.50** 🔥🔥🔥（新纪录），tracking_vx=1.41 (+0.11), tracking_vy=1.24 (+0.04), tracking_ang_vel=1.43 (+0.01), zero_command_stillness=0.26, episode=1000 零摔倒。**去掉 linvel 反而 final 更高——策略不再依赖不可靠信号，收敛更稳定。swing_feet_z 1.50 创下历史最高，证明非对称 actor-critic 完全生效。**

### Round 27 (步态结构, 800 iters) — joint_mirror 对称 + feet_air_time 飞行相
```yaml
# 核心修改 1: 新增 joint_mirror 奖励（scale=-0.05）
#   惩罚对角线腿对 (FL↔FR, RL↔RR) 的关节角度不对称
#   直接针对 tracking_vy (1.24) 滞后 tracking_vx (1.41) 的 ~12% 残余不对称
# 核心修改 2: 新增 feet_air_time 奖励（scale=0.5）
#   奖励触地时飞行相持续时间（上限 0.5s）
#   仅当 command_mag > 0.05（移动中）时激活
# 核心修改 3: max_iterations: 500→800
#   新 reward 信号需要更多迭代让策略探索新梯度景观
# 代码改动: joystick.py 新增 _update_contact_timers / _reward_feet_air_time / _reward_joint_mirror
#           DR provider _compute_reset_obs 调用 _reset_contact_timers
```
**动机**：R26 步态质量（swing_feet_z=1.50）已达历史最高，但 gait 结构完全依赖开环相位推进——没有 reward 直接要求正确的飞行相持续时间和左右对称。Go2 rough 已证明 `joint_mirror` 和 `feet_air_time` 是改善步态结构的有效 reward。对角线配对 (FL↔FR, RL↔RR) 而非单纯左右配对，因为 trot 步态下 FL+RR 同相、FR+RL 同相——比较同相腿对可避免与步态模式冲突。
**结果**：best=**140.85**, final=**117.03** 🔥🔥🔥（final **+4.00** vs R26！），episode=**1000 零摔倒** 🔥。tracking_vx=**1.45** (+0.04), tracking_vy=**1.27** (+0.03), tracking_ang_vel=**1.43** (持平), swing_feet_z=**1.56** 🔥🔥🔥（**+0.06 新纪录！**）, zero_command_stillness=**0.31** (+0.05), feet_air_time=**0.021**, joint_mirror=**-0.010**, action_std=**0.08**。

**Viser 诊断**：① 低速指令基本不动（exponential reward σ=0.25 在低 error 时梯度弱——cmd=0.15m/s 时站着不动也能拿 91% 奖励）；② 腿异常外摆（hip_yaw 无直接惩罚，策略学会叉腿走）；③ 颠簸感强（max_air_time=0.5s 对 2.2kg 机器人过高，策略为滞空用力蹬地导致弹跳）。

### Round 28 (低速+髋外摆+颠簸修复, 600 iters) — tracking_sigma↓ + hip_pos + max_air_time↓
```yaml
# 核心修改 1: tracking_sigma 0.25→0.18（低速梯度增强 ~2x）
#   cmd=0.15m/s 站着不动：91%→50% 奖励，策略被迫真正迈步
# 核心修改 2: 新增 hip_pos 惩罚（scale=-0.3）
#   仅约束 4 个 hip_yaw 关节（indices 0,3,6,9）不偏离 default_angles
#   不限制 thigh/calf 运动，保留步态自由度
# 核心修改 3: feet_air_time max_air_time 0.5s→0.25s
#   匹配 2.2kg 机器人在 2-2.5Hz 步频下的实际飞行相
#   避免策略为不合理滞空目标用力蹬地
# max_iterations: 600（改动量适中）
```
**动机**：R27 数值优秀但 viser 测试暴露三个体验问题。① 低速追踪：指数奖励本质缺陷——cmd 低时 error 也低，exp(-error²/σ) 在 error→0 时梯度趋于零。缩小 σ 是最小代码代价的修复。② 髋外摆：joint_mirror 只罚不对称不罚绝对值，叠加 hip_pos 直接限制髋关节范围。③ 颠簸：0.5s 是 Go2(12kg)的 max_air_time，按体重比例缩放给 2.2kg OpenDoge 约 0.25s。
**结果**：best=**140.76**, final=**107.99**, episode=**1000 零摔倒** 🔥。tracking_vx=1.45, tracking_vy=1.14, tracking_ang_vel=1.39, swing_feet_z=**1.55**, feet_air_time=**0.040**（+90% vs R27!）, hip_pos=**-0.018**, lin_vel_z=**-0.062**（阻尼 +38% vs R27）, joint_mirror=**-0.012**, action_smooth=**-0.005**（-25% vs R27，更平滑！）。

**注意**：final reward 数值下降（117→108）是 `tracking_sigma` 0.25→0.18 的**预期效应**——更窄的 sigma 让同一实际 tracking error 分数更低（0.1m/s error: σ=0.25 得 0.85, σ=0.18 得 0.73），因此总 reward 不可直接对比。真正的评判标准是 viser 行为质量。

**Viser 诊断**：① vx<0.25 不动，>0.25 才动；vy 完全废掉（σ=0.18 压缩 tracking reward → 策略放弃 vy）；② 髋外摆好一点不够（hip_pos -0.3 太弱）；③ 颠簸好一点不够（lin_vel_z -4.0 不够）。

### Round 29 (追踪权重非对称补偿, 800 iters) — σ 部分回退 + vy 非对称加成 + hip/弹跳增强
```yaml
# 核心修改 1: tracking_sigma 0.18→0.22（部分回退，保留低误差分辨力但不压缩过度）
# 核心修改 2: 追踪 scale 非对称补偿
#   tracking_vx: 1.5→1.8 (+20%) — 补偿 sigma 缩窄
#   tracking_vy: 1.5→2.5 (+67%) — vy 是非对称重心，更强梯度防止放弃
#   tracking_ang_vel: 1.5→1.8 (+20%)
# 核心修改 3: hip_pos -0.3→-0.5 (+67% 髋约束)
# 核心修改 4: lin_vel_z -4.0→-6.0 (+50% 弹跳阻尼)
# max_iterations: 800
# 保留 R28: max_air_time=0.25s
```
**动机**：R28 σ=0.18 过于激进，tracking reward 被整体压缩后策略优化重心漂移——vy 本就比 vx 难学，reward 压缩后策略直接放弃 vy 追踪。改为 σ=0.22 + 非对称 scale 补偿：vy scale 提 67% 远超 vx 的 20%，直接告诉策略"vy 很重要"。hip/lin_vel_z 纯增强，无风险。
**结果**：best=**171.94** 🔥🔥🔥🔥🔥🔥🔥, final=**146.95** 🔥🔥🔥🔥🔥🔥🔥（双双历史纪录！best 超 R23 +15.34, final +28.83）。tracking_vx=**1.71**/1.8 (95%), tracking_vy=**2.38**/2.5 (95%), tracking_ang_vel=**1.68**/1.8 (93%), swing_feet_z=**1.40**, feet_air_time=**0.023**, hip_pos=**-0.019**, lin_vel_z=**-0.045**, episode=**1000 零摔倒** 🔥。**非对称 vy 加成从"废掉"拉到 95% 归一化追踪，完全修复 R28 vy 崩溃。**

**Viser 诊断**：vy 恢复！但 VX/VY 步频过高、步高过低、步长过短。根因：`hip_pos -0.5` + `lin_vel_z -6.0` 双重过约束——髋不能外展 + 身体不能起伏 = 策略只能用极小碎步移动（R20 lin_vel_z=-10 的翻版）。

### Round 30 (释放过约束, 600 iters) — hip_pos↓ + lin_vel_z 回退
```yaml
# 核心修改 1: hip_pos -0.5→-0.25（释放髋关节，允许正常步幅）
# 核心修改 2: lin_vel_z -6.0→-4.0（回退到 R25-27 验证过的弹跳阻尼水平）
# 保留 R29: σ=0.22, tracking vx=1.8/vy=2.5/ang_vel=1.8, max_air_time=0.25s
# max_iterations: 600
```
**动机**：R29 追踪/vy 全部顶级，但 gait 被过度约束成碎步。hip_pos 降到 -0.4，lin_vel_z 回退到 -4.0 释放活动空间，保留 R29 的所有追踪收益。
**结果**：best=**171.16**, final=**146.21**, episode=**1000 零摔倒** 🔥。reward 与 R29 几乎持平（-0.5%），但 hip_pos 约束释放 53%（-0.019→-0.009），lin_vel_z 约束释放 36%（-0.045→-0.029）。tracking_vy=**2.46**/2.5 (98.6%🔥)。

**Viser 诊断**：全向移动恢复！但步态仍是小碎步。根因：tracking 贡献 ~5.87 占总 reward 过大，策略优化预算被追踪吸走，swing_feet_z 仅 1.35/6.0 (23%)——没有留给步态。

### Round 31 (tracking 回退 + swing 增强, 600 iters) — 降低追踪占比重，提高步态激励
```yaml
# 核心修改 1: tracking_vx 1.8→1.5, vy 2.5→2.0, ang_vel 1.8→1.5
#   tracking 总权重从 6.1 降到 5.0，释放优化预算给步态
# 核心修改 2: swing_feet_z 6.0→8.0 (+33% 步高激励)
#   加强 swing 在总 reward 中的占比，直接对抗追踪主导
# 保留: σ=0.22, hip_pos=-0.4, lin_vel_z=-4.0, max_air_time=0.25s
```
**动机**：R30 证实过约束不是碎步根因——释放后全向追踪正常但步态不变。真正的根因是 tracking scale 远高于 swing（6.1 vs 6.0 最大，但 tracking 命中 95% → 实际贡献 5.87，swing 命中仅 23% → 只贡献 1.35）。追踪主导导致策略忽视步态质量（R23 的翻版）。降低追踪权重 + 提 swing 权重纠正优化重心。
**结果**：best=**154.29**, final=**131.08**, episode=**1000 零摔倒**。tracking_vx=1.43/1.5 (95%), tracking_vy=1.60/2.0 (80%), swing_feet_z=**2.13**/8.0 (27%), feet_air_time=**0.043**。**步态恢复正常！步高 +54% vs R30。但 reward 因 tracking 权重降低而缩水。**

### Round 32 (线性追踪 + 去僵硬 + vy, 600 iters) — tracking_vel_linear 新奖励
```yaml
# 核心修改 1: 新增 tracking_vel_linear L1 惩罚（scale=-1.0）
#   L1 = |cmd_xy - vel_xy|，提供恒定梯度，修复 exponential reward 低速梯度为零的数学缺陷
# 核心修改 2: entropy_coef 3e-3→5e-3 (+67% 探索，去僵硬)
# 核心修改 3: action_smooth -0.004→-0.002, action_rate -0.008→-0.005 (更动态)
# 核心修改 4: tracking_vy 2.0→2.3 (+15% vy 补偿)
# 代码改动: joystick.py 新增 _reward_tracking_vel_linear
```
**动机**：R31 步态恢复但低速仍不动 + 动作僵硬 + vy 偏弱。exponential reward 的本质缺陷——gradient ∝ error → 0 as error → 0，无论怎么调 σ 低速 gradient 永远弱。线性 L1 penalty 在所有速度段提供恒定 gradient，彻底解决低速追踪。
**结果**：best=**162.07**, final=**139.68**, episode=**1000 零摔倒**。tracking_vel_linear 从 -0.58 收敛到 **-0.07**（-88%），恒定梯度压缩所有追踪误差。**低速修复 + vy 改善 + 僵硬减轻，但 linear=-1.0 过强又导致碎步回归。**

### Round 33 (linear 松弛 + swing 增强, 600 iters) — 最优平衡点 🔥
```yaml
# 核心修改 1: tracking_vel_linear -1.0→-0.4（释放瞬时速度波动容忍度）
# 核心修改 2: swing_feet_z 8.0→10.0 (+25% 步高激励)
# 保留 R32: σ=0.22, entropy=5e-3, tracking vx=1.5/vy=2.3/ang_vel=1.5
```
**动机**：R32 linear=-1.0 恒定梯度过强，策略为追求完美匀速压制了步态周期中自然的瞬时速度波动 → 碎步。降到 -0.4 保留低速梯度 + 释放速度波动容忍；swing 提到 10.0 增强步态信号对抗残余追踪压力。
**结果**：best=**176.01** 🔥🔥🔥🔥🔥🔥🔥, final=**154.09** 🔥🔥🔥🔥🔥🔥🔥（**双双历史纪录！**）。episode=**1000 零摔倒**。**低速追踪 ✅ + vy 侧移 ✅ + 动作自然 ✅ + 步态正常 ✅——四项全部修复。**

### Round 34 (全面增强域随机化, 1500 iters) — DR 6 维扩展，reward 不降反升 🔥🔥🔥

```yaml
# 核心修改 1: 推搡增强 — max_force: [0.6,0.6,0.4]→[0.8,0.8,0.5], push_interval: 350→300
# 核心修改 2: CoM Y/Z 偏移 NEW — com_offset_y: [-0.02,0.02], com_offset_z: [-0.01,0.01]
# 核心修改 3: per-link 质量随机化 NEW — randomize_body_mass: true, body_mass_multiplier_range: [0.92,1.08]
# 核心修改 4: 重力方向随机化 NEW — randomize_gravity: true, gravity_range: ±0.5 m/s² (~±3° 倾斜)
# 核心修改 5: 传感器噪声增强 — 关节角度 ±1.7°→±4.6°, 关节速度 +60%, 陀螺 +75%
# 核心修改 6: max_iterations: 600→1500
# 代码改动: domain_rand.py 新增 com_offset_y/z 字段
```
**动机**：R33 在最优奖励配置下达成 best=176、final=154，但 DR 存在 5 个已知 gap——推搡偏弱、CoM 仅 X 轴、无重力倾斜、无 per-link 质量、传感器噪声偏低（关节角度 ±1.7° vs 真实舵机 ±3-5°）。这些 gap 直接威胁 sim2real 转移。R23 曾证明激进 DR 会压低步态质量，本轮的策略是：**中等增量，6 维同时扩展，用更长训练（1500 iter）让策略充分适应新分布**。

**结果**：best=**179.28** 🔥🔥🔥🔥🔥🔥🔥🔥, final=**155.21** 🔥🔥🔥🔥🔥🔥🔥🔥（best **+3.27** vs R33, final **+1.12**，双双新纪录！）。episode=**1000 零摔倒** 🔥。tracking_vx=**1.47**/1.5 (97.9%), tracking_vy=**2.26**/2.3 (98.3%), tracking_ang_vel=**1.38**/1.5 (92.3%), tracking_vel_linear=**-0.040** (57% 改善 vs R33), swing_feet_z=**2.67**/10.0 (+25%), feet_air_time=**0.041**, zero_command_stillness=**0.248**, cross_axis=-0.017, lin_vel_z=-0.057, ang_vel_xy=-0.137。

**关键发现**：DR 增强后 **reward 不降反升**（best +3.27, final +1.12）！这与 R23（DR 增强 → reward 下降）的历史教训形成鲜明对比。分析原因：
1. **多维度小增量 > 单维度大增量**：6 个维度各加 20-67%，而非 R23 把推搡/质量/摩擦各翻倍。策略在每个维度上都能找到补偿策略，累积效果是更鲁棒而非被击垮。
2. **1500 iter 充分收敛**：R23 只跑 500 iter，策略来不及在强 DR 下找到最优解。本轮 1500 iter 给了策略 3x 的适应时间。
3. **速度追踪全部提升**（vx +2.8%, vy +2.7%, linear penalty 57% 改善）——更强的 DR 迫使策略学习更紧的速度闭环，而不是依赖开环步态。
4. **稳定性指标下降属预期**（zero_command_stillness 0.34→0.25, lin_vel_z -0.029→-0.057, ang_vel_xy -0.073→-0.137）——在推搡+重力倾斜下不可能完美静止，这是 sim2real 的合理 tradeoff。

### Round 35 (vy 侧移优化, 1000 iters) — 降低 vy 追踪压力 + 释放髋外展 + 增加纯轴暴露

```yaml
# 核心修改 1: tracking_vy scale: 2.3→1.8 (-22%)
#   接受髋内收 15° 的机械约束，不强迫策略用碎步补偿物理极限
# 核心修改 2: hip_pos scale: -0.4→-0.25 (-37%)
#   释放髋外展/内收约束，允许更大侧向步幅
# 核心修改 3: pure_axis_prob: 0.15→0.25 (+67%)
#   更多纯单轴命令暴露（含 vy-only），增强侧向独立追踪能力
# max_iterations: 1000（改动量适中）
# 代码改动: commands.py 新增 pure_axis_prob 字段（原 getattr 动态读取）
```
**动机**：R34 在 vy 方向仍存在小碎步——髋内收仅 15°（0.26 rad），单步侧移幅仅 ~22mm，策略被迫高频小步补偿。机械约束是根因，无法靠 reward 工程突破。本轮的策略：**A) 降低 vy 期望** 消除不适应的强迫；**B) 释放 hip 约束** 让策略充分利用可用运动范围；**C) 增加纯 vy 轴暴露** 强化侧移独立学习。

**结果**：best=**166.23**, final=**146.33**, episode=**1000 零摔倒** 🔥。tracking_vx=1.46/1.5 (97%), tracking_vy=**1.73**/1.8 (**96.4%**), tracking_ang_vel=1.36/1.5 (91%), tracking_vel_linear=-0.052, swing_feet_z=**2.85** (+6.8% vs R34), feet_air_time=**0.049** (+19.5%), zero_command_stillness=**0.299** (+20.6%), hip_pos=-0.030, ang_vel_xy=-0.199 (变差), lin_vel_z=-0.068。

**reward 下降分析**：best -13.05, final -8.88。但 ~60% 的降幅来自 tracking_vy scale 本身下调（2.3→1.8，满额差 0.5）——这是预期的，不代表策略变差。归一化 vy 追踪率 96.4%（R34 为 98.3%）基本持平。

**步态质量全面提升**：
① swing_feet_z +6.8% → 更舒展的抬腿
② feet_air_time +19.5% → 更好的飞行相持续时间
③ zero_command_stillness +20.6% → 零指令下更稳定静止
④ hip_pos 释放后策略确实使用了更多髋关节范围（per-unit 偏差 +41%）

**代价**：ang_vel_xy -0.137→-0.199（+45% 变差）——释放髋约束 + 更多 vy 暴露导致身体摇摆增加。这是预期内的 tradeoff。

**Viser 需验证**：reward 归一化后 vy 追踪持平，步态指标全面改善。但最终评判标准是 viser 中侧移是否从"高频小碎步"变为"更舒展的侧步"。

### Round 36 (动作平滑度, 600 iters) — action_smooth 翻倍 + ang_vel_xy 增强 + lin_vel_z 微调

```yaml
# 核心修改 1: action_smooth scale: -0.002→-0.004 (+100% jerk 惩罚)
#   直接压制二阶动作不平滑（抖动），最针对性的平滑度工具
# 核心修改 2: ang_vel_xy scale: -0.5→-0.7 (+40% 身体摇摆阻尼)
#   R35 释放 hip 后身体 roll/pitch 震荡恶化 45%，增强阻尼但不收紧 hip
# 核心修改 3: lin_vel_z scale: -4.0→-5.0 (+25% 竖向阻尼)
#   适度增强，在安全范围内（上限 -5.0）
# max_iterations: 600（改动量小，快速验证）
```

**动机**：R35 释放 hip 约束（-0.4→-0.25）改善了侧移步态但副作用是身体摇摆显著恶化（ang_vel_xy -0.137→-0.199）。根因是髋外展运动会通过机械耦合带动身体 roll/pitch，策略在弱平滑度惩罚下缺乏抑制震荡的激励。不重新收紧 hip（会撤销步态改善），改用平滑度专用工具三管齐下：jerk（action_smooth）、身体角速度（ang_vel_xy）、竖向震荡（lin_vel_z）各自独立约束不同维度的不平滑。

**结果**：best=**169.88** (+3.65 vs R35), final=**136.90** (-9.43 vs R35), episode=**1000 零摔倒** 🔥。tracking_vx=1.46/1.5 (97.6%), tracking_vy=1.73/1.8 (96.3%), tracking_ang_vel=1.37/1.5 (91.2%), tracking_vel_linear=-0.049 (+6% vs R35), swing_feet_z=2.41 (-15% vs R35), feet_air_time=0.027 (-45% vs R35), zero_command_stillness=0.257 (-14%), hip_pos=-0.011 (+63% 🔥), ang_vel_xy=-0.188 (+6%), lin_vel_z=-0.052 (+24%), action_smooth=-0.013, episode=1000 零摔倒。

**平滑度目标全部命中** 🔥：ang_vel_xy +6%（身体摇摆减轻），lin_vel_z +24%（竖向震荡减轻），hip_pos +63%（髋关节更稳定）。追踪持平（vx/vy/ang_vel 几乎不变）。

**代价**：swing_feet_z -15%、feet_air_time -45%——`action_smooth -0.004`（+100%）压制了动态迈步所需的自然加速度变化。步态从"舒展"变"保守"。best +3.65 但 final -9.43——策略探索到更高峰值但收敛稳定性下降，平滑约束使优化景观更崎岖。

**教训**：平滑度与步态动态性遵循经典 tradeoff。`action_smooth -0.004` 制造的是"僵硬平滑"（类似 R20 `lin_vel_z=-10`），把所有加速度变化（含自然迈步的）都压制了。Viser 验证：身体摇摆 ✅ 减轻，但动作不自然、步高低、vy 仍碎步。vx 基本可用。

### Round 37 (僵硬修复 + body sway 保留, 600 iters) — action_smooth 回退 + swing 补偿

```yaml
# 核心修改 1: action_smooth scale: -0.004→-0.002（完全回退 R36，消除僵硬）
# 核心修改 2: ang_vel_xy scale: keep -0.7（保留 R36 身体摇摆抑制收益）
# 核心修改 3: lin_vel_z scale: -5.0→-4.0（回退到 R35，避免与 ang_vel_xy 叠加过阻尼）
# 核心修改 4: swing_feet_z scale: 10.0→12.0 (+20% 步高补偿)
# 核心修改 5: action_rate scale: -0.005→-0.004 (-20% 释放一阶约束)
# max_iterations: 600（改动量适中）
```

**动机**：R36 三类平滑度指标全中但 viser 揭示动作僵硬不自然——`action_smooth -0.004` 是主犯。它不区分"好的自然加速度变化"和"坏的抖动"，统一压制。回退 action_smooth 到 R35 水平，保留 ang_vel_xy=-0.7（已验证有效且不导致僵硬——它约束身体角速度而非关节运动），提 swing_feet_z 补偿步高损失，微降 action_rate 释放一阶动态。

**结果**：best=**182.46** 🔥🔥🔥🔥🔥🔥🔥🔥🏆（**历史新纪录！** +16.23 vs R35, +12.58 vs R36）, final=**148.92** (+2.59 vs R35, +12.02 vs R36), episode=**1000 零摔倒** 🔥。tracking_vx=1.44/1.5 (96%), tracking_vy=1.68/1.8 (94%), tracking_ang_vel=1.35/1.5 (90%), tracking_vel_linear=-0.067, swing_feet_z=**3.32** 🔥🔥🔥（**历史最高！** +16% vs R35 的 2.85）, feet_air_time=0.043 (+59% vs R36), action_smooth=-0.005 (+58% vs R36), action_rate=-0.011 (+31% vs R36), ang_vel_xy=-0.267（实际物理摇摆 0.381 vs R35 的 0.398, -4%）, zero_command_stillness=0.217 (-27% vs R35), hip_pos=-0.024, joint_mirror=-0.012。

**三项核心问题全修复**：
① 僵硬消除 ✅ — action_smooth 从 -0.013→-0.005（实际 jerk 降低 17%），动作恢复动态
② 步高恢复 ✅ — swing_feet_z 从 2.41→3.32（+38% vs R36），超越 R35 创纪录
③ 收敛恢复 ✅ — final 从 136.90→148.92（+8.8%），不再末期坍塌

**ang_vel_xy "恶化"是数值假象**：R37 scale=-0.7 产生 -0.267，但实际身体角速度均方值 0.381 比 R35（scale=-0.5 时 0.398）更低。`ang_vel_xy=-0.7` 确实在抑制摇摆，只是更大的 scale 放大了 reward 数值。

**代价**：zero_command_stillness 从 0.299→0.217（-27%），更动态的步态在零指令时更难完美静止。swing_feet_z 虽创纪录但归一化仅 28%（3.32/12.0），说明 12.0 的 scale 有空间继续推动步高。

### 当前最优配置 (Round 37) 🏆

best=**182.46**, final=**148.92**, swing_feet_z=**3.32**（历史最高）

```yaml
# conf/ppo/task/opendoge_joystick_flat/mujoco.yaml
# @package _global_
training:
  task_name: OpenDogeJoystickFlat
  sim_backend: mujoco
algo:
  num_envs: 1024
  max_iterations: 1000
  empirical_normalization: true
  obs_groups:
    actor:
      - actor
  policy:
    init_noise_std: 0.5
  algorithm:
    learning_rate: 3.0e-4
    entropy_coef: 5.0e-3
env:
  commands:
    rel_standing_envs: 0.1
    pure_axis_prob: 0.25
    vel_limit:
      - [-0.8, -0.6, -1.5]
      - [0.8, 0.6, 1.5]
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
    tracking_ang_vel: 1.5
    cross_axis_suppression: -0.6
    lin_vel_z: -4.0
    ang_vel_xy: -0.7
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
    tracking_vel_linear: -0.4
  tracking_sigma: 0.22
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
| **R20** | **145.35** | **113.23** 🔥 | lin_vel_z ↑2x + action_smooth ↑67%, vy追平vx! 步高小步频快 |
| **R21** | **145.30** | **109.31** | lin_vel_z -4.0 释放震荡 + action_smooth -0.004 |
| **R22** | **141.28** | **101.31** | 地形摩擦 + 推搡 2x + 质量范围 +67%, vx=vy 🔥 |
| R23 | **156.60** 🔥🔥🔥 | **118.12** | tracking 提权过高，reward 虚高但压缩 gait |
| R24 | 141.09 | 103.02 | tracking 回退但 DR 仍压制步态 |
| **R25** | **143.40** | **109.98** | 适中 DR + swing_feet_z 6.0, gait 1.44 🔥🔥 |
| **R26** | **143.08** | **113.03** 🔥🔥 | 🔥 49-dim actor (去 linvel), 非对称 actor-critic, swing 1.50 新纪录 |
| **R27** | **140.85** | **117.03** 🔥🔥🔥 | 🔥 joint_mirror + feet_air_time, final +4.00, swing 1.56 新纪录 |
| **R28** | **140.76** | **107.99** ⚠️ | σ=0.18 + hip_pos -0.3 + max_air 0.25s, feet_air +90%, 更平滑（reward 降因 σ 收窄不可对比） |
| **R29** | **171.94** 🔥🔥🔥🔥🔥🔥🔥 | **146.95** 🔥🔥🔥🔥🔥🔥🔥 | 🔥 vy 2.5 非对称补偿, best+final 双历史纪录, vy 恢复但步态碎 |
| **R30** | **171.16** | **146.21** | hip_pos -0.4 + lin_vel_z -4.0, 全向恢复但步态仍碎（tracking 占比过高） |
| **R31** | **154.29** | **131.08** | tracking↓ swing↑ 8.0 → **步态恢复正常！** |
| **R32** | **162.07** | **139.68** | +tracking_vel_linear 线性追踪, 低速修复但 linear=-1.0 过强碎步回归 |
| **R33** | **176.01** 🔥🔥🔥🔥🔥🔥🔥 | **154.09** 🔥🔥🔥🔥🔥🔥🔥 | 🔥 linear -0.4 + swing 10.0, **best+final 双历史纪录, 四项全部修复** |
| **R34** | **179.28** 🔥🔥🔥🔥🔥🔥🔥🔥 | **155.21** 🔥🔥🔥🔥🔥🔥🔥🔥 | 🔥 全面增强 DR (6维: 推搡+CoM YZ+per-link质量+重力+噪声), **DR 增强后 reward 不降反升！** |
| **R35** | **166.23** | **146.33** | 🔥 vy 侧移优化: tracking_vy↓22% + hip_pos 释放 37% + pure_axis 25%, **步态质量全面改善** |
| **R36** | **169.88** | **136.90** ⚠️ | 🔧 动作平滑度: smoothness 全中 (ang_vel +6%, lin_vel_z +24%, hip +63%)，但 gait -15% (action_smooth 过强) |
| **R37** | **182.46** 🔥🏆 | **148.92** | 🔥 僵硬修复: action_smooth 回退 + swing 12.0, **best+final 双新高，swing 3.32 纪录** |

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
- ✅ **CoM 全向偏移**（X/Y/Z 三轴独立随机）
- ✅ **per-link 质量随机化**（每条连杆 ±8% 独立缩放）
- ✅ **重力方向随机化**（±0.5 m/s²，~±3° 地形倾斜）
- ✅ **观测噪声**（传感器噪声鲁棒）
- ✅ **actor linvel 观测**（策略可直接感知速度，零指令漂移自纠正）
- ✅ **joint_mirror 对称惩罚**（对角线腿对关节不对称抑制，消解 vx/vy 追踪 gap）
- ✅ **feet_air_time 飞行相奖励**（触地时奖励飞行相持续时间，max_air_time=0.25s 适配小型机器人）
- ✅ **hip_pos 髋外摆抑制**（hip_yaw L2 惩罚，约束外摆不限制步幅）
- ✅ **tracking_vel_linear 线性速度追踪**（L1 惩罚，恒定梯度修复 exponential 低速梯度为零的数学缺陷）
- ✅ **非对称 actor-critic 架构**（49 维 actor 全可部署，52 维 critic 含 privileged linvel）

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
19. **DR 强度与步态质量有直接 tradeoff，需找适中点而非最大化**：R23/R24 证明强 DR (push 0.8N/250 + friction [0.6,1.4]) 会直接压制 swing_feet_z (1.20→1.12)。增强 `swing_feet_z` 奖励 (5.0→6.0) 可以部分对抗，但根本解决方案是 DR 适度（push 0.6N/350 + friction [0.7,1.3]）。鲁棒性不是参数越大越好——找到不破坏 gait 的最大 DR 才是目标。
18. **增强 DR 必然降低训练 reward，但 episode 零摔才是真指标**：R22 增强 DR 后 best 降 4 分、final 降 8 分，但 episode=1000 零摔倒 + vx/vy 首次对称 — 策略在更嘈杂环境中学会了更鲁棒、更对称的行为。训练 reward 下降不一定是坏事，要结合 episode length 和追踪对称性判断。
20. **linvel 是 sim2real 根本性 gap，不应出现在 actor 观测中**：仿真 `get_local_linvel()` 提供精确 body frame 速度，但实机无法精确获取（腿运动学累积误差、IMU 积分漂移、HiMLoco 也需额外模型）。R15 引入 linvel 修复零指令漂移但创建了跨域不可迁移的依赖。R26 将 linvel 从 actor 移除（49 维），critic 保留为 privileged info（52 维），采用非对称 actor-critic。结果 final reward +3.05，swing_feet_z 1.50 创纪录 — 策略不再依赖不可靠信号反而表现更好。
17. **`lin_vel_z` 过度压制会摧毁步态**：`lin_vel_z` 惩罚所有竖向速度（含迈步必有的自然身体起伏），梯度 `2*scale*|vz|`（scale=-10 时为 `20|vz|`）远超 `swing_feet_z` 的指数梯度。过强的 `lin_vel_z` 迫使策略放弃抬腿（因为抬腿→身体微降→下一步推起→身体微升→全程被罚），选择极小碎步保持身体平板滑行。**竖向震荡是自然步态的副产品，只能适度约束不能彻底消灭**。OpenDoge（2.2kg）的合理范围约 `-3.0~-5.0`，`-10.0` 已明显过度。
21. **exponential tracking reward 存在数学缺陷——低速梯度趋于零**：`exp(-e²/σ²)` 的 gradient ∝ e → 0 as e → 0。无论怎么调 σ，低指令速度时"站着不动"的惩罚永远弱。cmd=0.2m/s 时 σ=0.10 也只扣 33%。**修复方式：叠加 L1 线性 penalty `|cmd - vel|`（scale ~-0.4），提供恒定梯度。**训练期用 linvel（和 exponential tracking 一样算 reward），部署期策略不需要 linvel。
22. **追踪权重过高挤压步态是反复出现的模式**：R23（tracking 1.8/2.0）、R29-30（tracking 1.8/2.5/1.8）、R32（linear -1.0）均出现 tracking 占主导导致碎步。**tracking 总 max 控制在 ~5.0，swing_feet_z 达到 8.0-10.0 才能保持步态质量。**
23. **过约束（hip + lin_vel_z 同时高强度）导致"束身衣"碎步**：R29 hip_pos=-0.5 + lin_vel_z=-6.0 双重压制，策略无法自然迈步只能碎步。**约束项要逐一调，不能同时加码。**
24. **de-stiffen 三件套有效**：同时提高 entropy（3e-3→5e-3）、降低 action_smooth（-0.004→-0.002）、降低 action_rate（-0.008→-0.005）可显著增加步态自然度，不牺牲追踪质量。
25. **max_air_time 须按机器人尺寸缩放**：Go2(12kg) 的 0.5s→OpenDoge(2.2kg) 约 0.25s。过长的目标迫使策略用力蹬地产生颠簸。**小型机器人物参数不能照搬大型机器人。**
26. **DR 多维度小增量优于单维度大增量**：R23 在 3 个维度各翻倍 DR（推搡 0.8N/250 + 质量 ±0.7 + 摩擦 [0.6,1.4]），reward 下降且步态崩溃。R34 在 6 个维度各增加 20-67%（推搡 0.8N/300 + 质量 ±0.6 + CoM YZ + per-link 质量 + 重力 + 噪声），reward 反而提升。策略在每个维度上都能找到补偿策略，多维度适中的累积扰动比少数维度的极端扰动更容易适应。**sim2real DR 的金发女孩原则：均匀覆盖所有 gap 维度，每个维度适度而非少数维度极端。**

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
