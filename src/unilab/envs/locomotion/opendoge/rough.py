from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg, TerrainSceneCfg
from unilab.dr import DomainRandomizationManager, ResetPlan
from unilab.dr.dr_utils import build_common_reset_randomization, zero_actions
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import (
    np_quat_apply_inverse,
    np_quat_from_euler_xyz,
    np_quat_mul,
)
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import (
    apply_heading_yaw_feedback,
    sample_heading_commands,
    zero_small_xy_commands,
)
from unilab.envs.locomotion.common.height_scan import (
    HeightScanConfig,
    base_height_from_scan,
    height_scan_obs,
    init_height_scan_sensor,
    raw_height_scan_obs,
    terrain_out_of_bounds,
)
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.opendoge.base import ControlConfig
from unilab.envs.locomotion.opendoge.joystick import (
    Commands,
    HIP_YAWS,
    JoystickSensor,
    OpenDogeJoystickCfg,
    OpenDogeJoystickDomainRandomizationProvider,
    OpenDogeWalkTask,
    RewardConfig,
)
from unilab.terrains import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    flat,
    hf_pyramid_slope,
    hf_pyramid_slope_inv,
    pyramid_stairs,
    pyramid_stairs_inv,
    random_rough,
    wave_terrain,
)

# ── OpenDoge-specific indices ──────────────────────────────────────────
FRONT_LEFT = 0
FRONT_RIGHT = 1
REAR_LEFT = 2
REAR_RIGHT = 3


@dataclass
class RoughControlConfig(ControlConfig):
    clip_actions: float = 100.0


@dataclass
class RoughCommands(Commands):
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-0.8, -0.6, -1.5], [0.8, 0.6, 1.5]]
    )
    resampling_time: float = 10.0
    heading_command: bool = True
    heading_range: list[float] = field(default_factory=lambda: [-np.pi, np.pi])


@dataclass
class RoughRewardConfig(RewardConfig):
    stand_still_command_threshold: float = 0.1
    contact_threshold: float = 1.0
    feet_air_time_threshold: float = 0.5
    feet_height_body_target: float = -0.1
    feet_height_body_tanh_mult: float = 2.0
    feet_gait_std: float = float(np.sqrt(0.5))
    feet_gait_max_err: float = 0.2
    feet_gait_velocity_threshold: float = 0.5
    feet_gait_command_threshold: float = 0.1
    max_air_time: float = 0.25


@dataclass
class RoughJoystickSensor(JoystickSensor):
    feet_vel = ["FL_vel", "FR_vel", "RL_vel", "RR_vel"]
    undesired_contact = [
        "base1_contact",
        "FL_hip_contact",
        "FR_hip_contact",
        "RL_hip_contact",
        "RR_hip_contact",
        "FL_thigh_contact",
        "FR_thigh_contact",
        "RL_thigh_contact",
        "RR_thigh_contact",
        "FL_calf_contact",
        "FR_calf_contact",
        "RL_calf_contact",
        "RR_calf_contact",
    ]


@dataclass
class RoughTerminationConfig:
    terrain_out_of_bounds: bool = True
    terrain_distance_buffer: float = 3.0


@dataclass(kw_only=True)
class OpenDogeRoughTerrainCfg(TerrainGeneratorCfg):
    """Terrain generator scaled for 2.2 kg OpenDoge (smaller steps, gentler roughness)."""

    size: tuple[float, float] = (8.0, 8.0)
    num_rows: int = 6
    num_cols: int = 6
    border_width: float = 20.0
    add_lights: bool = True
    horizontal_scale: float = 0.2

    sub_terrains: dict[str, SubTerrainCfg] = field(
        default_factory=lambda: {
            # R2: further reduced difficulty after R1 episode=191 (falls too often).
            # OpenDoge: 2.2 kg, 0.17 m standing height, 0.20 m leg length.
            "flat": flat(proportion=0.25),
            "pyramid_stairs": pyramid_stairs(
                proportion=0.08,
                step_height_range=(0.01, 0.03),
                step_width=0.4,
                platform_width=3.0,
                border_width=0.2,
            ),
            "pyramid_stairs_inv": pyramid_stairs_inv(
                proportion=0.08,
                step_height_range=(0.01, 0.03),
                step_width=0.4,
                platform_width=3.0,
                border_width=0.2,
            ),
            "hf_pyramid_slope": hf_pyramid_slope(
                proportion=0.15,
                slope_range=(0.0, 0.10),
                platform_width=2.0,
                border_width=0.2,
            ),
            "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
                proportion=0.15,
                slope_range=(0.0, 0.10),
                platform_width=2.0,
                border_width=0.2,
            ),
            "random_rough": random_rough(
                proportion=0.15,
                noise_range=(0.002, 0.012),
                noise_step=0.005,
                border_width=0.2,
            ),
            "wave_terrain": wave_terrain(
                proportion=0.14,
                amplitude_range=(0.0, 0.03),
                num_waves=4,
                border_width=0.2,
            ),
        }
    )


@registry.envcfg("OpenDogeJoystickRough")
@dataclass
class OpenDogeJoystickRoughCfg(OpenDogeJoystickCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "opendoge" / "opendoge.xml"),
            fragment_files=[
                str(ASSETS_ROOT_PATH / "robots" / "opendoge" / "locomotion_task.xml"),
            ],
            terrain=TerrainSceneCfg(
                generator=OpenDogeRoughTerrainCfg(),
                hfield_name="terrain_hfield",
                geom_name="floor",
            ),
        )
    )
    control_config: RoughControlConfig = field(default_factory=RoughControlConfig)
    commands: RoughCommands = field(default_factory=RoughCommands)
    terrain_scan: HeightScanConfig = field(default_factory=HeightScanConfig)
    termination_config: RoughTerminationConfig = field(
        default_factory=RoughTerminationConfig
    )
    sensor: RoughJoystickSensor = field(default_factory=RoughJoystickSensor)
    reward_config: RoughRewardConfig | None = None


# ── DR Provider ──────────────────────────────────────────────────────


class OpenDogeJoystickRoughDomainRandomizationProvider(
    OpenDogeJoystickDomainRandomizationProvider
):
    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        commands = super()._sample_commands(env, num_reset)
        zero_small_xy_commands(commands, threshold=0.03)
        if env.cfg.commands.heading_command:
            commands[:, 2] = 0.0
        return commands

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        # spawn on terrain cell with random xy offset
        qpos[:, 0:2] += np.random.uniform(-0.5, 0.5, (num_reset, 2))
        qpos[:, 2] += np.random.uniform(0.1, 0.3, (num_reset,))
        qpos[:, 0:3] += env._spawn.origins_for(env_ids)
        # random orientation
        roll = np.random.uniform(-np.pi, np.pi, (num_reset,))
        pitch = np.random.uniform(-np.pi, np.pi, (num_reset,))
        yaw = np.random.uniform(-np.pi, np.pi, (num_reset,))
        qpos[:, 3:7] = np_quat_mul(
            qpos[:, 3:7], np_quat_from_euler_xyz(roll, pitch, yaw)
        )
        # small random initial velocities
        qvel[:, 0:6] = np.asarray(
            np.random.uniform(-0.5, 0.5, size=(num_reset, 6)),
            dtype=get_global_dtype(),
        )
        commands = self._sample_commands(env, num_reset)
        info_updates: dict[str, Any] = {
            "commands": commands,
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
            "qacc": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
            "torques": np.zeros((num_reset, env._num_action), dtype=get_global_dtype()),
        }
        if env.cfg.commands.heading_command:
            info_updates["heading_commands"] = sample_heading_commands(env, num_reset)
        env._spawn.record_episode_start(env_ids, qpos[:, 0:3])
        base_kp, base_kd = self._get_base_actuator_gains(env)
        base_body_mass, base_geom_friction, ground_geom_id, base_dof_armature = (
            self._get_reset_randomization_baselines(env)
        )
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=build_common_reset_randomization(
                env,
                num_reset,
                base_kp=base_kp,
                base_kd=base_kd,
                base_body_mass=base_body_mass,
                base_geom_friction=base_geom_friction,
                ground_geom_id=ground_geom_id,
                base_dof_armature=base_dof_armature,
            ),
        )


# ── Env ──────────────────────────────────────────────────────────────


@registry.env("OpenDogeJoystickRough", sim_backend="mujoco")
class OpenDogeJoystickRoughEnv(OpenDogeWalkTask):
    _cfg: OpenDogeJoystickRoughCfg
    _reward_cfg: RoughRewardConfig

    def __init__(
        self, cfg: OpenDogeJoystickRoughCfg, num_envs=1, backend_type="mujoco"
    ):
        self._height_scan_dim = len(cfg.terrain_scan.measured_points_x) * len(
            cfg.terrain_scan.measured_points_y
        )
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        # Replace DR manager with terrain-aware version
        self._dr_manager = DomainRandomizationManager(
            self, OpenDogeJoystickRoughDomainRandomizationProvider(
                base_body_mass=getattr(
                    getattr(self._dr_manager, "_provider", None),
                    "_base_body_mass", None
                ),
                base_geom_friction=getattr(
                    getattr(self._dr_manager, "_provider", None),
                    "_base_geom_friction", None
                ),
                ground_geom_id=getattr(
                    getattr(self._dr_manager, "_provider", None),
                    "_ground_geom_id", None
                ),
            )
        )
        self.feet_vel = np.zeros(
            (num_envs, len(cfg.sensor.feet_vel), 3), dtype=np.float32
        )
        # Contact timers for terrain-style rewards
        n_feet = len(cfg.sensor.feet_force)
        self._last_foot_contact = np.zeros((num_envs, n_feet), dtype=bool)
        self._current_air_time = np.zeros((num_envs, n_feet), dtype=np.float32)
        self._current_contact_time = np.zeros(
            (num_envs, n_feet), dtype=np.float32
        )
        self._last_air_time = np.zeros((num_envs, n_feet), dtype=np.float32)
        self._last_contact_time = np.zeros(
            (num_envs, n_feet), dtype=np.float32
        )
        self._first_foot_contact = np.zeros((num_envs, n_feet), dtype=bool)
        self._contact_timers_initialized = False
        init_height_scan_sensor(self, cfg.terrain_scan, cfg.asset.base_name)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 49, "critic": 52 + self._height_scan_dim}

    def reset(
        self, env_indices: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict]:
        env_ids = np.asarray(env_indices, dtype=np.int32)
        obs, info = super().reset(env_ids)
        self._reset_contact_timers(env_ids)
        return obs, info

    # ── observation ────────────────────────────────────────────────

    def _compute_obs(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        feet_phase: np.ndarray,
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        diff = dof_pos - self.default_angles
        # Actor: 49 dims — deployable, no height scan, no linvel
        policy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro) * 0.25
        policy_gravity = self._obs_noise(-gravity, noise_cfg.scale_gravity)
        policy_diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        policy_dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel) * 0.05
        last_actions = info.get(
            "current_actions", np.zeros_like(diff)
        )
        commands = info["commands"]
        obs = np.concatenate(
            [
                policy_gyro, policy_gravity, policy_diff, policy_dof_vel,
                last_actions, commands, feet_phase,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        # Critic: 52 + H dims — privileged linvel + height scan
        critic_base = np.concatenate(
            [
                gyro, -gravity, diff, dof_vel, last_actions,
                commands, feet_phase, linvel,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        critic = np.concatenate(
            [
                critic_base,
                height_scan_obs(
                    self, self._cfg.terrain_scan, critic_base.shape[0]
                ),
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        return {"obs": obs, "critic": critic}

    # ── reward ─────────────────────────────────────────────────────

    def _init_reward_functions(self):
        self._previous_actions = np.zeros(
            (self._num_envs, 12), dtype=np.float32
        )
        self._reward_fns: dict[str, Any] = {
            # Core tracking (from flat)
            "tracking_vx": rewards.tracking_vx,
            "tracking_vy": rewards.tracking_vy,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "cross_axis_suppression": rewards.cross_axis_suppression,
            "tracking_vel_linear": self._reward_tracking_vel_linear,
            # Regularization (from flat)
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "base_height": rewards.base_height,
            "action_rate": rewards.action_rate,
            "action_smooth": rewards.action_smooth,
            "similar_to_default": rewards.similar_to_default,
            "alive": rewards.alive,
            "dof_acc": rewards.dof_acc,
            "stand_still": self._reward_stand_still,
            "zero_command_stillness": rewards.zero_command_stillness,
            "torques": rewards.torques,
            "energy": rewards.energy,
            # Gait (from flat)
            "swing_feet_z": self._reward_swing_feet_z,
            "contact": self._reward_contact,
            "feet_air_time": self._reward_feet_air_time_rough,
            "joint_mirror": self._reward_joint_mirror,
            "hip_pos": self._reward_hip_pos,
            # Terrain-specific
            "undesired_contacts": self._reward_undesired_contacts,
            "feet_slide": self._reward_feet_slide,
            "feet_height_body": self._reward_feet_height_body,
            "feet_gait": self._reward_feet_gait,
            "feet_air_time_variance": self._reward_feet_air_time_variance,
            "feet_contact_without_cmd": self._reward_feet_contact_without_cmd,
            "upward": rewards.upward,
        }

    def _compute_reward(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> np.ndarray:
        cfg = self._reward_cfg
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            gravity=gravity,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            num_envs=self._num_envs,
            default_angles=self.default_angles,
            tracking_sigma=cfg.tracking_sigma,
            base_height_target=cfg.base_height_target,
            base_height=self._reward_base_height_values(),
        )
        return rewards.run_reward_dispatch(
            scales=cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )

    def _reward_base_height_values(
        self, num_obs: int | None = None
    ) -> np.ndarray:
        return base_height_from_scan(self, num_obs)

    # ── update_state ───────────────────────────────────────────────

    def update_state(self, state: NpEnvState) -> NpEnvState:
        # Adaptive gait frequency
        cmd_speed = np.linalg.norm(state.info["commands"][:, :3], axis=1)
        freq = np.clip(1.2 + 1.3 * cmd_speed / 0.6, 1.2, 2.5)
        self.phase = np.fmod(self.phase + self._cfg.ctrl_dt * freq, 1.0)
        self.feet_phase[:, 0] = self.phase
        self.feet_phase[:, 3] = self.phase
        self.feet_phase[:, 1] = (self.phase + 0.5) % 1
        self.feet_phase[:, 2] = (self.phase + 0.5) % 1

        state.info["previous_actions"] = state.info.get(
            "last_actions", self._previous_actions
        )

        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data("upvector")
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        self.feet_force[:, :, :] = 0
        for i in range(len(self._cfg.sensor.feet_force)):
            self.feet_force[:, i, :] = self._backend.get_sensor_data(
                self._cfg.sensor.feet_force[i]
            )
        for i in range(len(self._cfg.sensor.feet_pos)):
            self.feet_pos[:, i, :] = self._backend.get_sensor_data(
                self._cfg.sensor.feet_pos[i]
            )
        for i in range(len(self._cfg.sensor.feet_vel)):
            self.feet_vel[:, i, :] = self._backend.get_sensor_data(
                self._cfg.sensor.feet_vel[i]
            )

        # PD torque estimation
        target = (
            state.info["current_actions"] * self._cfg.control_config.action_scale
            + self.default_angles
        )
        state.info["torques"] = np.asarray(
            self._cfg.control_config.Kp * (target - dof_pos)
            - self._cfg.control_config.Kd * dof_vel,
            dtype=get_global_dtype(),
        )

        # heading command update
        if self._cfg.commands.heading_command:
            self._update_commands(state.info)

        cmd_override = getattr(self, "_command_override", None)
        if cmd_override is not None:
            state.info["commands"][:] = cmd_override.reshape(1, 3)

        self._update_contact_timers()
        terminated = self._compute_terminated(gravity)
        reward = self._compute_reward(
            state.info, linvel, gyro, gravity, dof_pos, dof_vel
        )
        obs = self._compute_obs(
            state.info, linvel, gyro, gravity, dof_pos, dof_vel,
            self.feet_phase,
        )
        truncated = self._compute_truncated(state)
        state = state.replace(
            obs=obs, reward=reward, terminated=terminated, truncated=truncated
        )
        done = state.terminated | state.truncated
        if np.any(done):
            done_indices = np.where(done)[0]
            stats = self._spawn.update_on_done(
                done_indices, self._backend.get_base_pos()[done_indices]
            )
            if stats:
                if "log" not in state.info:
                    state.info["log"] = {}
                for k, v in stats.items():
                    state.info["log"][f"terrain_curriculum/{k}"] = float(v)
        return state

    # ── termination / truncation ───────────────────────────────────

    def _compute_terminated(self, gravity: np.ndarray) -> np.ndarray:
        return np.asarray(gravity[:, 2] <= 0.5, dtype=bool)

    def _compute_truncated(self, state: NpEnvState) -> np.ndarray:
        truncated = super()._compute_truncated(state)
        if self._cfg.termination_config.terrain_out_of_bounds:
            terrain_scene = self._cfg.scene.terrain
            terrain_cfg = (
                terrain_scene.generator if terrain_scene is not None else None
            )
            np.logical_or(
                truncated,
                terrain_out_of_bounds(
                    self,
                    terrain_cfg,
                    float(
                        self._cfg.termination_config.terrain_distance_buffer
                    ),
                ),
                out=truncated,
            )
        return truncated

    # ── commands ───────────────────────────────────────────────────

    def _update_commands(self, info: dict) -> None:
        commands_arr = np.asarray(info["commands"], dtype=get_global_dtype())
        resampling_time = float(self._cfg.commands.resampling_time)
        if resampling_time > 0.0:
            interval_steps = max(
                int(round(resampling_time / self._cfg.ctrl_dt)), 1
            )
            steps = np.asarray(
                info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
            )
            resample_mask = (steps > 0) & ((steps % interval_steps) == 0)
            if np.any(resample_mask):
                num_resample = int(np.count_nonzero(resample_mask))
                low = np.asarray(
                    self._cfg.commands.vel_limit[0], dtype=get_global_dtype()
                )
                high = np.asarray(
                    self._cfg.commands.vel_limit[1], dtype=get_global_dtype()
                )
                sampled = np.random.uniform(
                    low=low, high=high, size=(num_resample, 3)
                ).astype(get_global_dtype())
                zero_small_xy_commands(sampled, threshold=0.03)
                commands_arr[resample_mask] = sampled
                if self._cfg.commands.heading_command:
                    heading_commands = self._ensure_heading_commands(
                        info, commands_arr.shape[0]
                    )
                    heading_commands[resample_mask] = sample_heading_commands(
                        self, num_resample
                    )
                    info["heading_commands"] = heading_commands

        if self._cfg.commands.heading_command:
            heading_commands = self._ensure_heading_commands(
                info, commands_arr.shape[0]
            )
            base_quat = np.asarray(
                self._backend.get_base_quat(), dtype=get_global_dtype()
            )
            if base_quat.shape[0] == commands_arr.shape[0]:
                apply_heading_yaw_feedback(
                    commands_arr, base_quat, heading_commands, stiffness=0.5
                )
        info["commands"] = commands_arr

    def _ensure_heading_commands(
        self, info: dict, num_obs: int
    ) -> np.ndarray:
        heading_commands = info.get("heading_commands")
        if heading_commands is None or np.asarray(heading_commands).shape != (
            num_obs,
        ):
            heading_commands = sample_heading_commands(self, num_obs)
        heading_commands = np.asarray(
            heading_commands, dtype=get_global_dtype()
        )
        info["heading_commands"] = heading_commands
        return heading_commands

    # ── raw height scan (for viser / debugging) ────────────────────

    def _raw_height_scan_obs(
        self, num_obs: int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        return raw_height_scan_obs(self, num_obs)

    # ── contact timer helpers ──────────────────────────────────────

    def _foot_contact_mask(self) -> np.ndarray:
        contact_force = np.linalg.norm(self.feet_force, axis=2)
        return np.asarray(
            contact_force > self._reward_cfg.contact_threshold, dtype=bool
        )

    def _reset_contact_timers(self, env_ids: np.ndarray) -> None:
        self._current_air_time[env_ids] = 0.0
        self._current_contact_time[env_ids] = 0.0
        self._last_air_time[env_ids] = 0.0
        self._last_contact_time[env_ids] = 0.0
        self._first_foot_contact[env_ids] = False
        self._last_foot_contact[env_ids] = self._foot_contact_mask()[env_ids]

    def _update_contact_timers(self) -> None:
        contact = self._foot_contact_mask()
        first_contact = contact & ~self._last_foot_contact
        first_air = ~contact & self._last_foot_contact
        self._first_foot_contact[:] = first_contact
        self._last_air_time[first_contact] = self._current_air_time[
            first_contact
        ]
        self._last_contact_time[first_air] = self._current_contact_time[
            first_air
        ]
        self._current_air_time[contact] = 0.0
        self._current_air_time[~contact] += self._cfg.ctrl_dt
        self._current_contact_time[~contact] = 0.0
        self._current_contact_time[contact] += self._cfg.ctrl_dt
        self._last_foot_contact[:] = contact

    # ── upright scale (gates rewards when robot is tilted) ─────────

    def _upright_scale(
        self, gravity: np.ndarray | None
    ) -> np.ndarray:
        return rewards.upright_scale(gravity, self._num_envs)

    # ── terrain-specific reward methods ────────────────────────────

    def _reward_feet_air_time_rough(
        self, ctx: RewardContext
    ) -> np.ndarray:
        cfg = self._reward_cfg
        reward = np.sum(
            (self._last_air_time - cfg.feet_air_time_threshold)
            * self._first_foot_contact,
            axis=1,
        )
        moving = np.linalg.norm(ctx.info["commands"], axis=1) > 0.1
        return np.asarray(
            reward * moving * self._upright_scale(ctx.gravity),
            dtype=get_global_dtype(),
        )

    def _reward_feet_air_time_variance(
        self, ctx: RewardContext
    ) -> np.ndarray:
        air_var = np.var(np.clip(self._last_air_time, 0.0, 0.5), axis=1)
        contact_var = np.var(
            np.clip(self._last_contact_time, 0.0, 0.5), axis=1
        )
        return np.asarray(
            (air_var + contact_var) * self._upright_scale(ctx.gravity),
            dtype=get_global_dtype(),
        )

    def _reward_feet_contact_without_cmd(
        self, ctx: RewardContext
    ) -> np.ndarray:
        reward = np.sum(self._first_foot_contact, axis=1)
        stopped = np.linalg.norm(ctx.info["commands"], axis=1) < 0.1
        return np.asarray(
            reward * stopped * self._upright_scale(ctx.gravity),
            dtype=get_global_dtype(),
        )

    def _reward_undesired_contacts(
        self, ctx: RewardContext
    ) -> np.ndarray:
        contacts = [
            _force_norm_columns(
                np.asarray(
                    self._backend.get_sensor_data(name),
                    dtype=get_global_dtype(),
                ),
                ctx.num_envs,
            )
            for name in self._cfg.sensor.undesired_contact
        ]
        if not contacts:
            return np.zeros(
                (ctx.num_envs,), dtype=get_global_dtype()
            )
        contact_force = np.concatenate(contacts, axis=1)
        contact_count = np.sum(
            contact_force > self._reward_cfg.contact_threshold, axis=1
        )
        return np.asarray(
            contact_count * self._upright_scale(ctx.gravity),
            dtype=get_global_dtype(),
        )

    def _relative_foot_vel_body(self) -> np.ndarray:
        base_quat = np.asarray(
            self._backend.get_base_quat(), dtype=get_global_dtype()
        )
        base_linvel = np.asarray(
            self._backend.get_sensor_data("global_linvel"),
            dtype=get_global_dtype(),
        )
        relative_vel = self.feet_vel - base_linvel[:, None, :]
        flat = relative_vel.reshape(
            self._num_envs * relative_vel.shape[1], 3
        )
        quat = np.repeat(base_quat, relative_vel.shape[1], axis=0)
        return np_quat_apply_inverse(quat, flat).reshape(relative_vel.shape)

    def _relative_foot_pos_body(self) -> np.ndarray:
        base_quat = np.asarray(
            self._backend.get_base_quat(), dtype=get_global_dtype()
        )
        base_pos = np.asarray(
            self._backend.get_base_pos(), dtype=get_global_dtype()
        )
        relative_pos = self.feet_pos - base_pos[:, None, :]
        flat = relative_pos.reshape(
            self._num_envs * relative_pos.shape[1], 3
        )
        quat = np.repeat(base_quat, relative_pos.shape[1], axis=0)
        return np_quat_apply_inverse(quat, flat).reshape(relative_pos.shape)

    def _reward_feet_slide(self, ctx: RewardContext) -> np.ndarray:
        foot_vel_body = self._relative_foot_vel_body()
        lateral_vel = np.linalg.norm(foot_vel_body[:, :, :2], axis=2)
        reward = np.sum(lateral_vel * self._foot_contact_mask(), axis=1)
        return np.asarray(
            reward * self._upright_scale(ctx.gravity),
            dtype=get_global_dtype(),
        )

    def _reward_feet_height_body(
        self, ctx: RewardContext
    ) -> np.ndarray:
        cfg = self._reward_cfg
        foot_pos_body = self._relative_foot_pos_body()
        foot_vel_body = self._relative_foot_vel_body()
        z_error = np.square(
            foot_pos_body[:, :, 2] - cfg.feet_height_body_target
        )
        velocity_tanh = np.tanh(
            cfg.feet_height_body_tanh_mult
            * np.linalg.norm(foot_vel_body[:, :, :2], axis=2)
        )
        moving = np.linalg.norm(ctx.info["commands"], axis=1) > 0.1
        reward = np.sum(z_error * velocity_tanh, axis=1)
        return np.asarray(
            reward * moving * self._upright_scale(ctx.gravity),
            dtype=get_global_dtype(),
        )

    def _reward_feet_gait(self, ctx: RewardContext) -> np.ndarray:
        cfg = self._reward_cfg
        command_norm = np.linalg.norm(ctx.info["commands"], axis=1)
        body_vel = np.linalg.norm(ctx.linvel[:, :2], axis=1)
        enabled = (command_norm > cfg.feet_gait_command_threshold) | (
            body_vel > cfg.feet_gait_velocity_threshold
        )
        air = self._current_air_time
        contact = self._current_contact_time
        sync_fl_rr = _gait_sync_reward(
            air, contact, FRONT_LEFT, REAR_RIGHT,
            cfg.feet_gait_std, cfg.feet_gait_max_err,
        )
        sync_fr_rl = _gait_sync_reward(
            air, contact, FRONT_RIGHT, REAR_LEFT,
            cfg.feet_gait_std, cfg.feet_gait_max_err,
        )
        async_fl_fr = _gait_async_reward(
            air, contact, FRONT_LEFT, FRONT_RIGHT,
            cfg.feet_gait_std, cfg.feet_gait_max_err,
        )
        async_rr_rl = _gait_async_reward(
            air, contact, REAR_RIGHT, REAR_LEFT,
            cfg.feet_gait_std, cfg.feet_gait_max_err,
        )
        async_fl_rl = _gait_async_reward(
            air, contact, FRONT_LEFT, REAR_LEFT,
            cfg.feet_gait_std, cfg.feet_gait_max_err,
        )
        async_fr_rr = _gait_async_reward(
            air, contact, FRONT_RIGHT, REAR_RIGHT,
            cfg.feet_gait_std, cfg.feet_gait_max_err,
        )
        reward = (
            sync_fl_rr * sync_fr_rl
            * async_fl_fr * async_rr_rl
            * async_fl_rl * async_fr_rr
        )
        return np.asarray(
            reward * enabled * self._upright_scale(ctx.gravity),
            dtype=get_global_dtype(),
        )

    def _reward_stand_still(self, ctx: RewardContext) -> np.ndarray:
        return rewards.stand_still(
            ctx,
            command_threshold=self._reward_cfg.stand_still_command_threshold,
        ) * self._upright_scale(ctx.gravity)


# ── helpers ─────────────────────────────────────────────────────────


def _force_norm_columns(force: np.ndarray, num_envs: int) -> np.ndarray:
    force = np.asarray(force, dtype=get_global_dtype()).reshape(num_envs, -1)
    if force.shape[1] == 0:
        return force
    if force.shape[1] % 3 == 0:
        return np.linalg.norm(force.reshape(num_envs, -1, 3), axis=2)
    return np.abs(force)


def _gait_sync_reward(
    air: np.ndarray,
    contact: np.ndarray,
    foot_0: int,
    foot_1: int,
    std: float,
    max_err: float,
) -> np.ndarray:
    se_air = np.clip(
        np.square(air[:, foot_0] - air[:, foot_1]), 0.0, max_err**2
    )
    se_contact = np.clip(
        np.square(contact[:, foot_0] - contact[:, foot_1]),
        0.0,
        max_err**2,
    )
    return np.exp(-(se_air + se_contact) / std)


def _gait_async_reward(
    air: np.ndarray,
    contact: np.ndarray,
    foot_0: int,
    foot_1: int,
    std: float,
    max_err: float,
) -> np.ndarray:
    se_act_0 = np.clip(
        np.square(air[:, foot_0] - contact[:, foot_1]),
        0.0,
        max_err**2,
    )
    se_act_1 = np.clip(
        np.square(contact[:, foot_0] - air[:, foot_1]),
        0.0,
        max_err**2,
    )
    return np.exp(-(se_act_0 + se_act_1) / std)


# Register for motrix backend as well (even if not yet validated)
registry.register_env(
    "OpenDogeJoystickRough", OpenDogeJoystickRoughEnv, sim_backend="motrix"
)
