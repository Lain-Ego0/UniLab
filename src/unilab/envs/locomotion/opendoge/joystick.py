from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.base import Sensor
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.common.terrain_spawn import (
    TerrainCurriculumCfg,
    TerrainSpawnManager,
)
from unilab.envs.locomotion.opendoge.base import OpenDogeBaseCfg, OpenDogeBaseEnv


@dataclass
class InitState:
    pos = [0.0, 0.0, 0.17]


@dataclass
class OpenDogeDomainRandConfig(DomainRandConfig):
    randomize_kp: bool = True
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])

    randomize_kd: bool = True
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])


@dataclass
class RewardConfig:
    scales: dict[str, float]
    tracking_sigma: float
    base_height_target: float
    target_foot_height: float = 0.03


@dataclass
class JoystickSensor(Sensor):
    local_linvel = "local_linvel"
    gyro = "gyro"
    feet_force = ["FL_foot_contact", "FR_foot_contact", "RL_foot_contact", "RR_foot_contact"]
    feet_pos = ["FL_pos", "FR_pos", "RL_pos", "RR_pos"]


@registry.envcfg("OpenDogeJoystickFlat")
@dataclass
class OpenDogeJoystickCfg(OpenDogeBaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "opendoge" / "scene_flat.xml")
        )
    )
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: RewardConfig | None = None
    sensor: JoystickSensor = field(default_factory=JoystickSensor)
    domain_rand: OpenDogeDomainRandConfig = field(default_factory=OpenDogeDomainRandConfig)
    terrain_curriculum: TerrainCurriculumCfg = field(default_factory=TerrainCurriculumCfg)


class OpenDogeJoystickDomainRandomizationProvider(LocomotionDRProvider):
    def __init__(
        self,
        *,
        base_body_mass: np.ndarray | None = None,
        base_geom_friction: np.ndarray | None = None,
        ground_geom_id: int | None = None,
    ):
        self._base_body_mass = base_body_mass
        self._base_geom_friction = base_geom_friction
        self._ground_geom_id = ground_geom_id

    def _get_reset_randomization_baselines(
        self, env: Any
    ) -> tuple[np.ndarray | None, np.ndarray | None, int | None, np.ndarray | None]:
        return self._base_body_mass, self._base_geom_friction, self._ground_geom_id, None
    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        commands = super()._sample_commands(env, num_reset)
        # Inject pure single-axis commands so the policy sees "vx-only / vy-only / vyaw-only"
        # scenarios during training (improves cross-axis suppression).
        pure_prob = getattr(env.cfg.commands, "pure_axis_prob", 0.15)
        if pure_prob > 0.0:
            pure_mask = np.random.uniform(size=(num_reset,)) < min(pure_prob, 1.0)
            if np.any(pure_mask):
                n_pure = int(np.sum(pure_mask))
                axis_choice = np.random.randint(0, 3, size=(n_pure,))
                pure_cmd = np.zeros((n_pure, 3), dtype=commands.dtype)
                vel_limit = env.cfg.commands.vel_limit
                for i in range(n_pure):
                    ax = axis_choice[i]
                    lo, hi = vel_limit[0][ax], vel_limit[1][ax]
                    # Ensure non-trivial magnitude (at least 20% of range away from zero)
                    half_range = max(abs(lo), abs(hi)) * 0.2
                    if lo >= 0:
                        val = np.random.uniform(max(lo, half_range), hi)
                    elif hi <= 0:
                        val = np.random.uniform(lo, min(hi, -half_range))
                    else:
                        # Range spans zero — sample from [half_range, hi] or [lo, -half_range]
                        if np.random.uniform() < 0.5:
                            val = np.random.uniform(max(lo, half_range), hi) if hi > half_range else np.random.uniform(lo, -half_range)
                        else:
                            val = np.random.uniform(lo, -half_range) if lo < -half_range else np.random.uniform(half_range, hi)
                    pure_cmd[i, ax] = val
                commands[pure_mask] = pure_cmd
        # Zero tiny commands to avoid noise drift
        mask = np.linalg.norm(commands, axis=1) < 0.03
        commands[mask] = 0.0
        # Force a fraction of envs to stand still (teach zero-speed behaviour)
        standing_prob = getattr(env.cfg.commands, "rel_standing_envs", 0.1)
        if standing_prob > 0.0:
            standing = np.random.uniform(size=(num_reset,)) < min(standing_prob, 1.0)
            commands[standing] = 0.0
        return commands

    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: Any,
        info_updates: Any,
        linvel: Any,
        gyro: Any,
        gravity: Any,
        dof_pos: Any,
        dof_vel: Any,
    ) -> dict[str, np.ndarray]:
        env._reset_contact_timers(env_ids)
        return cast(
            dict[str, np.ndarray],
            env._compute_obs(
                info_updates, linvel, gyro, gravity, dof_pos, dof_vel, env.feet_phase[env_ids]
            ),
        )


HIP_YAWS = [0, 3, 6, 9]  # FL, FR, RL, RR hip abduction joints


@registry.env("OpenDogeJoystickFlat", sim_backend="mujoco")
class OpenDogeWalkTask(OpenDogeBaseEnv):
    _cfg: OpenDogeJoystickCfg

    def __init__(self, cfg: OpenDogeJoystickCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")

        self._scene_terrain_origins: np.ndarray | None = None
        scene_cfg = cfg.scene
        terrain_generator = scene_cfg.terrain.generator if scene_cfg.terrain is not None else None

        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            position_actuator_gains={"kp": cfg.control_config.Kp, "kd": cfg.control_config.Kd},
            motrix_max_iterations=cfg.motrix_max_iterations,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
        )
        self._terrain_surface_sampler = getattr(backend, "terrain_surface_sampler", None)
        self._terrain_surface_sample_height = self._resolve_terrain_surface_sample_height()
        terrain_origins = getattr(backend, "terrain_origins", None)
        if terrain_origins is not None:
            self._scene_terrain_origins = terrain_origins
        super().__init__(cfg, backend, num_envs)
        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config
        self._init_reward_functions()
        base_body_mass: np.ndarray | None = None
        if cfg.domain_rand.randomize_base_mass:
            base_body_mass = backend.get_body_mass()

        base_geom_friction: np.ndarray | None = None
        ground_geom_id: int | None = None
        if cfg.domain_rand.randomize_ground_friction:
            base_geom_friction = backend.get_geom_friction()
            ground_geom_id = backend.get_geom_id(cfg.asset.ground)

        self._init_domain_randomization(
            OpenDogeJoystickDomainRandomizationProvider(
                base_body_mass=base_body_mass,
                base_geom_friction=base_geom_friction,
                ground_geom_id=ground_geom_id,
            )
        )
        if self._scene_terrain_origins is not None and terrain_generator is not None:
            self._spawn = TerrainSpawnManager(
                num_envs,
                self._scene_terrain_origins,
                cell_size=float(terrain_generator.size[0]),
                cfg=cfg.terrain_curriculum,
                terrain_surface_sampler=self._terrain_surface_sampler,
            )
        self.phase = np.zeros((num_envs,), dtype=np.float32)
        self.feet_phase = np.zeros((num_envs, len(cfg.sensor.feet_force)), dtype=np.float32)
        self.gait_frequency = 2
        self.feet_force = np.zeros((num_envs, len(cfg.sensor.feet_force), 3), dtype=np.float32)
        self.feet_pos = np.zeros((num_envs, len(cfg.sensor.feet_pos), 3), dtype=np.float32)
        # Contact timers for feet_air_time reward
        n_feet = len(cfg.sensor.feet_force)
        self._last_foot_contact = np.zeros((num_envs, n_feet), dtype=bool)
        self._current_air_time = np.zeros((num_envs, n_feet), dtype=np.float32)
        self._last_air_time = np.zeros((num_envs, n_feet), dtype=np.float32)
        self._first_foot_contact = np.zeros((num_envs, n_feet), dtype=bool)
        self._contact_timers_initialized = False

    def get_playback_model(self, env_index: int | None = None) -> Any:
        return super().get_playback_model(env_index)

    def _resolve_terrain_surface_sample_height(
        self,
    ) -> Callable[[np.ndarray], np.ndarray] | None:
        sampler = self._terrain_surface_sampler
        if sampler is None:
            return None

        sample_height = getattr(sampler, "sample_height", None)
        if not callable(sample_height):
            raise TypeError("terrain_surface_sampler must expose sample_height(xy)")
        return cast(Callable[[np.ndarray], np.ndarray], sample_height)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # actor: gyro(3) + gravity(3) + diff(12) + dof_vel(12) + action(12) + cmd(3) + phase(4) = 49
        # critic: same + linvel(3) = 52 (privileged info for value estimation)
        return {"obs": 49, "critic": 52}

    def _init_reward_functions(self):
        self._previous_actions = np.zeros((self._num_envs, 12), dtype=np.float32)
        self._reward_fns: dict[str, Any] = {
            "tracking_vx": rewards.tracking_vx,
            "tracking_vy": rewards.tracking_vy,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "cross_axis_suppression": rewards.cross_axis_suppression,
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "base_height": rewards.base_height,
            "action_rate": rewards.action_rate,
            "action_smooth": rewards.action_smooth,
            "similar_to_default": rewards.similar_to_default,
            "alive": rewards.alive,
            "dof_acc": rewards.dof_acc,
            "stand_still": rewards.stand_still,
            "zero_command_stillness": rewards.zero_command_stillness,
            "torques": rewards.torques,
            "energy": rewards.energy,
            "swing_feet_z": self._reward_swing_feet_z,
            "contact": self._reward_contact,
            "foot_drag": self._reward_foot_drag,
            "feet_air_time": self._reward_feet_air_time,
            "joint_mirror": self._reward_joint_mirror,
            "hip_pos": self._reward_hip_pos,
        }

    def update_state(self, state: NpEnvState) -> NpEnvState:
        # Adaptive gait frequency: faster commands → higher step rate
        cmd_speed = np.linalg.norm(state.info["commands"][:, :3], axis=1)
        freq = np.clip(1.2 + 1.3 * cmd_speed / 0.6, 1.2, 2.5)
        self.phase = np.fmod(self.phase + self._cfg.ctrl_dt * freq, 1.0)
        self.feet_phase[:, 0] = self.phase
        self.feet_phase[:, 3] = self.phase

        self.feet_phase[:, 1] = (self.phase + 0.5) % 1
        self.feet_phase[:, 2] = (self.phase + 0.5) % 1

        # Track previous actions for action_smooth (second-order penalty)
        state.info["previous_actions"] = state.info.get("last_actions", self._previous_actions)

        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data("upvector")
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        self.feet_force[:, :, :] = 0
        for i in range(len(self._cfg.sensor.feet_force)):
            self.feet_force[:, i, :] = self._backend.get_sensor_data(self._cfg.sensor.feet_force[i])
        for i in range(len(self._cfg.sensor.feet_pos)):
            self.feet_pos[:, i, :] = self._backend.get_sensor_data(self._cfg.sensor.feet_pos[i])

        # Estimate PD torques: tau = Kp*(target - q) - Kd*qdot
        target = (
            state.info["current_actions"] * self._cfg.control_config.action_scale
            + self.default_angles
        )
        state.info["torques"] = np.asarray(
            self._cfg.control_config.Kp * (target - dof_pos)
            - self._cfg.control_config.Kd * dof_vel,
            dtype=get_global_dtype(),
        )

        # Apply command override (set by viser/external tooling)
        cmd_override = getattr(self, "_command_override", None)
        if cmd_override is not None:
            state.info["commands"][:] = cmd_override.reshape(1, 3)

        self._update_contact_timers()
        terminated = gravity[:, 2] <= 0.5
        reward = self._compute_reward(state.info, linvel, gyro, dof_pos, dof_vel)
        obs = self._compute_obs(
            state.info, linvel, gyro, gravity, dof_pos, dof_vel, self.feet_phase
        )
        state = state.replace(obs=obs, reward=reward, terminated=terminated)
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

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel, feet_phase
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        diff = dof_pos - self.default_angles
        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        noisy_dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
        command = info["commands"]
        last_actions = info.get("current_actions", np.zeros_like(diff))
        # Actor obs: 49 dims — no linvel (not deployable on real hardware).
        # gyro(3) + neg_gravity(3) + diff(12) + dof_vel(12) + action(12) + cmd(3) + phase(4)
        obs = np.concatenate(
            [
                noisy_gyro,
                -noisy_gravity,
                noisy_diff,
                noisy_dof_vel,
                last_actions,
                command,
                feet_phase,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        # Critic obs: 52 dims — includes privileged linvel for better value estimation.
        critic = np.concatenate(
            [gyro, -gravity, diff, dof_vel, last_actions, command, feet_phase, linvel],
            axis=1,
            dtype=get_global_dtype(),
        )
        return {"obs": obs, "critic": critic}

    def _compute_reward(self, info: dict, linvel, gyro, dof_pos, dof_vel) -> np.ndarray:
        cfg = self._reward_cfg
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
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

    # ── reward functions (robot-specific) ────────────────────────────

    def _reward_base_height_values(self) -> np.ndarray:
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        sample_height = self._terrain_surface_sample_height
        if sample_height is None:
            return np.asarray(base_pos[:, 2], dtype=get_global_dtype())

        surface = np.asarray(sample_height(base_pos[:, :2]), dtype=get_global_dtype())
        return np.asarray(base_pos[:, 2] - surface, dtype=get_global_dtype())

    def _reward_swing_feet_z(self, ctx: RewardContext) -> np.ndarray:
        is_swing = self.feet_phase >= 0.6
        # Adaptive target: larger command (vx+vy+vyaw) → higher step
        cmd_mag = np.linalg.norm(ctx.info["commands"], axis=1)  # (N,) 3D
        # Disable swing height reward at very low speed; ramp in over [0.03, 0.08]
        swing_active = np.clip((cmd_mag - 0.03) / 0.05, 0.0, 1.0)
        target_height = np.clip(0.015 + 0.105 * cmd_mag, 0.015, 0.12)
        height_error = np.square(self.feet_pos[:, :, 2] - target_height[:, None])
        swing_rew = np.exp(-height_error / 0.015) * is_swing * swing_active[:, None]
        reward: np.ndarray = np.sum(swing_rew, axis=1) / len(self._cfg.sensor.feet_pos)
        return reward

    def _reward_foot_drag(self, ctx: RewardContext) -> np.ndarray:
        foot_pos = self.get_foot_pos()
        foot_heights = foot_pos[..., 2]
        foot_contact = self.get_foot_contact()
        is_swing = foot_contact < 0.5
        safe_height = self._reward_cfg.target_foot_height / 2.0
        height_error = np.clip(safe_height - foot_heights, 0.0, None)
        error = np.square(height_error) * is_swing
        drag_penalty: np.ndarray = np.sum(error, axis=1)
        return drag_penalty

    def _reward_contact(self, ctx: RewardContext) -> np.ndarray:
        contact = self.feet_force[:, :, 2] > 0.1
        res = np.zeros(self._num_envs, dtype=np.float32)
        cmd_mag = np.linalg.norm(ctx.info["commands"], axis=1)  # (N,) 3D
        for i in range(len(self._cfg.sensor.feet_force)):
            # At very low speed, expect all feet in contact (no swing phase)
            is_contact = (self.feet_phase[:, i] < 0.6) | (cmd_mag < 0.03)
            res += (contact[:, i] == is_contact).astype(np.float32)
        return res / len(self._cfg.sensor.feet_force)

    def _update_contact_timers(self) -> None:
        """Track per-foot air/contact durations for feet_air_time reward."""
        contact = self.feet_force[:, :, 2] > 0.1
        if not self._contact_timers_initialized:
            # First call: seed with current contact to avoid spurious transitions
            self._last_foot_contact[:] = contact
            self._contact_timers_initialized = True
            return
        first_contact = contact & ~self._last_foot_contact
        self._first_foot_contact[:] = first_contact
        # Record air time at touchdown
        self._last_air_time[first_contact] = self._current_air_time[first_contact]
        # Reset air time when in contact; increment when in air
        self._current_air_time[contact] = 0.0
        self._current_air_time[~contact] += self._cfg.ctrl_dt
        self._last_foot_contact[:] = contact

    def _reset_contact_timers(self, env_ids: np.ndarray) -> None:
        """Reset per-foot contact timers for envs that just restarted."""
        self._current_air_time[env_ids] = 0.0
        self._last_air_time[env_ids] = 0.0
        self._first_foot_contact[env_ids] = False
        # Seed with current contact to avoid transition spikes on first post-reset step
        contact = self.feet_force[env_ids, :, 2] > 0.1
        self._last_foot_contact[env_ids] = contact

    def _reward_feet_air_time(self, ctx: RewardContext) -> np.ndarray:
        """Reward proper flight phase duration during swing (capped at 0.5s).

        Only active when robot is commanded to move (cmd_mag > 0.05).
        """
        max_air_time = 0.25
        air_time_norm = np.clip(self._last_air_time, 0.0, max_air_time) / max_air_time
        reward = np.sum(air_time_norm * self._first_foot_contact, axis=1)
        moving = np.linalg.norm(ctx.info["commands"], axis=1) > 0.05
        return np.asarray(reward * moving, dtype=get_global_dtype())

    def _reward_joint_mirror(self, ctx: RewardContext) -> np.ndarray:
        """Penalize left-right joint asymmetry between diagonal trot pairs.

        OpenDoge DOF order (12): FL(0:3), FR(3:6), RL(6:9), RR(9:12).
        Trot gait: FL+RR in phase, FR+RL in phase — compare within same-phase pairs.
        """
        fl_fr = ctx.dof_pos[:, 0:3] - ctx.dof_pos[:, 3:6]
        rl_rr = ctx.dof_pos[:, 6:9] - ctx.dof_pos[:, 9:12]
        mirror = 0.5 * (np.sum(np.square(fl_fr), axis=1) + np.sum(np.square(rl_rr), axis=1))
        return np.asarray(mirror, dtype=get_global_dtype())

    def _reward_hip_pos(self, ctx: RewardContext) -> np.ndarray:
        """Penalize hip abduction (yaw) joints deviating from default angles.

        Only constrains the 4 hip_yaw joints (indices 0,3,6,9) — does not affect
        thigh/calf motion, so the policy can still use full leg swing for gait.
        """
        diff = ctx.dof_pos[:, HIP_YAWS] - self.default_angles[HIP_YAWS]
        return np.asarray(np.sum(np.square(diff), axis=1), dtype=get_global_dtype())
