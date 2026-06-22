"""Smoke test for OpenDoge environment integration."""

import mujoco
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.opendoge.joystick import (
    OpenDogeJoystickCfg,
    OpenDogeWalkTask,
    RewardConfig,
)


def test_opendoge_registry():
    """OpenDogeJoystickFlat is registered after ensure_registries."""
    registry.ensure_registries()
    meta = registry._envs.get("OpenDogeJoystickFlat")
    assert meta is not None, "OpenDogeJoystickFlat not found in registry"
    assert meta.env_cfg_cls.__name__ == "OpenDogeJoystickCfg"


def test_opendoge_xml_loads():
    """opendoge.xml loads in MuJoCo without errors."""
    model_path = str(ASSETS_ROOT_PATH / "robots" / "opendoge" / "opendoge.xml")
    m = mujoco.MjModel.from_xml_path(model_path)
    assert m.nbody == 14
    assert m.nu == 12  # 12 actuators
    names = [m.actuator(i).name for i in range(m.nu)]
    assert "FL_hip" in names
    assert "RR_calf" in names


def test_scene_flat_loads_with_keyframe():
    """scene_flat.xml loads and keyframe applies correctly."""
    scene_path = str(ASSETS_ROOT_PATH / "robots" / "opendoge" / "scene_flat.xml")
    m = mujoco.MjModel.from_xml_path(scene_path)
    d = mujoco.MjData(m)

    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert key_id >= 0, "Keyframe 'home' not found"

    mujoco.mj_resetDataKeyframe(m, d, key_id)
    mujoco.mj_forward(m, d)

    assert d.qpos[2] == 0.17, f"Expected base z=0.17, got {d.qpos[2]}"
    np.testing.assert_allclose(d.qpos[7:10], [0.0, 0.5, -1.3], atol=1e-6)


def test_backend_keyframe_qpos():
    """Backend returns correct keyframe qpos."""
    scene_cfg = SceneCfg(
        model_file=str(ASSETS_ROOT_PATH / "robots" / "opendoge" / "scene_flat.xml")
    )
    backend = create_backend("mujoco", scene_cfg, num_envs=1, sim_dt=0.01)
    qpos = backend.get_keyframe_qpos("home")
    assert len(qpos) == 19
    np.testing.assert_allclose(
        qpos[-12:], [0.0, 0.5, -1.3, 0.0, 0.5, -1.3, 0.0, 0.7, -1.3, 0.0, 0.7, -1.3], atol=1e-6
    )


def test_env_instantiation():
    """Full env instantiation and reset works."""
    registry.ensure_registries()

    reward_cfg = RewardConfig(
        scales={
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -5.0,
            "ang_vel_xy": -0.1,
            "base_height": -100.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "contact": 0.24,
            "swing_feet_z": 4.0,
        },
        tracking_sigma=0.25,
        base_height_target=0.25,
    )

    cfg = OpenDogeJoystickCfg(
        reward_config=reward_cfg,
    )

    env = OpenDogeWalkTask(cfg, num_envs=4, backend_type="mujoco")

    # Check obs spec
    spec = env.obs_groups_spec
    assert spec == {"obs": 49, "critic": 52}, f"Unexpected obs spec: {spec}"

    # Check default angles are from keyframe
    np.testing.assert_allclose(
        env.default_angles,
        [0.0, 0.5, -1.3, 0.0, 0.5, -1.3, 0.0, 0.7, -1.3, 0.0, 0.7, -1.3],
        atol=1e-6,
    )

    # Check action space
    assert env.action_space.shape == (12,)

    # Reset and check obs
    obs, info = env.reset(np.array([0, 1, 2, 3]))
    assert set(obs.keys()) == {"obs", "critic"}
    assert obs["obs"].shape == (4, 49)
    assert obs["critic"].shape == (4, 52)
    assert "commands" in info
