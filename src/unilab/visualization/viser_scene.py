# pyright: reportMissingImports=false
"""MuJoCo-to-viser scene adapter for interactive web-based 3D visualization.

This module renders MuJoCo scenes via a viser web server, providing browser-based
interactive 3D viewing without requiring a local display or GLFW.  It is gated
behind the ``viser`` optional-dependency group and is **not** imported by default.

Usage (from ``scripts/play_viser.py``)::

    from unilab.visualization.viser_scene import MujocoViserScene, VISER_AVAILABLE
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

try:
    import trimesh
    import viser

    VISER_AVAILABLE = True
except ImportError:
    VISER_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Rotation helpers (pure numpy, no scipy dependency)                          #
# --------------------------------------------------------------------------- #


def _rotmat_to_wxyz(mat: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a (w, x, y, z) quaternion."""
    m = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return (float(w), float(x), float(y), float(z))


# --------------------------------------------------------------------------- #
# Geometry extraction helpers                                                 #
# --------------------------------------------------------------------------- #


def _rgba_to_color(rgba: np.ndarray) -> tuple[int, int, int]:
    """Convert MuJoCo float RGBA [0,1] to viser int RGB [0,255]."""
    return (
        int(np.clip(rgba[0] * 255, 0, 255)),
        int(np.clip(rgba[1] * 255, 0, 255)),
        int(np.clip(rgba[2] * 255, 0, 255)),
    )


def _rgba_to_opacity(rgba: np.ndarray) -> float:
    return float(np.clip(rgba[3], 0.0, 1.0))


def _extract_mesh(model: mujoco.MjModel, geom_dataid: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract vertices and faces for a MuJoCo mesh geom."""
    vert_adr = model.mesh_vertadr[geom_dataid]
    vert_num = model.mesh_vertnum[geom_dataid]
    face_adr = model.mesh_faceadr[geom_dataid]
    face_num = model.mesh_facenum[geom_dataid]

    vertices = model.mesh_vert[vert_adr : vert_adr + vert_num].copy()
    faces = model.mesh_face[face_adr : face_adr + face_num].copy()
    return vertices, faces


def build_visible_env_indices(num_envs: int, visible_envs: int) -> np.ndarray:
    """Select a stable subset of env indices spread across the full batch.

    Args:
        num_envs: Total number of runtime environments.
        visible_envs: Number of env slots exposed in the viewer.

    Returns:
        A monotonically increasing array of runtime env indices.
    """
    if visible_envs <= 0:
        raise ValueError(f"visible_envs must be positive, got {visible_envs}")
    if visible_envs >= num_envs:
        return np.arange(num_envs, dtype=np.int32)
    return np.floor(np.linspace(0, num_envs, visible_envs, endpoint=False)).astype(np.int32)


# --------------------------------------------------------------------------- #
# MujocoViserScene                                                           #
# --------------------------------------------------------------------------- #


class MujocoViserScene:
    """Bridges a ``mujoco.MjModel`` to a ``viser.ViserServer`` scene graph.

    Call :meth:`build` once to populate the scene with geometry handles, then
    call :meth:`update` each frame to sync body transforms from ``MjData``.
    """

    def __init__(
        self,
        server: Any,
        model: mujoco.MjModel,
        *,
        name_prefix: str = "/mujoco",
        position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        render_plane: bool = True,
        terrain_cfg: Any | None = None,
    ) -> None:
        if not VISER_AVAILABLE:
            raise ImportError("viser is not installed. Install with: uv sync --extra viser")
        self._server: viser.ViserServer = server
        self._model = model
        self._name_prefix = name_prefix.rstrip("/") or "/mujoco"
        self._position_offset = np.asarray(position_offset, dtype=np.float64)
        self._render_plane = bool(render_plane)
        self._terrain_cfg = terrain_cfg
        self._handles: dict[int, Any] = {}
        self._terrain_handle: Any | None = None
        self._build()
        if self._terrain_cfg is not None:
            self._build_terrain_mesh()

    def reset(
        self,
        model: mujoco.MjModel,
        *,
        position_offset: tuple[float, float, float] | None = None,
        render_plane: bool | None = None,
    ) -> None:
        """Rebuild the viser scene for a new MuJoCo model."""
        self.close()
        self._model = model
        if position_offset is not None:
            self._position_offset = np.asarray(position_offset, dtype=np.float64)
        if render_plane is not None:
            self._render_plane = bool(render_plane)
        self._build()
        if self._terrain_cfg is not None:
            self._build_terrain_mesh()

    def close(self) -> None:
        """Remove all scene handles owned by this adapter."""
        for handle in self._handles.values():
            handle.remove()
        self._handles.clear()
        if self._terrain_handle is not None:
            self._terrain_handle.remove()
            self._terrain_handle = None

    # ------------------------------------------------------------------ #
    # Terrain mesh                                                         #
    # ------------------------------------------------------------------ #

    def _build_terrain_mesh(self) -> None:
        """Generate a triangle mesh from the training terrain config.

        Uses the exact same :class:`TerrainGenerator` that produced the
        heightfield during training, so the visual surface matches the
        physics terrain bit-for-bit.
        """
        from unilab.terrains import TerrainGenerator

        gt = TerrainGenerator(self._terrain_cfg).generate()
        heights = gt.heights_yx  # (nrow, ncol) world-space Z, row-major
        nrow, ncol = heights.shape
        hs = gt.horizontal_scale

        # --- vertices ---------------------------------------------------
        x = np.linspace(-gt.size[0] / 2, gt.size[0] / 2, ncol, dtype=np.float64)
        y = np.linspace(-gt.size[1] / 2, gt.size[1] / 2, nrow, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        zz = np.asarray(heights, dtype=np.float64)
        verts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

        # --- faces (CCW from +Z) ----------------------------------------
        max_samples = 150
        stride_r = max(1, nrow // max_samples)
        stride_c = max(1, ncol // max_samples)
        if stride_r > 1 or stride_c > 1:
            verts = verts.reshape(nrow, ncol, 3)[::stride_r, ::stride_c, :]
            verts = verts.reshape(-1, 3)
            nrow_s = (nrow + stride_r - 1) // stride_r
            ncol_s = (ncol + stride_c - 1) // stride_c
        else:
            nrow_s, ncol_s = nrow, ncol

        faces: list[tuple[int, int, int]] = []
        for r in range(nrow_s - 1):
            for c in range(ncol_s - 1):
                a = r * ncol_s + c
                b = a + 1
                d = a + ncol_s
                e = d + 1
                faces.append((a, b, d))
                faces.append((b, e, d))

        self._terrain_handle = self._server.scene.add_mesh_simple(
            f"{self._name_prefix}/terrain",
            vertices=verts.astype(np.float32),
            faces=np.array(faces, dtype=np.int32),
            color=(0.55, 0.55, 0.55),
            opacity=1.0,
        )

    # ------------------------------------------------------------------ #
    # Scene construction                                                  #
    # ------------------------------------------------------------------ #

    def _build(self) -> None:
        """Create viser scene nodes for every MuJoCo geom."""
        model = self._model
        server = self._server

        server.scene.set_up_direction("+z")

        for i in range(model.ngeom):
            geom_type = model.geom_type[i]
            size = model.geom_size[i]
            rgba = model.geom_rgba[i]
            color = _rgba_to_color(rgba)
            opacity = _rgba_to_opacity(rgba)
            name = f"{self._name_prefix}/geom/{i}"

            handle: Any | None = None

            if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
                if not self._render_plane:
                    continue
                # Render ground plane as a grid
                plane_size = float(size[0]) if size[0] > 0 else 10.0
                handle = server.scene.add_grid(
                    name,
                    width=plane_size * 2,
                    height=plane_size * 2,
                    cell_size=0.5,
                )

            elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                handle = server.scene.add_icosphere(
                    name,
                    radius=float(size[0]),
                    color=color,
                    opacity=opacity,
                )

            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                half_len = float(size[1])
                radius = float(size[0])
                mesh = trimesh.creation.capsule(height=half_len * 2, radius=radius)
                handle = server.scene.add_mesh_trimesh(name, mesh=mesh)
                # Manually set color since trimesh mesh may not carry it
                if hasattr(handle, "color"):
                    handle.color = color

            elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                # Use a unit sphere mesh scaled non-uniformly
                mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
                mesh.vertices *= np.array([float(size[0]), float(size[1]), float(size[2])])
                handle = server.scene.add_mesh_trimesh(name, mesh=mesh)

            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                handle = server.scene.add_cylinder(
                    name,
                    radius=float(size[0]),
                    height=float(size[1]) * 2,
                    color=color,
                    opacity=opacity,
                )

            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                handle = server.scene.add_box(
                    name,
                    dimensions=(
                        float(size[0]) * 2,
                        float(size[1]) * 2,
                        float(size[2]) * 2,
                    ),
                    color=color,
                    opacity=opacity,
                )

            elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                dataid = model.geom_dataid[i]
                if dataid >= 0:
                    vertices, faces = _extract_mesh(model, dataid)
                    handle = server.scene.add_mesh_simple(
                        name,
                        vertices=vertices.astype(np.float32),
                        faces=faces.astype(np.int32),
                        color=color,
                        opacity=opacity,
                    )

            elif geom_type == mujoco.mjtGeom.mjGEOM_HFIELD:
                # Hfield terrain is rendered separately via the terrain generator
                # config (see _build_terrain_mesh).  Skip here so the old
                # per-geom loop does not try to handle it.
                continue

            if handle is not None:
                self._handles[i] = handle

    # ------------------------------------------------------------------ #
    # Per-frame update                                                    #
    # ------------------------------------------------------------------ #

    def update(self, data: mujoco.MjData) -> None:
        """Sync all geom transforms from *data* into the viser scene."""
        with self._server.atomic():
            for i, handle in self._handles.items():
                xpos = data.geom_xpos[i] + self._position_offset
                xmat = data.geom_xmat[i]

                handle.position = (float(xpos[0]), float(xpos[1]), float(xpos[2]))
                handle.wxyz = _rotmat_to_wxyz(xmat)
