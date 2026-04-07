from __future__ import annotations

import argparse
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


BOOK_SIDE_METERS = 12.0 * 0.0254
BOOK_THICKNESS_METERS = 0.01
DEFAULT_FPS = 30.0
DEFAULT_GRIPPER_CLOSE_SCALE = 0.8
DEFAULT_QUAT_ORDER = "xyzw"


@dataclass
class PosePlaybackData:
    box_poses: np.ndarray
    left_gripper_poses: np.ndarray
    right_gripper_poses: np.ndarray
    left_gripper_grasps: np.ndarray
    right_gripper_grasps: np.ndarray
    replaced_pose_rows: dict[str, list[int]]

    @property
    def num_frames(self) -> int:
        return int(self.box_poses.shape[0])


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else _repo_root() / path


def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)

    k = rvec / theta
    kx, ky, kz = k
    K = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _wxyz_from_rotation_matrix(R: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    return tuple(float(v) for v in q)


def _pose6_to_viser_pose(pose: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    position = tuple(float(v) for v in pose[:3])
    wxyz = _wxyz_from_rotation_matrix(_rodrigues(np.asarray(pose[3:6], dtype=np.float64)))
    return position, wxyz


def _normalize_wxyz(wxyz: np.ndarray) -> tuple[float, float, float, float]:
    wxyz = np.asarray(wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(wxyz))
    if norm < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(float(v) for v in wxyz / norm)


def _quat_to_wxyz(quat: np.ndarray, quat_order: str) -> tuple[float, float, float, float]:
    quat = np.asarray(quat, dtype=np.float64)
    if quat_order == "xyzw":
        quat = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)
    elif quat_order != "wxyz":
        raise ValueError(f"Unsupported quaternion order: {quat_order}")
    return _normalize_wxyz(quat)


def _pose_to_viser_pose(pose: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape[0] == 6:
        return _pose6_to_viser_pose(pose)
    if pose.shape[0] == 7:
        position = tuple(float(v) for v in pose[:3])
        wxyz = _normalize_wxyz(pose[3:7])
        return position, wxyz
    raise ValueError(f"Expected pose row with 6 or 7 values, got shape {pose.shape}.")


def _fill_nonfinite_rows(
    values: np.ndarray,
    name: str,
    expected_columns: tuple[int, ...],
) -> tuple[np.ndarray, list[int]]:
    values = np.asarray(values, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] not in expected_columns:
        raise ValueError(f"{name} must have shape (N, {expected_columns}), got {values.shape}.")

    valid = np.isfinite(values).all(axis=1)
    invalid_indices = np.flatnonzero(~valid)
    if invalid_indices.size == 0:
        return values, []

    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        raise ValueError(f"{name} has no finite rows.")

    for idx in invalid_indices:
        previous = valid_indices[valid_indices < idx]
        replacement_idx = previous[-1] if previous.size else valid_indices[0]
        values[idx] = values[replacement_idx]

    return values, [int(i) for i in invalid_indices]


def _normalize_pose_array(poses: np.ndarray, name: str, quat_order: str) -> np.ndarray:
    poses, _ = _fill_nonfinite_rows(poses, name, (6, 7))
    if poses.shape[1] == 6:
        return poses

    wxyzs = np.array([_quat_to_wxyz(quat, quat_order) for quat in poses[:, 3:7]], dtype=np.float64)
    return np.concatenate([poses[:, :3], wxyzs], axis=1)


def _load_npz_key(data: Any, key: str, fallback_keys: tuple[str, ...] = ()) -> np.ndarray:
    for candidate in (key, *fallback_keys):
        if candidate in data.files:
            return data[candidate]
    keys = ", ".join((key, *fallback_keys))
    raise KeyError(f"Missing required NPZ key. Expected one of: {keys}. Found: {data.files}")


def _load_gripper_poses(data: Any, side: str, quat_order: str) -> tuple[np.ndarray, list[int]]:
    base_key = f"{side}_bases"
    quat_key = f"{side}_quats"
    if base_key in data.files and quat_key in data.files:
        bases, bad_bases = _fill_nonfinite_rows(data[base_key], base_key, (3,))
        quats, bad_quats = _fill_nonfinite_rows(data[quat_key], quat_key, (4,))
        wxyzs = np.array([_quat_to_wxyz(quat, quat_order) for quat in quats], dtype=np.float64)
        return np.concatenate([bases, wxyzs], axis=1), sorted(set(bad_bases + bad_quats))

    raw_poses = _load_npz_key(data, f"{side}_gripper_poses", (f"{side}_poses",))
    poses, bad_poses = _fill_nonfinite_rows(raw_poses, f"{side}_gripper_poses", (6, 7))
    return _normalize_pose_array(poses, f"{side}_gripper_poses", quat_order), bad_poses


def _load_playback_data(npz_path: Path, quat_order: str = DEFAULT_QUAT_ORDER) -> PosePlaybackData:
    if not npz_path.exists():
        raise FileNotFoundError(f"Pose NPZ not found: {npz_path}")

    with np.load(npz_path) as data:
        raw_box_poses, bad_box = _fill_nonfinite_rows(data["box_poses"], "box_poses", (6, 7))
        box_poses = _normalize_pose_array(raw_box_poses, "box_poses", quat_order)
        left_gripper_poses, bad_left = _load_gripper_poses(data, "left", quat_order)
        right_gripper_poses, bad_right = _load_gripper_poses(data, "right", quat_order)
        left_gripper_grasps = np.asarray(
            _load_npz_key(data, "left_gripper_grasps", ("left_grasps",)),
            dtype=np.float64,
        ).reshape(-1)
        right_gripper_grasps = np.asarray(
            _load_npz_key(data, "right_gripper_grasps", ("right_grasps",)),
            dtype=np.float64,
        ).reshape(-1)

    frame_count = box_poses.shape[0]
    for key, arr in {
        "left_gripper_poses": left_gripper_poses,
        "right_gripper_poses": right_gripper_poses,
        "left_gripper_grasps": left_gripper_grasps,
        "right_gripper_grasps": right_gripper_grasps,
    }.items():
        if arr.shape[0] != frame_count:
            raise ValueError(f"{key} has {arr.shape[0]} frames, expected {frame_count}.")

    return PosePlaybackData(
        box_poses=box_poses,
        left_gripper_poses=left_gripper_poses,
        right_gripper_poses=right_gripper_poses,
        left_gripper_grasps=np.clip(
            np.nan_to_num(left_gripper_grasps, nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        ),
        right_gripper_grasps=np.clip(
            np.nan_to_num(right_gripper_grasps, nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        ),
        replaced_pose_rows={
            "box_poses": bad_box,
            "left_gripper_poses": bad_left,
            "right_gripper_poses": bad_right,
        },
    )


def _make_resolved_urdf(urdf_path: Path) -> Path:
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    package_root = urdf_path.parent.parent
    mesh_root = package_root / "meshes"
    text = urdf_path.read_text()
    text = text.replace(
        "package://robotiq_arg85_description/meshes/",
        mesh_root.resolve().as_posix() + "/",
    )
    text = text.replace(
        "package://robotiq_arg85_description/",
        package_root.resolve().as_posix() + "/",
    )

    output_path = Path(tempfile.gettempdir()) / f"{urdf_path.stem}_viser_resolved.urdf"
    output_path.write_text(text)
    return output_path


def _require_viser() -> tuple[Any, Any]:
    try:
        import viser
        from viser.extras import ViserUrdf
    except ImportError as exc:
        raise ImportError(
            "This visualization needs Viser URDF dependencies. Install them with:\n"
            "  pip install 'viser[examples]'\n"
            "or, if you prefer minimal packages:\n"
            "  pip install viser yourdfpy trimesh"
        ) from exc
    return viser, ViserUrdf


def _joint_configuration_from_grasp(
    viser_urdf: Any,
    grasp: float,
    close_scale: float,
) -> np.ndarray:
    joint_names = tuple(viser_urdf.get_actuated_joint_names())
    if not joint_names:
        return np.empty((0,), dtype=np.float64)

    limits = viser_urdf.get_actuated_joint_limits()
    finger_upper = 0.725
    if "finger_joint" in limits and limits["finger_joint"][1] is not None:
        finger_upper = float(limits["finger_joint"][1])

    finger_angle = float(np.clip(grasp, 0.0, 1.0)) * finger_upper * close_scale
    mimic_values = {
        "finger_joint": finger_angle,
        "left_inner_knuckle_joint": finger_angle,
        "left_inner_finger_joint": -finger_angle,
        "right_inner_knuckle_joint": -finger_angle,
        "right_inner_finger_joint": finger_angle,
        "right_outer_knuckle_joint": -finger_angle,
    }
    return np.array([mimic_values.get(name, 0.0) for name in joint_names], dtype=np.float64)


def _update_urdf_grasp(viser_urdf: Any, grasp: float, close_scale: float) -> None:
    cfg = _joint_configuration_from_grasp(viser_urdf, grasp, close_scale)
    viser_urdf.update_cfg(cfg)


def _apply_pose(handle: Any, pose: np.ndarray) -> None:
    position, wxyz = _pose_to_viser_pose(pose)
    handle.position = position
    handle.wxyz = wxyz


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Loop book and gripper poses in a Viser scene.")
    parser.add_argument("--npz", type=Path, default=Path("data/demo_quaternion_poses.npz"))
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("gripper_model/robots/robotiq_arg85_description.URDF"),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--gripper-close-scale", type=float, default=DEFAULT_GRIPPER_CLOSE_SCALE)
    parser.add_argument(
        "--quat-order",
        choices=("xyzw", "wxyz"),
        default=DEFAULT_QUAT_ORDER,
        help="Quaternion component order in NPZ arrays. Viser uses wxyz internally.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    npz_path = _resolve_repo_path(args.npz)
    urdf_path = _resolve_repo_path(args.urdf)

    playback = _load_playback_data(npz_path, args.quat_order)
    replaced = {key: rows for key, rows in playback.replaced_pose_rows.items() if rows}
    if replaced:
        print(f"Replaced non-finite pose rows with previous valid poses: {replaced}")

    viser, ViserUrdf = _require_viser()
    resolved_urdf_path = _make_resolved_urdf(urdf_path)

    server = viser.ViserServer(host=args.host, port=args.port, label="Book/gripper playback")
    server.scene.set_up_direction("+z")

    scene_center = np.median(playback.box_poses[:, :3], axis=0)
    server.initial_camera.position = tuple(float(v) for v in scene_center + np.array([0.6, -0.8, 0.45]))
    server.initial_camera.look_at = tuple(float(v) for v in scene_center)

    server.scene.add_grid(
        "/grid",
        width=1.0,
        height=1.0,
        plane="xy",
        cell_size=0.05,
        section_size=0.25,
        position=tuple(float(v) for v in scene_center - np.array([0.0, 0.0, 0.15])),
    )

    book_root = server.scene.add_frame("/book", show_axes=True, axes_length=0.05, axes_radius=0.002)
    server.scene.add_box(
        "/book/block",
        color=(130, 90, 45),
        dimensions=(BOOK_SIDE_METERS, BOOK_SIDE_METERS, BOOK_THICKNESS_METERS),
        opacity=0.85,
    )

    left_root = server.scene.add_frame("/left_gripper", show_axes=True, axes_length=0.05, axes_radius=0.002)
    right_root = server.scene.add_frame("/right_gripper", show_axes=True, axes_length=0.05, axes_radius=0.002)
    server.scene.add_label("/left_gripper/label", "left", position=(0.0, 0.0, 0.12))
    server.scene.add_label("/right_gripper/label", "right", position=(0.0, 0.0, 0.12))

    left_urdf = ViserUrdf(
        server,
        urdf_or_path=resolved_urdf_path,
        root_node_name="/left_gripper",
        mesh_color_override=(255, 150, 40),
        load_meshes=True,
        load_collision_meshes=False,
    )
    right_urdf = ViserUrdf(
        server,
        urdf_or_path=resolved_urdf_path,
        root_node_name="/right_gripper",
        mesh_color_override=(40, 190, 210),
        load_meshes=True,
        load_collision_meshes=False,
    )

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=playback.num_frames - 1,
            step=1,
            initial_value=0,
        )
        play_checkbox = server.gui.add_checkbox("Play", initial_value=not args.start_paused)
        loop_checkbox = server.gui.add_checkbox("Loop", initial_value=True)
        fps_slider = server.gui.add_slider("FPS", min=1.0, max=60.0, step=1.0, initial_value=float(args.fps))

    state = {"frame": 0}

    def set_frame(frame_idx: int) -> None:
        frame_idx = int(np.clip(frame_idx, 0, playback.num_frames - 1))
        state["frame"] = frame_idx
        with server.atomic():
            _apply_pose(book_root, playback.box_poses[frame_idx])
            _apply_pose(left_root, playback.left_gripper_poses[frame_idx])
            _apply_pose(right_root, playback.right_gripper_poses[frame_idx])
            _update_urdf_grasp(
                left_urdf,
                playback.left_gripper_grasps[frame_idx],
                args.gripper_close_scale,
            )
            _update_urdf_grasp(
                right_urdf,
                playback.right_gripper_grasps[frame_idx],
                args.gripper_close_scale,
            )

    @frame_slider.on_update
    def _(_: Any) -> None:
        set_frame(int(frame_slider.value))

    set_frame(0)
    print(f"Loaded {playback.num_frames} frames from {npz_path}")
    print(f"Open the Viser viewer at http://{server.get_host()}:{server.get_port()}")

    try:
        while True:
            if not play_checkbox.value:
                time.sleep(0.03)
                continue

            next_frame = state["frame"] + 1
            if next_frame >= playback.num_frames:
                if loop_checkbox.value:
                    next_frame = 0
                else:
                    next_frame = playback.num_frames - 1
                    play_checkbox.value = False

            frame_slider.value = next_frame
            set_frame(next_frame)
            time.sleep(1.0 / max(float(fps_slider.value), 1.0))
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
