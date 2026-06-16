"""Launch localization sensor processing and fusion nodes.

T5 정공법 stack: joint_state_splitter + wheel_odom + imu_integrator + sun_yaw
+ local_height_patch + trn + ekf_fusion + localization_node.
EKF 가 /rover/estimated_odom (Odometry) + /rover/estimated_pose (PoseWithCov)
모두 발행 → arm_executor (Odometry) 와 coverage (PoseWithCov) 둘 다 호환.

**EKF Initial Pose Prior** (real-Mars-rover EDL 패턴, 9b8ebeb): terrain meta.json
의 spawn_locations[0] 을 launch 시점에 읽어 ekf_fusion 의 initial_x/y/z/yaw
파라미터로 주입. mission control 이 위성사진 + DTE uplink 로 EDL 직후 초기
절대 위치 알려주는 것과 동일 (GT cheat 아님 — 시작 1회).
없으면 (0,0,0) 으로 시작 → coverage 의 map frame 과 mismatch → 어제 본 그림.

실행 (terrain_00004 기본):
    ros2 launch isaac_bringup localization.launch.py

다른 terrain:
    ros2 launch isaac_bringup localization.launch.py terrain_id:=terrain_00010
"""

import json
import math
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _default_terrain_root() -> str:
    """isaac_sim package source 의 generated_terrains 디렉토리.
    env var ISAAC_TERRAIN_ROOT 가 있으면 우선. 없으면 git checkout 위치 추정."""
    env = os.environ.get("ISAAC_TERRAIN_ROOT")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(os.path.join(
        here, "..", "..", "..", "isaac_sim", "assets", "generated_terrains"))
    if os.path.isdir(candidate):
        return candidate
    return os.path.expanduser(
        "~/dev_ws/rover_ws/src/a2_isaac/isaac_sim/assets/generated_terrains")


def _read_initial_pose(terrain_root: str, terrain_id: str) -> dict:
    """terrain meta.json 의 spawn_locations[0] → ekf prior dict.
    실패 시 (0,0,0) 반환 — EKF 가 default 동작."""
    meta_path = os.path.join(terrain_root, terrain_id, "meta.json")
    try:
        with open(meta_path) as f:
            data = json.load(f)
        spawns = data.get("spawn_locations", [])
        if spawns:
            s = spawns[0]
            return {
                "initial_x": float(s.get("x", 0.0)),
                "initial_y": float(s.get("y", 0.0)),
                "initial_z": float(s.get("z", 0.0)) + 0.3,
                "initial_yaw": float(s.get("yaw", 0.0)),
            }
    except Exception as e:
        print(f"[localization.launch] WARN: cannot read {meta_path}: {e}")
    return {"initial_x": 0.0, "initial_y": 0.0,
            "initial_z": 0.0, "initial_yaw": 0.0}


def _build_nodes(context, *args, **kwargs):
    """OpaqueFunction — launch arg resolve 후 prior 읽어 EKF Node 생성."""
    terrain_id_str = LaunchConfiguration("terrain_id").perform(context)
    terrain_root_str = LaunchConfiguration("terrain_root").perform(context)
    prior = _read_initial_pose(terrain_root_str, terrain_id_str)
    print(f"[localization.launch] EKF prior from "
          f"{terrain_root_str}/{terrain_id_str}/meta.json: "
          f"x={prior['initial_x']:.2f} y={prior['initial_y']:.2f} "
          f"z={prior['initial_z']:.2f} "
          f"yaw={math.degrees(prior['initial_yaw']):.1f}°")

    return [
        Node(
            package="isaac_localization",
            executable="joint_state_splitter_node",
            name="joint_state_splitter",
            output="screen",
        ),
        Node(
            package="isaac_localization",
            executable="wheel_odom_node",
            name="wheel_odom_node",
            output="screen",
            parameters=[
                {
                    **prior,
                }
            ],
        ),
        Node(
            package="isaac_localization",
            executable="imu_integrator_node",
            name="imu_integrator_node",
            output="screen",
        ),
        Node(
            package="isaac_localization",
            executable="sun_yaw_node",
            name="sun_yaw_node",
            output="screen",
            parameters=[
                {
                    "image_topic": "/camera/sun/image_raw",
                    "camera_info_topic": "/camera/sun/camera_info",
                    "world_sun_yaw": 0.929,
                    # 2026-05-26 진단: -0.253/0.253 두 후보 모두 innovation 그대로
                    # ±π reject → camera_yaw_offset 으로는 못 고치는 frame 정의
                    # 문제. T5 원본 값 2.889 로 원복하고 sun_yaw_node 내부 yaw
                    # 계산 검증으로 졸업 과제 이관 (list_to_fix).
                    "camera_yaw_offset": 2.889,
                    "camera_elevation": 1.5707963268,
                    "initial_yaw": prior["initial_yaw"],
                    "auto_align_initial_yaw": True,
                    "alignment_samples": 6,
                    "alignment_min_confidence": 0.20,
                    "alignment_max_residual": 3.1415926536,
                    "debug_log_hz": 0.5,
                    "imu_topic": "/imu/data",
                    "use_imu_tilt_compensation": True,
                    "base_yaw_variance": 0.10,
                    "max_yaw_variance": 4.0,
                    "min_confidence": 0.20,
                    "top_crop_ratio": 1.0,
                    "bright_percentile": 99.7,
                    "min_peak_luma": 190.0,
                    "max_area_ratio": 0.018,
                    "max_blob_width_ratio": 0.18,
                    "max_blob_height_ratio": 0.18,
                    "min_dominance": 18.0,
                    "temporal_alpha": 0.30,
                    "max_bearing_jump": 0.40,
                    "max_publish_hz": 10.0,
                }
            ],
        ),
        Node(
            package="isaac_localization",
            executable="local_height_patch_node",
            name="local_height_patch_node",
            output="screen",
            parameters=[
                {
                    "odom_topic": "/rover/estimated_odom",
                    "pixel_stride": 2,
                    "accumulate_window_s": 4.0,
                    "fill_sparse_holes": True,
                    "min_valid_cells": 250,
                    "obstacle_patch_topic": "/rover/local_obstacle_patch",
                    "obstacle_height_delta_m": 0.35,
                    "obstacle_slope_deg": 20.0,
                    "obstacle_dilation_cells": 1,
                    # vehicle_v3 /Root/Vehicle/rover/Body/Camera transform,
                    # expressed in the rover body frame. USD camera looks along
                    # local -Z, which maps to rover +X; ROS optical +X maps to
                    # rover -Y, and optical +Y/down maps to rover -Z.
                    "camera_x": 0.3,
                    "camera_y": 0.0,
                    "camera_z": -0.1,
                    "camera_forward_x": 1.0,
                    "camera_forward_y": 0.0,
                    "camera_forward_z": 0.0,
                    "camera_right_x": 0.0,
                    "camera_right_y": -1.0,
                    "camera_right_z": 0.0,
                    "camera_down_x": 0.0,
                    "camera_down_y": 0.0,
                    "camera_down_z": -1.0,
                }
            ],
        ),
        Node(
            package="isaac_localization",
            executable="trn_node",
            name="trn_node",
            output="screen",
            parameters=[
                {
                    "terrain_id": terrain_id_str,
                    "terrain_root": terrain_root_str,
                    # TRN은 scan matching의 탐색 중심만 필요하다. raw wheel_odom은
                    # yaw drift가 누적되어 카메라 patch 후보를 엉뚱한 곳으로 끌고
                    # 갈 수 있으므로 EKF prediction을 prior로 사용한다.
                    "prior_odom_topic": "/rover/estimated_odom",
                    "local_obstacle_topic": "/rover/local_obstacle_patch",
                    "match_sigma": 0.85,
                    "height_weight": 0.85,
                    "slope_weight": 0.25,
                    "obstacle_weight": 1.40,
                    "obstacle_dilation_m": 1.25,
                    "obstacle_distance_sigma_m": 0.75,
                    "obstacle_distance_max_m": 2.0,
                    "yaw_search_max_deg": 12.0,
                    "yaw_search_step_deg": 4.0,
                    "slope_sigma_deg": 12.0,
                    "local_obstacle_slope_deg": 20.0,
                    "local_obstacle_height_delta_m": 0.35,
                    "min_feature_overlap_cells": 450,
                    "prior_weight": 0.12,
                    "prior_sigma_m": 1.0,
                    "min_confidence": 0.30,
                    "min_publish_local_obstacle_ratio": 0.08,
                    "max_publish_obstacle_score": 0.70,
                    "min_publish_ambiguity_margin": 0.07,
                    "base_xy_covariance": 0.50,
                    "base_z_covariance": 0.50,
                    "distinct_candidate_separation_m": 0.75,
                    "ambiguity_margin_m": 0.04,
                }
            ],
        ),
        Node(
            package="isaac_localization",
            executable="ekf_fusion_node",
            name="ekf_fusion_node",
            output="screen",
            parameters=[
                {
                    "sun_yaw_topic": "/rover/sun_yaw",
                    "use_sun_yaw": True,
                    "front_depth_topic": "/camera/rover/depth",
                    "wheel_slip_guard_enabled": True,
                    "front_block_depth_m": 0.75,
                    "front_block_min_ratio": 0.08,
                    "front_block_hold_s": 0.60,
                    "slip_guard_linear_scale": 0.0,
                    "slip_guard_lateral_scale": 0.0,
                    "slip_guard_yaw_scale": 0.20,
                    "slip_guard_noise_multiplier": 25.0,
                    "default_sun_yaw_cov": 0.60,
                    "min_sun_yaw_cov": 0.10,
                    # π 로 풀어 큰 보정도 수용 — EKF wheel-drift 누적이 sun_yaw
                    # innovation 을 1071회 reject 시켰던 2026-05-26 회귀 대응.
                    "max_sun_yaw_innovation": math.pi,
                    "max_trn_innovation_m": 1.0,
                    # EDL initial pose prior — terrain spawn 으로 시드.
                    **prior,
                }
            ],
        ),
        Node(
            package="isaac_localization",
            executable="localization_node",
            name="localization_node",
            output="screen",
            parameters=[
                {
                    "terrain_id": terrain_id_str,
                    "terrain_root": terrain_root_str,
                }
            ],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "terrain_id",
                default_value="terrain_00004",
                description="Generated terrain directory name used by TRN.",
            ),
            DeclareLaunchArgument(
                "terrain_root",
                default_value=_default_terrain_root(),
                description="Directory containing terrain_<id>/heightmap.npy and meta.json.",
            ),
            OpaqueFunction(function=_build_nodes),
        ]
    )
