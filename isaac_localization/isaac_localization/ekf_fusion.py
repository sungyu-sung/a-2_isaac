#!/usr/bin/env python3

import math
from typing import List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, TransformStamped
from sensor_msgs.msg import Image

from tf2_ros import TransformBroadcaster


class EKFFusionNode(Node):
    """
    EKF Fusion Node

    역할:
    - /rover/wheel_odom을 이용해 상대 이동을 예측한다.
    - /rover/imu_odom의 orientation으로 roll, pitch, yaw를 보정한다.
    - /rover/sun_yaw의 yaw-only observation으로 절대 방위를 약하게 보정한다.
    - /rover/trn_pose의 x, y, z로 위치를 보정한다.
    - 최종 추정 결과를 /rover/estimated_odom, /rover/estimated_pose로 publish한다.

    상태 벡터:
    x = [pos_x, pos_y, pos_z, roll, pitch, yaw]^T
    """

    IDX_X = 0
    IDX_Y = 1
    IDX_Z = 2
    IDX_ROLL = 3
    IDX_PITCH = 4
    IDX_YAW = 5

    STATE_SIZE = 6

    def __init__(self):
        super().__init__("ekf_fusion_node")

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter("wheel_odom_topic", "/rover/wheel_odom")
        self.declare_parameter("imu_odom_topic", "/rover/imu_odom")
        self.declare_parameter("trn_pose_topic", "/rover/trn_pose")
        self.declare_parameter("sun_yaw_topic", "/rover/sun_yaw")
        self.declare_parameter("use_sun_yaw", True)
        self.declare_parameter("front_depth_topic", "/camera/rover/depth")
        self.declare_parameter("wheel_slip_guard_enabled", True)

        self.declare_parameter("estimated_odom_topic", "/rover/estimated_odom")
        self.declare_parameter("estimated_pose_topic", "/rover/estimated_pose")

        self.declare_parameter("frame_id", "world")
        self.declare_parameter("child_frame_id", "base_link")

        # wheel prediction process noise
        self.declare_parameter("base_process_noise_xy", 0.001)
        self.declare_parameter("process_noise_xy_per_m", 0.05)
        self.declare_parameter("base_process_noise_z", 0.0001)
        self.declare_parameter("base_process_noise_rp", 0.0001)
        self.declare_parameter("base_process_noise_yaw", 0.001)
        self.declare_parameter("process_noise_yaw_per_rad", 0.05)

        # measurement covariance fallback
        self.declare_parameter("default_imu_roll_cov", 0.05)
        self.declare_parameter("default_imu_pitch_cov", 0.05)
        self.declare_parameter("default_imu_yaw_cov", 0.05)

        self.declare_parameter("default_trn_x_cov", 0.10)
        self.declare_parameter("default_trn_y_cov", 0.10)
        self.declare_parameter("default_trn_z_cov", 0.20)
        self.declare_parameter("max_trn_innovation_m", 1.0)
        self.declare_parameter("default_sun_yaw_cov", 1.50)
        self.declare_parameter("min_sun_yaw_cov", 0.25)
        self.declare_parameter("max_sun_yaw_innovation", 1.20)

        # If the rover is pushing into an obstacle, wheel joints can keep
        # spinning while the chassis is stationary. A close front-depth return
        # is used as a realistic stuck/slip cue to suppress forward wheel
        # integration until the rover backs out or turns away.
        self.declare_parameter("front_block_depth_m", 0.75)
        self.declare_parameter("front_block_min_ratio", 0.08)
        self.declare_parameter("front_block_center_width_ratio", 0.35)
        self.declare_parameter("front_block_center_height_ratio", 0.45)
        self.declare_parameter("front_block_hold_s", 0.60)
        self.declare_parameter("slip_guard_min_forward_delta_m", 0.002)
        self.declare_parameter("slip_guard_linear_scale", 0.0)
        self.declare_parameter("slip_guard_lateral_scale", 0.0)
        self.declare_parameter("slip_guard_yaw_scale", 0.20)
        self.declare_parameter("slip_guard_noise_multiplier", 25.0)

        # output
        self.declare_parameter("publish_tf", False)

        # -----------------------------
        # Read parameters
        # -----------------------------
        self.wheel_odom_topic = self.get_parameter("wheel_odom_topic").value
        self.imu_odom_topic = self.get_parameter("imu_odom_topic").value
        self.trn_pose_topic = self.get_parameter("trn_pose_topic").value
        self.sun_yaw_topic = self.get_parameter("sun_yaw_topic").value
        self.use_sun_yaw = bool(self.get_parameter("use_sun_yaw").value)
        self.front_depth_topic = self.get_parameter("front_depth_topic").value
        self.wheel_slip_guard_enabled = bool(
            self.get_parameter("wheel_slip_guard_enabled").value
        )

        self.estimated_odom_topic = self.get_parameter("estimated_odom_topic").value
        self.estimated_pose_topic = self.get_parameter("estimated_pose_topic").value

        self.frame_id = self.get_parameter("frame_id").value
        self.child_frame_id = self.get_parameter("child_frame_id").value

        self.base_process_noise_xy = float(
            self.get_parameter("base_process_noise_xy").value
        )
        self.process_noise_xy_per_m = float(
            self.get_parameter("process_noise_xy_per_m").value
        )
        self.base_process_noise_z = float(
            self.get_parameter("base_process_noise_z").value
        )
        self.base_process_noise_rp = float(
            self.get_parameter("base_process_noise_rp").value
        )
        self.base_process_noise_yaw = float(
            self.get_parameter("base_process_noise_yaw").value
        )
        self.process_noise_yaw_per_rad = float(
            self.get_parameter("process_noise_yaw_per_rad").value
        )

        self.default_imu_roll_cov = float(
            self.get_parameter("default_imu_roll_cov").value
        )
        self.default_imu_pitch_cov = float(
            self.get_parameter("default_imu_pitch_cov").value
        )
        self.default_imu_yaw_cov = float(
            self.get_parameter("default_imu_yaw_cov").value
        )

        self.default_trn_x_cov = float(
            self.get_parameter("default_trn_x_cov").value
        )
        self.default_trn_y_cov = float(
            self.get_parameter("default_trn_y_cov").value
        )
        self.default_trn_z_cov = float(
            self.get_parameter("default_trn_z_cov").value
        )
        self.max_trn_innovation_m = float(
            self.get_parameter("max_trn_innovation_m").value
        )
        self.default_sun_yaw_cov = float(
            self.get_parameter("default_sun_yaw_cov").value
        )
        self.min_sun_yaw_cov = float(
            self.get_parameter("min_sun_yaw_cov").value
        )
        self.max_sun_yaw_innovation = float(
            self.get_parameter("max_sun_yaw_innovation").value
        )
        self.front_block_depth_m = float(
            self.get_parameter("front_block_depth_m").value
        )
        self.front_block_min_ratio = float(
            self.get_parameter("front_block_min_ratio").value
        )
        self.front_block_center_width_ratio = float(
            self.get_parameter("front_block_center_width_ratio").value
        )
        self.front_block_center_height_ratio = float(
            self.get_parameter("front_block_center_height_ratio").value
        )
        self.front_block_hold_s = float(
            self.get_parameter("front_block_hold_s").value
        )
        self.slip_guard_min_forward_delta_m = float(
            self.get_parameter("slip_guard_min_forward_delta_m").value
        )
        self.slip_guard_linear_scale = float(
            self.get_parameter("slip_guard_linear_scale").value
        )
        self.slip_guard_lateral_scale = float(
            self.get_parameter("slip_guard_lateral_scale").value
        )
        self.slip_guard_yaw_scale = float(
            self.get_parameter("slip_guard_yaw_scale").value
        )
        self.slip_guard_noise_multiplier = float(
            self.get_parameter("slip_guard_noise_multiplier").value
        )

        self.publish_tf_enabled = bool(self.get_parameter("publish_tf").value)

        # -----------------------------
        # Initial Pose Prior — real-Mars-rover EDL 패턴.
        # mission control 이 위성사진 + DTE uplink 로 초기 절대 위치를 알려주는 것과
        # 동일하게, 시뮬에서는 terrain meta.json 의 spawn_locations 값을 launch
        # 시점에 parameter 로 주입한다. 매 frame 받는 GT cheat 와 다름 (한 번만).
        # -----------------------------
        self.declare_parameter("initial_x", 0.0)
        self.declare_parameter("initial_y", 0.0)
        self.declare_parameter("initial_z", 0.0)
        self.declare_parameter("initial_yaw", 0.0)
        initial_x = float(self.get_parameter("initial_x").value)
        initial_y = float(self.get_parameter("initial_y").value)
        initial_z = float(self.get_parameter("initial_z").value)
        initial_yaw = float(self.get_parameter("initial_yaw").value)

        # -----------------------------
        # EKF internal state
        # -----------------------------
        self.x = np.zeros((self.STATE_SIZE, 1), dtype=np.float64)
        self.x[self.IDX_X, 0] = initial_x
        self.x[self.IDX_Y, 0] = initial_y
        self.x[self.IDX_Z, 0] = initial_z
        self.x[self.IDX_YAW, 0] = initial_yaw

        self.P = np.eye(self.STATE_SIZE, dtype=np.float64)
        # 초기 위치 prior 가 들어왔으면 covariance 작게 (신뢰), 0 default 면 큰 그대로.
        prior_given = any(
            v != 0.0 for v in (initial_x, initial_y, initial_z, initial_yaw)
        )
        if prior_given:
            self.P[self.IDX_X, self.IDX_X] = 0.01      # orbital imagery 정확도
            self.P[self.IDX_Y, self.IDX_Y] = 0.01
            self.P[self.IDX_Z, self.IDX_Z] = 0.04
            self.P[self.IDX_YAW, self.IDX_YAW] = 0.0012  # celestial fix (≈ 2°)
            self.get_logger().info(
                f"Initial pose prior applied: x={initial_x:.2f} y={initial_y:.2f} "
                f"z={initial_z:.2f} yaw={math.degrees(initial_yaw):.1f}°"
            )
        else:
            self.P[self.IDX_X, self.IDX_X] = 1.0
            self.P[self.IDX_Y, self.IDX_Y] = 1.0
            self.P[self.IDX_Z, self.IDX_Z] = 1.0
        # roll/pitch 는 prior 와 무관하게 IMU 가 빠르게 잡음
        self.P[self.IDX_ROLL, self.IDX_ROLL] = 0.5
        self.P[self.IDX_PITCH, self.IDX_PITCH] = 0.5
        if not prior_given:
            self.P[self.IDX_YAW, self.IDX_YAW] = 0.5

        # prior 있으면 첫 wheel_odom 메시지로 state 가 덮어쓰여지지 않도록 True 로 시작.
        # (wheel_odom 자체는 자기 origin=(0,0,0) 이라 첫 msg 가 절대값 prior 를 망가뜨림)
        self.initialized = prior_given

        # wheel odom relative delta 계산용
        self.prev_wheel_pose: Optional[Tuple[float, float, float, float, float, float]] = None

        # IMU가 먼저 들어올 경우 초기 자세 후보로 저장
        self.latest_imu_rpy: Optional[Tuple[float, float, float]] = None

        # twist output용
        self.last_linear_v = 0.0
        self.last_angular_z = 0.0
        self.wheel_slip_guard_active = False
        self.front_blocked = False
        self.front_blocked_until_ns = 0
        self.front_block_ratio = 0.0
        self.front_block_min_depth = math.inf
        self.last_slip_guard_log_ns = 0

        self.last_stamp: Optional[Time] = None

        # -----------------------------
        # ROS interfaces
        # -----------------------------
        self.wheel_sub = self.create_subscription(
            Odometry,
            self.wheel_odom_topic,
            self.wheel_odom_callback,
            20,
        )

        self.imu_sub = self.create_subscription(
            Odometry,
            self.imu_odom_topic,
            self.imu_odom_callback,
            20,
        )

        self.trn_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.trn_pose_topic,
            self.trn_pose_callback,
            10,
        )

        self.sun_yaw_sub = None
        if self.use_sun_yaw:
            self.sun_yaw_sub = self.create_subscription(
                PoseWithCovarianceStamped,
                self.sun_yaw_topic,
                self.sun_yaw_callback,
                10,
            )

        self.front_depth_sub = None
        if self.wheel_slip_guard_enabled:
            self.front_depth_sub = self.create_subscription(
                Image,
                self.front_depth_topic,
                self.front_depth_callback,
                qos_profile_sensor_data,
            )

        self.estimated_odom_pub = self.create_publisher(
            Odometry,
            self.estimated_odom_topic,
            10,
        )

        self.estimated_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            self.estimated_pose_topic,
            10,
        )

        self.tf_broadcaster = None
        if self.publish_tf_enabled:
            self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info("EKF Fusion Node initialized.")
        self.get_logger().info(f"Subscribe wheel odom: {self.wheel_odom_topic}")
        self.get_logger().info(f"Subscribe imu odom  : {self.imu_odom_topic}")
        self.get_logger().info(f"Subscribe trn pose  : {self.trn_pose_topic}")
        if self.use_sun_yaw:
            self.get_logger().info(f"Subscribe sun yaw   : {self.sun_yaw_topic}")
        if self.wheel_slip_guard_enabled:
            self.get_logger().info(
                f"Wheel slip guard : depth={self.front_depth_topic}, "
                f"block<{self.front_block_depth_m:.2f}m ratio>{self.front_block_min_ratio:.2f}"
            )
        self.get_logger().info(f"Publish estimated odom: {self.estimated_odom_topic}")
        self.get_logger().info(f"Publish estimated pose: {self.estimated_pose_topic}")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def wheel_odom_callback(self, msg: Odometry) -> None:
        stamp = self.get_odom_time(msg)

        current_pose = self.extract_pose_6d_from_odom(msg)

        self.last_linear_v = msg.twist.twist.linear.x
        self.last_angular_z = msg.twist.twist.angular.z

        if self.prev_wheel_pose is None:
            self.prev_wheel_pose = current_pose

            if not self.initialized:
                self.initialize_from_wheel(current_pose)
                self.last_stamp = stamp
                self.publish_estimate(stamp)

            return

        if not self.initialized:
            self.initialize_from_wheel(current_pose)
            self.prev_wheel_pose = current_pose
            self.last_stamp = stamp
            self.publish_estimate(stamp)
            return

        self.predict_from_wheel_delta(
            prev_pose=self.prev_wheel_pose,
            current_pose=current_pose,
        )
        if self.wheel_slip_guard_active and self.last_linear_v > 0.0:
            self.last_linear_v = 0.0

        self.prev_wheel_pose = current_pose
        self.last_stamp = stamp

        self.publish_estimate(stamp)

    def front_depth_callback(self, msg: Image) -> None:
        if not self.wheel_slip_guard_enabled:
            return

        depth = self.depth_image_to_array(msg)
        if depth is None or depth.size == 0:
            return

        height, width = depth.shape
        wr = min(1.0, max(0.05, self.front_block_center_width_ratio))
        hr = min(1.0, max(0.05, self.front_block_center_height_ratio))

        x0 = int(width * (0.5 - 0.5 * wr))
        x1 = int(width * (0.5 + 0.5 * wr))
        y0 = int(height * (0.5 - 0.5 * hr))
        y1 = int(height * (0.5 + 0.5 * hr))
        roi = depth[max(0, y0):min(height, y1), max(0, x0):min(width, x1)]
        finite = np.isfinite(roi) & (roi > 0.02)
        if not np.any(finite):
            self.front_blocked = False
            return

        finite_depth = roi[finite]
        close = finite_depth < self.front_block_depth_m
        ratio = float(np.count_nonzero(close)) / float(finite_depth.size)
        self.front_block_ratio = ratio
        self.front_block_min_depth = float(np.nanmin(finite_depth))

        now_ns = self.get_clock().now().nanoseconds
        blocked = ratio >= self.front_block_min_ratio
        self.front_blocked = blocked
        if blocked:
            self.front_blocked_until_ns = now_ns + int(self.front_block_hold_s * 1.0e9)

    def depth_image_to_array(self, msg: Image) -> Optional[np.ndarray]:
        encoding = msg.encoding.upper()
        if encoding == "32FC1":
            dtype = np.float32
            scale = 1.0
        elif encoding == "16UC1":
            dtype = np.uint16
            scale = 0.001
        else:
            return None

        try:
            depth = np.frombuffer(msg.data, dtype=dtype).reshape((msg.height, msg.width))
        except ValueError:
            return None
        return depth.astype(np.float32) * scale

    def imu_odom_callback(self, msg: Odometry) -> None:
        stamp = self.get_odom_time(msg)

        roll, pitch, yaw = self.euler_from_quaternion(msg.pose.pose.orientation)
        self.latest_imu_rpy = (roll, pitch, yaw)

        if not self.initialized:
            return

        z = np.array([[roll], [pitch], [yaw]], dtype=np.float64)

        H = np.zeros((3, self.STATE_SIZE), dtype=np.float64)
        H[0, self.IDX_ROLL] = 1.0
        H[1, self.IDX_PITCH] = 1.0
        H[2, self.IDX_YAW] = 1.0

        roll_cov = self.valid_covariance_or_default(
            msg.pose.covariance[21],
            self.default_imu_roll_cov,
        )
        pitch_cov = self.valid_covariance_or_default(
            msg.pose.covariance[28],
            self.default_imu_pitch_cov,
        )
        yaw_cov = self.valid_covariance_or_default(
            msg.pose.covariance[35],
            self.default_imu_yaw_cov,
        )

        R = np.diag([roll_cov, pitch_cov, yaw_cov]).astype(np.float64)

        self.measurement_update(
            z=z,
            H=H,
            R=R,
            angle_measurement_rows=[0, 1, 2],
        )

        self.last_stamp = stamp
        self.publish_estimate(stamp)

    def trn_pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        stamp = Time.from_msg(msg.header.stamp)

        trn_x = msg.pose.pose.position.x
        trn_y = msg.pose.pose.position.y
        trn_z = msg.pose.pose.position.z

        roll, pitch, yaw = self.euler_from_quaternion(msg.pose.pose.orientation)

        if not self.initialized:
            self.initialize_from_trn(
                trn_x=trn_x,
                trn_y=trn_y,
                trn_z=trn_z,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
            )
            self.last_stamp = stamp
            self.publish_estimate(stamp)
            return

        trn_xy_innovation = math.hypot(
            trn_x - self.x[self.IDX_X, 0],
            trn_y - self.x[self.IDX_Y, 0],
        )
        if (
            self.max_trn_innovation_m > 0.0
            and trn_xy_innovation > self.max_trn_innovation_m
        ):
            self.get_logger().warn(
                f"TRN rejected: innovation={trn_xy_innovation:.3f} m "
                f"(measurement x={trn_x:.2f}, y={trn_y:.2f})"
            )
            return

        z = np.array([[trn_x], [trn_y], [trn_z]], dtype=np.float64)

        H = np.zeros((3, self.STATE_SIZE), dtype=np.float64)
        H[0, self.IDX_X] = 1.0
        H[1, self.IDX_Y] = 1.0
        H[2, self.IDX_Z] = 1.0

        x_cov = self.valid_covariance_or_default(
            msg.pose.covariance[0],
            self.default_trn_x_cov,
        )
        y_cov = self.valid_covariance_or_default(
            msg.pose.covariance[7],
            self.default_trn_y_cov,
        )
        z_cov = self.valid_covariance_or_default(
            msg.pose.covariance[14],
            self.default_trn_z_cov,
        )

        R = np.diag([x_cov, y_cov, z_cov]).astype(np.float64)

        self.measurement_update(
            z=z,
            H=H,
            R=R,
            angle_measurement_rows=[],
        )

        self.last_stamp = stamp
        self.publish_estimate(stamp)

    def sun_yaw_callback(self, msg: PoseWithCovarianceStamped) -> None:
        if not self.initialized:
            return

        stamp = Time.from_msg(msg.header.stamp)
        _, _, yaw = self.euler_from_quaternion(msg.pose.pose.orientation)
        innovation = self.normalize_angle(yaw - self.x[self.IDX_YAW, 0])

        if (
            self.max_sun_yaw_innovation > 0.0
            and abs(innovation) > self.max_sun_yaw_innovation
        ):
            self.get_logger().warn(
                f"Sun yaw rejected: innovation={innovation:.3f} rad"
            )
            return

        yaw_cov = self.valid_covariance_or_default(
            msg.pose.covariance[35],
            self.default_sun_yaw_cov,
        )
        yaw_cov = max(self.min_sun_yaw_cov, yaw_cov)

        z = np.array([[yaw]], dtype=np.float64)
        H = np.zeros((1, self.STATE_SIZE), dtype=np.float64)
        H[0, self.IDX_YAW] = 1.0
        R = np.array([[yaw_cov]], dtype=np.float64)

        self.measurement_update(
            z=z,
            H=H,
            R=R,
            angle_measurement_rows=[0],
        )

        self.last_stamp = stamp
        self.publish_estimate(stamp)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_from_wheel(
        self,
        pose: Tuple[float, float, float, float, float, float],
    ) -> None:
        px, py, pz, roll, pitch, yaw = pose

        if self.latest_imu_rpy is not None:
            roll, pitch, yaw = self.latest_imu_rpy

        self.x[self.IDX_X, 0] = px
        self.x[self.IDX_Y, 0] = py
        self.x[self.IDX_Z, 0] = pz
        self.x[self.IDX_ROLL, 0] = roll
        self.x[self.IDX_PITCH, 0] = pitch
        self.x[self.IDX_YAW, 0] = yaw

        self.normalize_state_angles()
        self.initialized = True

        self.get_logger().info("EKF initialized from wheel odometry.")

    def initialize_from_trn(
        self,
        trn_x: float,
        trn_y: float,
        trn_z: float,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> None:
        if self.latest_imu_rpy is not None:
            roll, pitch, yaw = self.latest_imu_rpy

        self.x[self.IDX_X, 0] = trn_x
        self.x[self.IDX_Y, 0] = trn_y
        self.x[self.IDX_Z, 0] = trn_z
        self.x[self.IDX_ROLL, 0] = roll
        self.x[self.IDX_PITCH, 0] = pitch
        self.x[self.IDX_YAW, 0] = yaw

        self.normalize_state_angles()
        self.initialized = True

        self.get_logger().info("EKF initialized from TRN pose.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_from_wheel_delta(
        self,
        prev_pose: Tuple[float, float, float, float, float, float],
        current_pose: Tuple[float, float, float, float, float, float],
    ) -> None:
        prev_x, prev_y, _, _, _, prev_yaw = prev_pose
        cur_x, cur_y, _, _, _, cur_yaw = current_pose

        dx_odom = cur_x - prev_x
        dy_odom = cur_y - prev_y
        dyaw = self.normalize_angle(cur_yaw - prev_yaw)

        # wheel odom frame에서의 delta를 robot body frame delta로 변환
        c_prev = math.cos(prev_yaw)
        s_prev = math.sin(prev_yaw)

        dx_body = c_prev * dx_odom + s_prev * dy_odom
        dy_body = -s_prev * dx_odom + c_prev * dy_odom
        original_dx_body = dx_body
        original_dy_body = dy_body
        original_dyaw = dyaw

        slip_guard_active = self.should_suppress_forward_wheel_delta(dx_body)
        self.wheel_slip_guard_active = slip_guard_active
        if slip_guard_active:
            dx_body *= self.slip_guard_linear_scale
            dy_body *= self.slip_guard_lateral_scale
            dyaw *= self.slip_guard_yaw_scale
            self.maybe_log_slip_guard(
                original_dx_body=original_dx_body,
                original_dy_body=original_dy_body,
                original_dyaw=original_dyaw,
                dx_body=dx_body,
                dy_body=dy_body,
                dyaw=dyaw,
            )

        yaw_est = self.x[self.IDX_YAW, 0]

        c = math.cos(yaw_est)
        s = math.sin(yaw_est)

        dx_map = c * dx_body - s * dy_body
        dy_map = s * dx_body + c * dy_body

        self.x[self.IDX_X, 0] += dx_map
        self.x[self.IDX_Y, 0] += dy_map
        self.x[self.IDX_YAW, 0] = self.normalize_angle(
            self.x[self.IDX_YAW, 0] + dyaw
        )

        # Jacobian
        F = np.eye(self.STATE_SIZE, dtype=np.float64)
        F[self.IDX_X, self.IDX_YAW] = -s * dx_body - c * dy_body
        F[self.IDX_Y, self.IDX_YAW] = c * dx_body - s * dy_body

        distance = math.sqrt(dx_body * dx_body + dy_body * dy_body)

        q_xy = self.base_process_noise_xy + self.process_noise_xy_per_m * abs(distance)
        q_z = self.base_process_noise_z
        q_rp = self.base_process_noise_rp
        q_yaw = self.base_process_noise_yaw + self.process_noise_yaw_per_rad * abs(dyaw)
        if slip_guard_active:
            multiplier = max(1.0, self.slip_guard_noise_multiplier)
            q_xy *= multiplier
            q_yaw *= multiplier

        Q = np.diag([q_xy, q_xy, q_z, q_rp, q_rp, q_yaw]).astype(np.float64)

        self.P = F @ self.P @ F.T + Q
        self.P = self.symmetrize_covariance(self.P)

        self.normalize_state_angles()

    def should_suppress_forward_wheel_delta(self, dx_body: float) -> bool:
        if not self.wheel_slip_guard_enabled:
            return False
        if dx_body <= self.slip_guard_min_forward_delta_m:
            return False

        now_ns = self.get_clock().now().nanoseconds
        return self.front_blocked or now_ns <= self.front_blocked_until_ns

    def maybe_log_slip_guard(
        self,
        original_dx_body: float,
        original_dy_body: float,
        original_dyaw: float,
        dx_body: float,
        dy_body: float,
        dyaw: float,
    ) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_slip_guard_log_ns < int(1.0e9):
            return
        self.last_slip_guard_log_ns = now_ns
        self.get_logger().warn(
            "Wheel slip guard active: front blocked "
            f"(min_depth={self.front_block_min_depth:.2f}m, "
            f"ratio={self.front_block_ratio:.2f}); "
            f"delta body ({original_dx_body:.3f}, {original_dy_body:.3f}, "
            f"{original_dyaw:.3f}rad) -> "
            f"({dx_body:.3f}, {dy_body:.3f}, {dyaw:.3f}rad)"
        )

    # ------------------------------------------------------------------
    # Measurement update
    # ------------------------------------------------------------------

    def measurement_update(
        self,
        z: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        angle_measurement_rows: List[int],
    ) -> None:
        """
        일반 Kalman update.

        angle_measurement_rows:
        - measurement residual 중 angle normalize가 필요한 row index 목록
        """
        z_pred = H @ self.x
        y = z - z_pred

        for row in angle_measurement_rows:
            y[row, 0] = self.normalize_angle(y[row, 0])

        S = H @ self.P @ H.T + R

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.get_logger().warn("EKF update skipped because innovation covariance is singular.")
            return

        K = self.P @ H.T @ S_inv

        self.x = self.x + K @ y

        I = np.eye(self.STATE_SIZE, dtype=np.float64)

        # Joseph form for numerical stability
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        self.P = self.symmetrize_covariance(self.P)

        self.normalize_state_angles()

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def publish_estimate(self, stamp: Time) -> None:
        if not self.initialized:
            return

        odom_msg = Odometry()

        odom_msg.header.stamp = stamp.to_msg()
        odom_msg.header.frame_id = self.frame_id
        odom_msg.child_frame_id = self.child_frame_id

        odom_msg.pose.pose.position.x = float(self.x[self.IDX_X, 0])
        odom_msg.pose.pose.position.y = float(self.x[self.IDX_Y, 0])
        odom_msg.pose.pose.position.z = float(self.x[self.IDX_Z, 0])

        q = self.quaternion_from_euler(
            roll=float(self.x[self.IDX_ROLL, 0]),
            pitch=float(self.x[self.IDX_PITCH, 0]),
            yaw=float(self.x[self.IDX_YAW, 0]),
        )

        odom_msg.pose.pose.orientation = q

        odom_msg.pose.covariance = self.covariance_to_ros_pose_covariance(self.P)

        odom_msg.twist.twist.linear.x = float(self.last_linear_v)
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = float(self.last_angular_z)

        odom_msg.twist.covariance = self.make_twist_covariance()

        self.estimated_odom_pub.publish(odom_msg)

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header = odom_msg.header
        pose_msg.pose = odom_msg.pose

        self.estimated_pose_pub.publish(pose_msg)

        if self.publish_tf_enabled:
            self.publish_tf(stamp, q)

    def publish_tf(self, stamp: Time, q: Quaternion) -> None:
        if self.tf_broadcaster is None:
            return

        transform = TransformStamped()

        transform.header.stamp = stamp.to_msg()
        transform.header.frame_id = self.frame_id
        transform.child_frame_id = self.child_frame_id

        transform.transform.translation.x = float(self.x[self.IDX_X, 0])
        transform.transform.translation.y = float(self.x[self.IDX_Y, 0])
        transform.transform.translation.z = float(self.x[self.IDX_Z, 0])

        transform.transform.rotation = q

        self.tf_broadcaster.sendTransform(transform)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def extract_pose_6d_from_odom(
        self,
        msg: Odometry,
    ) -> Tuple[float, float, float, float, float, float]:
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z

        roll, pitch, yaw = self.euler_from_quaternion(msg.pose.pose.orientation)

        return px, py, pz, roll, pitch, yaw

    @staticmethod
    def get_odom_time(msg: Odometry) -> Time:
        return Time.from_msg(msg.header.stamp)

    @staticmethod
    def valid_covariance_or_default(value: float, default: float) -> float:
        if not math.isfinite(value):
            return default

        if value <= 1.0e-12:
            return default

        if value > 1.0e6:
            return default

        return value

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def normalize_state_angles(self) -> None:
        self.x[self.IDX_ROLL, 0] = self.normalize_angle(self.x[self.IDX_ROLL, 0])
        self.x[self.IDX_PITCH, 0] = self.normalize_angle(self.x[self.IDX_PITCH, 0])
        self.x[self.IDX_YAW, 0] = self.normalize_angle(self.x[self.IDX_YAW, 0])

    @staticmethod
    def symmetrize_covariance(P: np.ndarray) -> np.ndarray:
        return 0.5 * (P + P.T)

    @staticmethod
    def covariance_to_ros_pose_covariance(P: np.ndarray) -> List[float]:
        """
        EKF state order가 ROS pose covariance order와 동일하게 구성되어 있음:
        [x, y, z, roll, pitch, yaw]
        """
        covariance = [0.0] * 36

        for row in range(6):
            for col in range(6):
                covariance[row * 6 + col] = float(P[row, col])

        return covariance

    @staticmethod
    def make_twist_covariance() -> List[float]:
        covariance = [0.0] * 36

        covariance[0] = 0.10      # linear x
        covariance[7] = 999.0     # linear y
        covariance[14] = 999.0    # linear z
        covariance[21] = 999.0    # angular x
        covariance[28] = 999.0    # angular y
        covariance[35] = 0.10     # angular z

        return covariance

    @staticmethod
    def euler_from_quaternion(q: Quaternion) -> Tuple[float, float, float]:
        """
        Quaternion -> roll, pitch, yaw
        """
        x = q.x
        y = q.y
        z = q.z
        w = q.w

        # roll
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # pitch
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        # yaw
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @staticmethod
    def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> Quaternion:
        """
        roll, pitch, yaw -> Quaternion
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q = Quaternion()
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy

        return q


def main(args=None):
    rclpy.init(args=args)

    node = EKFFusionNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
