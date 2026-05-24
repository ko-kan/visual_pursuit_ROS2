import os

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

from modules.camera import Camera
from modules.jacobian import image_jacobian, image_jacobian_pinv
from modules.pose import Pose


class VMONode(Node):
    """
    Visual Motion Observer (VMO) — pure observer without internal feedback.

    Integrates the observer dynamics (eq 6.9):
        ġ̄ = -V̂^b_wc ḡ - ḡ û_e

    u_e is received from an **external** controller via topic
    ``visual_pursuit/u_e``.  The node is deliberately kept as a pure
    observer (RRBM Model + Camera Model + J†) so that the feedback gain
    k_e lives in a separate node, keeping the passivity-based
    decomposition explicit (Fig 6.12 / Fig 7.4, Hatanaka et al. 2015).

    Subscribes
    ----------
    camera/image_raw            : sensor_msgs/Image          (RGB, from Unity)
    camera/body_velocity        : geometry_msgs/Twist         (V^b_wc)
    visual_pursuit/u_e          : std_msgs/Float64MultiArray  u_e ∈ R^6

    Publishes
    ---------
    visual_pursuit/estimated_pose   : Float64MultiArray  4×4 SE(3), row-major (16)
    visual_pursuit/estimation_error : Float64MultiArray  e_e ∈ R^6
    """

    def __init__(self):
        super().__init__('vmo_node')
        self._load_params()

        self._g_bar = Pose(self._g_0.copy())
        # u_e ∈ R^6  (theory notation: u_e = -k_e * e_e, eq 6.22)
        # Initialised to zero; updated by _cb_u_e when the controller publishes.
        self._u_e = np.zeros(6)
        self._V_wc = np.zeros(6)
        self._f_measured: np.ndarray | None = None
        self._latest_img: np.ndarray | None = None
        self._last_ns: int | None = None

        self._sub_image = self.create_subscription(
            Image, 'camera/image_raw', self._cb_image, 10)
        self._sub_image   # prevent GC

        self._sub_vel = self.create_subscription(
            Twist, 'camera/body_velocity', self._cb_velocity, 10)
        self._sub_vel     # prevent GC

        self._sub_ue = self.create_subscription(
            Float64MultiArray, 'visual_pursuit/u_e', self._cb_u_e, 10)
        self._sub_ue      # prevent GC

        self._pub_pose = self.create_publisher(
            Float64MultiArray, 'visual_pursuit/estimated_pose', 10)
        self._pub_ee = self.create_publisher(
            Float64MultiArray, 'visual_pursuit/estimation_error', 10)

    def _load_params(self):
        pkg = get_package_share_directory('visual_pursuit')

        with open(os.path.join(pkg, 'data', 'targets.yaml')) as f:
            targets = yaml.safe_load(f)
        self._p_oi_list = [np.array(p, dtype=float)
                           for p in targets['feature_points']]

        with open(os.path.join(pkg, 'data', 'camera.yaml')) as f:
            cam = yaml.safe_load(f)

        with open(os.path.join(pkg, 'config', 'control.yaml')) as f:
            ctrl = yaml.safe_load(f)

        self._camera = Camera(
            focal_length=cam['focal_length'],
            cx=cam['cx'],
            cy=cam['cy'],
            p_oi_list=self._p_oi_list,
            vertex_colors=targets['vertex_colors'],
        )
        self._g_d = np.array(ctrl['g_d'], dtype=float)
        # VMO 初期推定値。g_0 が未定義なら g_d にフォールバック
        self._g_0 = np.array(ctrl.get('g_0', ctrl['g_d']), dtype=float)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _cb_image(self, msg: Image):
        # Unity publishes RGB; convert to BGR and flip vertically
        img_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.flip(img_bgr, 0)
        self._latest_img = img_bgr

        result = self._camera.feature_from_image(img_bgr)
        if result is None:
            cv2.imshow('VMO: detected (green) / model (red)', img_bgr)
            cv2.waitKey(1)
            return   # skip step if any vertex not detected

        self._f_measured = result
        self._step()

    def _cb_velocity(self, msg: Twist):
        # Store latest velocity; step is driven by image callback
        self._V_wc = np.array([
            msg.linear.x,  msg.linear.y,  msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z,
        ])

    def _cb_u_e(self, msg: Float64MultiArray):
        """Receive observer correction input u_e ∈ R^6 from the controller.

        Theory convention (eq 6.22): u_e = -k_e * e_e.
        The observer dynamics (eq 6.9) require V_second = -u_e, so
        pose.step is called with ``-self._u_e``.
        """
        self._u_e = np.array(msg.data, dtype=float)

    # ------------------------------------------------------------------
    # Observer integration step
    # ------------------------------------------------------------------

    def _step(self):
        now = self.get_clock().now().nanoseconds
        if self._last_ns is None:
            self._last_ns = now
            return
        dt = (now - self._last_ns) * 1e-9
        self._last_ns = now

        if self._f_measured is None or dt <= 0.0:
            return

        g_bar = self._g_bar.g

        # Image feature error (normalized coordinates)
        f_model = self._camera.feature_from_model(g_bar)
        f_e = self._f_measured - f_model

        # Estimation error via image Jacobian pseudo-inverse (eq 6.16)
        J = image_jacobian(g_bar, self._p_oi_list)
        e_e = image_jacobian_pinv(J) @ f_e

        # Integrate observer: ġ̄ = -V̂_wc ḡ - ḡ V̂(u_e)   (eq 6.9)
        # pose.step(dt, V_wc, V_second) computes:
        #   g_dot = -wedge(V_wc) @ g + g @ wedge(V_second)
        # For eq 6.9 we need V_second = -u_e.
        self._g_bar.step(dt, self._V_wc, -self._u_e)

        if self._latest_img is not None:
            self._show_debug(self._latest_img, self._f_measured)
        pose_msg = Float64MultiArray()
        pose_msg.data = self._g_bar.g.flatten().tolist()
        self._pub_pose.publish(pose_msg)

        ee_msg = Float64MultiArray()
        ee_msg.data = e_e.tolist()
        self._pub_ee.publish(ee_msg)

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    # 四面体の全辺 (頂点インデックスのペア)
    _TETRA_EDGES = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]

    def _show_debug(self, img: np.ndarray,
                    f_measured: np.ndarray) -> None:
        """
        Overlays per-vertex colored markers on the image.
          - Detected centroid : hollow circle in vertex color
          - Model prediction  : cross marker in vertex color + wireframe edges
            (computed from the post-step g_bar via feature_from_model)
          - Error vector      : white line between detected and model
        f_measured is in normalized coords.
        Coordinates are clamped to image bounds.
        """
        vis = img.copy()
        h, w = vis.shape[:2]
        f = self._camera.f
        cx = self._camera.cx
        cy = self._camera.cy
        colors = self._camera.vertex_bgr_colors
        n = len(self._p_oi_list)

        # ステップ後の最新推定姿勢でモデル予測を計算
        f_model = self._camera.feature_from_model(self._g_bar.g)

        def to_pixel(nx: float, ny: float) -> tuple[int, int]:
            u = int(np.clip(nx * f + cx, 0, w - 1))
            v = int(np.clip(ny * f + cy, 0, h - 1))
            return (u, v)

        # モデル予測点を収集
        pts_mod: list[tuple[int, int] | None] = []
        for i in range(n):
            nx_mod = f_model[2 * i]
            ny_mod = f_model[2 * i + 1]
            if np.isfinite(nx_mod) and np.isfinite(ny_mod):
                pts_mod.append(to_pixel(nx_mod, ny_mod))
            else:
                pts_mod.append(None)

        # 推定剛体ワイヤーフレーム: モデル予測頂点を辺で結ぶ (薄いグレー)
        for a, b in self._TETRA_EDGES:
            if pts_mod[a] is not None and pts_mod[b] is not None:
                cv2.line(vis, pts_mod[a], pts_mod[b], (160, 160, 160), 1)

        for i in range(n):
            color = colors[i]

            # 検出頂点: 対応色の中抜き円
            pt_det = to_pixel(f_measured[2 * i], f_measured[2 * i + 1])
            cv2.circle(vis, pt_det, 10, color, 2)

            if pts_mod[i] is not None:
                # モデル予測: 対応色の十字
                cv2.drawMarker(vis, pts_mod[i], color, cv2.MARKER_CROSS, 16, 2)
                # 誤差ベクトル: 白線
                cv2.line(vis, pt_det, pts_mod[i], (255, 255, 255), 1)

        cv2.imshow('VMO: feature error', vis)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = VMONode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
