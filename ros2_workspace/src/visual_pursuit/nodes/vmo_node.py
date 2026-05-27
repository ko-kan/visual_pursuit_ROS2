import os
import threading
import time

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
    Visual Motion Observer (VMO) — Predictor–Corrector 型実装。

    スレッド / スピン設計
    ────────────────────
    [メインスレッド]
        spin_once ループ + cv2.imshow / cv2.waitKey
        └─ cv2 GUI をメインスレッドで呼ぶことで X11 スレッド安全性を保証
        └─ spin_once で ROS コールバック・タイマーを処理

    [画像処理スレッド (daemon)]
        _cb_image が保存した最新 ROS メッセージをポーリング
        └─ frombuffer / cvtColor / flip / feature_from_image / 描画
        └─ _f_measured (observer 補正用) と _vis_frame (表示用) を更新

    [ROS コールバック (メインスレッド内の spin_once で実行)]
        _cb_image         : 最新メッセージ参照を保存するだけ（高速）
        _cb_velocity      : V_wc を更新
        _cb_u_e           : u_e を更新
        _integration_step : g_bar 積分、e_e 計算・公開

    Subscribes
    ----------
    camera/image_raw          : sensor_msgs/Image
    camera/body_velocity      : geometry_msgs/Twist
    visual_pursuit/u_e        : std_msgs/Float64MultiArray

    Publishes
    ---------
    visual_pursuit/estimated_pose   : Float64MultiArray  4×4 SE(3) row-major
    visual_pursuit/estimation_error : Float64MultiArray  e_e ∈ R^6
    """

    def __init__(self):
        super().__init__('vmo_node')
        self._load_params()

        self._g_bar   = Pose(self._g_0.copy())
        self._u_e     = np.zeros(6)
        self._V_wc    = np.zeros(6)
        self._last_ns: int | None = None

        # ── スレッド間共有変数 ──────────────────────────────────────────
        # Python の GIL により、オブジェクト参照の代入はアトミック。
        # ロックなしで読み書き可能（最悪 1 ステップ古い値を読むが制御上許容）。

        self._latest_msg  = None   # _cb_image 書き, 処理スレッド読み
        self._f_measured  = None   # 処理スレッド書き, _integration_step 読み
        self._vis_g_bar   = None   # _integration_step 書き, 処理スレッド読み
        self._vis_frame   = None   # 処理スレッド書き, メインスレッド読み

        # ── Subscriptions / Publishers ──────────────────────────────────
        self.create_subscription(Image, 'camera/image_raw', self._cb_image, 10)
        self.create_subscription(Twist, 'camera/body_velocity', self._cb_velocity, 10)
        self.create_subscription(
            Float64MultiArray, 'visual_pursuit/u_e', self._cb_u_e, 10)

        self._pub_pose = self.create_publisher(
            Float64MultiArray, 'visual_pursuit/estimated_pose', 10)
        self._pub_ee = self.create_publisher(
            Float64MultiArray, 'visual_pursuit/estimation_error', 10)

        # ── Predictor タイマー ───────────────────────────────────────────
        self.create_timer(1.0 / self._integration_hz, self._integration_step)

        # ── 画像処理スレッド ─────────────────────────────────────────────
        self._stop_proc = threading.Event()
        self._proc_thread = threading.Thread(
            target=self._process_loop, daemon=True, name='vmo_proc')
        self._proc_thread.start()

        self.get_logger().info(
            f'[VMO] integration={self._integration_hz} Hz  '
            f'image processing in background thread, GUI in main thread')

    # ------------------------------------------------------------------
    # Parameter loading
    # ------------------------------------------------------------------

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
        self._g_0 = np.array(ctrl.get('g_0', ctrl['g_d']), dtype=float)
        self._integration_hz  = float(ctrl.get('integration_hz',  50.0))
        self._image_proc_hz   = float(ctrl.get('image_proc_hz',   15.0))

    # ------------------------------------------------------------------
    # ROS コールバック（メインスレッド内 spin_once で実行）
    # ------------------------------------------------------------------

    def _cb_image(self, msg: Image):
        """最新メッセージ参照を保存するだけ（重い処理は処理スレッドへ）。"""
        self._latest_msg = msg

    def _cb_velocity(self, msg: Twist):
        self._V_wc = np.array([
            msg.linear.x,  msg.linear.y,  msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z,
        ])

    def _cb_u_e(self, msg: Float64MultiArray):
        self._u_e = np.array(msg.data, dtype=float)

    # ------------------------------------------------------------------
    # [Predictor] 高レート積分タイマー（メインスレッド内 spin_once で実行）
    # ------------------------------------------------------------------

    def _integration_step(self):
        now = self.get_clock().now().nanoseconds
        if self._last_ns is None:
            self._last_ns = now
            return
        dt = (now - self._last_ns) * 1e-9
        self._last_ns = now
        if dt <= 0.0:
            return

        g_bar = self._g_bar.g

        f_measured = self._f_measured   # アトミック読み込み
        if f_measured is not None:
            f_model = self._camera.feature_from_model(g_bar)
            f_e     = f_measured - f_model
            J       = image_jacobian(g_bar, self._p_oi_list)
            e_e     = image_jacobian_pinv(J) @ f_e
        else:
            e_e = np.zeros(6)

        self._g_bar.step(dt, self._V_wc, -self._u_e)

        # 処理スレッド用スナップショット（新配列 → 参照代入でアトミック）
        self._vis_g_bar = self._g_bar.g.copy()

        pose_msg      = Float64MultiArray()
        pose_msg.data = self._g_bar.g.flatten().tolist()
        self._pub_pose.publish(pose_msg)

        ee_msg      = Float64MultiArray()
        ee_msg.data = e_e.tolist()
        self._pub_ee.publish(ee_msg)

    # ------------------------------------------------------------------
    # [画像処理スレッド]  frombuffer / feature_from_image / 描画
    # ------------------------------------------------------------------

    _TETRA_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    def _process_loop(self) -> None:
        """
        重い画像処理をメインスレッドとは独立して実行する。

        image_proc_hz でレート制限することで feature_from_image の
        呼び出しを抑え、処理スレッドの負荷を一定に保つ。
        """
        last_msg  = None
        last_proc = 0.0
        interval  = 1.0 / self._image_proc_hz   # 例: 15 Hz → 67 ms

        while not self._stop_proc.is_set():
            msg = self._latest_msg
            now = time.monotonic()

            if (msg is not None
                    and msg is not last_msg
                    and (now - last_proc) >= interval):
                last_msg  = msg
                last_proc = now
                self._process_frame(msg)
            else:
                time.sleep(0.005)   # 5ms ポーリング

    def _process_frame(self, msg: Image) -> None:
        """1 フレーム分の処理: デコード → 特徴点検出 → 描画 → _vis_frame 更新。"""
        # ── デコード ──────────────────────────────────────────────────
        img_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.flip(img_bgr, 0)

        # ── 特徴点検出 ────────────────────────────────────────────────
        result = self._camera.feature_from_image(img_bgr)
        if result is None:
            return   # 検出失敗: _vis_frame を更新しない（最後の成功フレームを保持）

        self._f_measured = result       # アトミック代入

        # ── デバッグ描画 ──────────────────────────────────────────────
        g_bar = self._vis_g_bar         # スナップショット（古い値でも可）
        if g_bar is None:
            return   # g_bar 未初期化: _vis_frame を更新しない

        vis    = img_bgr.copy()
        h, w   = vis.shape[:2]
        f      = self._camera.f
        cx     = self._camera.cx
        cy     = self._camera.cy
        colors = self._camera.vertex_bgr_colors
        n      = len(self._p_oi_list)

        f_model = self._camera.feature_from_model(g_bar)

        def to_pixel(nx: float, ny: float) -> tuple[int, int]:
            return (int(np.clip(nx * f + cx, 0, w - 1)),
                    int(np.clip(ny * f + cy, 0, h - 1)))

        pts_mod: list[tuple[int, int] | None] = []
        for i in range(n):
            nx_m, ny_m = float(f_model[2 * i]), float(f_model[2 * i + 1])
            pts_mod.append(
                to_pixel(nx_m, ny_m) if np.isfinite(nx_m) and np.isfinite(ny_m)
                else None)

        for a, b in self._TETRA_EDGES:
            if pts_mod[a] is not None and pts_mod[b] is not None:
                cv2.line(vis, pts_mod[a], pts_mod[b], (160, 160, 160), 1)

        for i in range(n):
            color  = colors[i]
            pt_det = to_pixel(result[2 * i], result[2 * i + 1])
            cv2.circle(vis, pt_det, 10, color, 2)
            if pts_mod[i] is not None:
                cv2.drawMarker(vis, pts_mod[i], color, cv2.MARKER_CROSS, 16, 2)
                cv2.line(vis, pt_det, pts_mod[i], (255, 255, 255), 1)

        self._vis_frame = vis           # アトミック代入


def main(args=None):
    rclpy.init(args=args)
    node = VMONode()

    # spin_once を使い、メインスレッドで cv2 GUI を回す
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    try:
        while rclpy.ok():
            # ROS コールバック・タイマーを最大 10ms 待って 1 つ処理
            executor.spin_once(timeout_sec=0.01)

            # cv2 表示はメインスレッドで（X11 スレッド安全）
            frame = node._vis_frame
            if frame is not None:
                cv2.imshow('VMO: feature error', frame)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        pass
    finally:
        node._stop_proc.set()
        node._proc_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
