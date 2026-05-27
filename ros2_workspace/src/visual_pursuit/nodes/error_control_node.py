import os
import time

import numpy as np
import rclpy
import scipy.optimize as opt
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

from modules.jacobian import N_matrix
from modules.se3 import Ad, inv_se3, skew, vec


class ErrorControlNode(Node):
    """
    Estimation/Control Error System (Fig 7.4, Hatanaka et al. 2015).

    Computes errors, assembles ν = N e, and applies the control law
        u_nom = -K ν   (eq 7.15)
    where K = diag(K_c, k_e I_6) and ν = [ν_c; ν_e] ∈ R^12.

    The nominal camera control u_c_nom is then filtered by a **CBF-QP
    safety layer** (when enabled) based on Kanno et al. (Appl. Sci. 2024):

        u_c_safe = argmin_{u_c} ½ ‖u_c − u_c_nom‖²
                   s.t.  L_f h_i + L_g h_i · u_c + α h_i(x) ≥ 0   ∀ i

    Two barrier functions from the paper are implemented:

    [1] Field-of-View maintenance  (Theorem 1, eq. 28):
            h_fov = p_co^T T_M p_co − Δ_fov  ≥ 0
        where T_M = diag(−1/tan²φ_M, −1/tan²φ_M, 1),  φ_M < actual half-FoV.
        State:  p_co = ḡ[:3, 3]  (estimated object position in camera frame)

    [2] Occlusion Avoidance  (Theorem 2, eq. 34, sphere-obstacle approximation):
            h_occ,k = l_c,k + l_o,k − ‖p_co‖ − γ/‖p_co‖ − Δ_occ  ≥ 0
        where l_c,k = ‖p_wc − c_k‖ − r_k  (camera–obstacle clearance)
              l_o,k = ‖p_wo − c_k‖ − r_k  (target–obstacle clearance)
        State:  p_wc, p_wo  (absolute positions, world frame)
        ⟹ Requires camera absolute pose via topic  camera/absolute_pose

    The observer correction u_e is not modified by the CBF.

    Subscribes
    ----------
    visual_pursuit/estimated_pose   : Float64MultiArray  ḡ  4×4 row-major (16)
    visual_pursuit/estimation_error : Float64MultiArray  e_e ∈ R^6
    camera/absolute_pose            : Float64MultiArray  g_wc 4×4 row-major (16)
                                        Camera pose in world frame (for occlusion CBF).
                                        Occlusion constraints are skipped if not received.

    Publishes
    ---------
    visual_pursuit/control_output : Float64MultiArray  V^b_wc ∈ R^6
    visual_pursuit/u_e            : Float64MultiArray  u_e ∈ R^6
    visual_pursuit/output_nu      : Float64MultiArray  ν ∈ R^12  (monitoring)
    visual_pursuit/cbf_values     : Float64MultiArray  [h_fov, h_occ_0, …] (monitoring)
    camera/body_velocity          : geometry_msgs/Twist  V^b_wc (for VMO)
    """

    def __init__(self):
        super().__init__('error_control_node')
        self._load_params()

        self._g_bar = np.eye(4)
        self._e_e   = np.zeros(6)
        self._g_wc  = None     # camera absolute pose g_wc ∈ SE(3), None until received

        self.create_subscription(
            Float64MultiArray,
            'visual_pursuit/estimated_pose', self._cb_pose, 10)
        self.create_subscription(
            Float64MultiArray,
            'visual_pursuit/estimation_error', self._cb_ee, 10)
        self.create_subscription(
            Float64MultiArray,
            'camera/absolute_pose', self._cb_abs_pose, 10)
        # 静的環境: 最初の1メッセージだけ受け取ったら購読を解除する
        self._obs_sub = self.create_subscription(
            Float64MultiArray,
            'environment/obstacles', self._cb_obstacles, 10)

        self._pub_uc  = self.create_publisher(
            Float64MultiArray, 'visual_pursuit/control_output', 10)
        self._pub_ue  = self.create_publisher(
            Float64MultiArray, 'visual_pursuit/u_e', 10)
        self._pub_nu  = self.create_publisher(
            Float64MultiArray, 'visual_pursuit/output_nu', 10)
        self._pub_cbf = self.create_publisher(
            Float64MultiArray, 'visual_pursuit/cbf_values', 10)
        self._pub_bv  = self.create_publisher(
            Twist, 'camera/body_velocity', 10)

        # 制御ループタイマー（ROS 標準機能で Hz を固定）
        self.create_timer(1.0 / self._control_hz, self._compute_and_publish)
        self.get_logger().info(f'[control] loop rate: {self._control_hz} Hz')

    # ------------------------------------------------------------------
    # Parameter loading
    # ------------------------------------------------------------------

    def _load_params(self):
        pkg = get_package_share_directory('visual_pursuit')
        with open(os.path.join(pkg, 'config', 'control.yaml')) as f:
            ctrl = yaml.safe_load(f)

        self._control_hz = float(ctrl.get('control_hz', 5.0))
        self._g_d = np.array(ctrl['g_d'], dtype=float)   # 4×4

        Kc_param = ctrl['Kc']
        if isinstance(Kc_param, list):
            self._Kc = np.diag([float(v) for v in Kc_param])
        else:
            self._Kc = float(Kc_param) * np.eye(6)

        ke = float(ctrl['ke'])
        self._ke = ke
        self._K = np.block([
            [self._Kc,              np.zeros((6, 6))],
            [np.zeros((6, 6)), ke * np.eye(6)],
        ])  # 12×12

        # --- CBF parameters (Kanno et al. 2024) ----------------------
        cbf = ctrl.get('cbf', {})
        self._cbf_enabled = bool(cbf.get('enabled', False))
        self._cbf_alpha   = float(cbf.get('alpha', 1.0))

        # [1] FoV maintenance (eq. 28)
        phi_M = float(cbf.get('phi_M', np.pi / 4))     # design half-angle [rad]
        self._T_M      = np.diag([
            -1.0 / np.tan(phi_M) ** 2,
            -1.0 / np.tan(phi_M) ** 2,
            1.0,
        ])
        self._delta_fov = float(cbf.get('delta_fov', 1.0))

        # [2] Occlusion avoidance (eq. 34)
        self._gamma     = float(cbf.get('gamma',     1e-4))
        self._delta_occ = float(cbf.get('delta_occ', 0.2))

        # Obstacles — list of {center: [x,y,z], radius: r}
        self._obstacles = []
        for obs in cbf.get('obstacles', []):
            self._obstacles.append({
                'center': np.array(obs['center'], dtype=float),
                'radius': float(obs['radius']),
            })

        if self._cbf_enabled:
            self.get_logger().info(
                f'[CBF] enabled  α={self._cbf_alpha}  '
                f'φ_M={np.degrees(phi_M):.1f}°  Δ_fov={self._delta_fov}  '
                f'γ={self._gamma}  Δ_occ={self._delta_occ}  '
                f'n_obstacles={len(self._obstacles)}'
            )
            if self._obstacles:
                for i, obs in enumerate(self._obstacles):
                    self.get_logger().info(
                        f'[CBF]  obs[{i}] center={obs["center"].tolist()}  '
                        f'r={obs["radius"]}'
                    )
            else:
                self.get_logger().info('[CBF]  no obstacles — only h_fov active')

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _cb_pose(self, msg: Float64MultiArray):
        self._g_bar = np.array(msg.data).reshape(4, 4)

    def _cb_ee(self, msg: Float64MultiArray):
        self._e_e = np.array(msg.data)

    def _cb_abs_pose(self, msg: Float64MultiArray):
        """Receive camera absolute pose g_wc ∈ SE(3) (world frame)."""
        self._g_wc = np.array(msg.data).reshape(4, 4)

    def _cb_obstacles(self, msg: Float64MultiArray):
        """Receive obstacle list from Unity (environment/obstacles).

        Message layout: [N, cx0, cy0, cz0, r0, type0,  cx1, cy1, cz1, r1, type1, ...]
          type: 0.0 = sphere  (3-D distance)
                1.0 = cylinder (2-D XY horizontal distance, infinite height)
        Positions are in ROS world frame (converted by ObstacleManager.cs).
        Overrides the static obstacle list from control.yaml while CBF is enabled.
        """
        if not self._cbf_enabled:
            return

        data = msg.data
        if len(data) < 1:
            return

        n = int(data[0])
        if len(data) < 1 + 5 * n:
            self.get_logger().warn(
                f'[CBF] environment/obstacles: expected {1 + 5*n} values, '
                f'got {len(data)}. Ignoring.',
                throttle_duration_sec=2.0,
            )
            return

        new_obs = []
        for i in range(n):
            base = 1 + 5 * i
            new_obs.append({
                'center': np.array(data[base:base + 3], dtype=float),
                'radius': float(data[base + 3]),
                'type':   'cylinder' if int(round(data[base + 4])) == 1 else 'sphere',
            })

        self._obstacles = new_obs
        self.get_logger().info(
            f'[CBF] obstacles loaded from Unity ({n} obstacle(s)). '
            f'Unsubscribing — static environment assumed.'
        )

        # 静的環境なので購読を解除（以降このコールバックは呼ばれない）
        self.destroy_subscription(self._obs_sub)
        self._obs_sub = None

    # ------------------------------------------------------------------
    # CBF barrier functions  (Kanno et al. 2024)
    # ------------------------------------------------------------------

    def _barrier_fov(self, p_co: np.ndarray):
        """
        Field-of-View maintenance barrier  (Theorem 1, eq. 28).

            h_fov = p_co^T T_M p_co − Δ_fov

        Time derivative (eq. 30):
            ḣ_fov = 2 p_co^T T_M ṗ_co
                  = 2 p_co^T T_M (R_co v_wo^b − v_wc^b − ω_wc^b × p_co)

        L_f h_fov = 2 p_co^T T_M R_co v_wo^b  ≈ 0  (target-velocity unknown)

        L_g h_fov w.r.t. V_wc = [v_wc^b; ω_wc^b]:
            = 2 p_co^T T_M (−v_wc^b + skew(p_co) ω_wc^b)
            = −2 (T_M p_co)^T [I₃ | −skew(p_co)] V_wc

        Transformed to u_c via  V_wc = −Ad(g_d) u_c:
            L_g h_fov · u_c = 2 (T_M p_co)^T [I₃ | −skew(p_co)] Ad(g_d) u_c

        Returns
        -------
        h   : float   barrier value
        A   : (6,)    row vector s.t. constraint is  A @ u_c + α h ≥ 0
        """
        h = float(p_co @ self._T_M @ p_co) - self._delta_fov

        # L_g h_fov w.r.t. V_wc
        J_p      = np.hstack([np.eye(3), -skew(p_co)])          # 3×6
        Lgh_Vwc  = -2.0 * (self._T_M @ p_co) @ J_p              # (6,) row

        # Transform: V_wc = −Ad(g_d) u_c  ⟹  L_g h · u_c = Lgh_Vwc @ (−Ad(g_d))
        A = Lgh_Vwc @ (-Ad(self._g_d))                          # (6,)
        return h, A

    def _barrier_occ_k(self, p_co: np.ndarray, obs: dict):
        """
        Occlusion-avoidance barrier for obstacle k  (Theorem 2, eq. 34).

        Sphere-obstacle approximation:
            l_c,k = ‖p_wc − c_k‖ − r_k
            l_o,k = ‖p_wo − c_k‖ − r_k
            h_occ,k = l_c,k + l_o,k − ‖p_co‖ − γ/‖p_co‖ − Δ_occ

        Time derivative (static target, ṗ_wo = 0):
            ḣ_occ,k = ė_c,k-contrib + 0 − d/dt‖p_co‖ − d/dt(γ/‖p_co‖)

            Each term (p_co in camera frame, ṗ_co = −v^b_wc − ω^b_wc × p_co):

            d/dt l_c,k  = e_{c,k}^T Ṙ_wc ... = e_{c,k}^T R_wc v^b_wc
            d/dt l_o,k  = 0                          (static target)
            d/dt ‖p_co‖ = p̂_co^T ṗ_co             = −p̂_co^T v^b_wc
                           (ω term ⊥ p_co → vanishes)
            d/dt(γ/‖p_co‖) = (γ/‖p_co‖²) p̂_co^T v^b_wc

            ∴  ḣ_occ,k = [e_{c,k}^T R_wc + β p̂_co^T | 0₃^T] V^b_wc

            where  p̂_co = p_co / ‖p_co‖   (camera frame unit vector)
                   β    = 1 − γ/‖p_co‖²

        NOTE: e_co = (p_wc − p_wo)/‖p_co‖ = −R_wc p̂_co, so
              (e_c,k + β e_co)^T R_wc  = e_c,k^T R_wc − β p̂_co^T  ← wrong sign.
              The correct Lie derivative uses p̂_co directly in camera frame.

        Transformed to u_c via  V^b_wc = −Ad(g_d) u_c:
            A = [e_{c,k}^T R_wc + β p̂_co^T | 0₃^T] (−Ad(g_d))

        L_f h_occ,k  ≈ 0  (target velocity unknown; conservative)

        Returns (None, None) when g_wc not yet received or geometry degenerate.

        Parameters
        ----------
        p_co : (3,) object position in camera frame  (= ḡ[:3, 3])
        obs  : dict with 'center' (3,) and 'radius' float  (world frame)
        """
        if self._g_wc is None:
            return None, None

        p_wc = self._g_wc[:3, 3]
        R_wc = self._g_wc[:3, :3]
        p_wo = p_wc + R_wc @ p_co          # object position in world frame

        c_k      = obs['center']
        r_k      = obs['radius']
        obs_type = obs.get('type', 'sphere')

        if obs_type == 'cylinder':
            # Vertical cylinder: use 2-D XY horizontal distances.
            # Assumption: cylinder extends infinitely in height (Z direction),
            # so occlusion is determined solely by the horizontal projection.
            d_c = np.linalg.norm(p_wc[:2] - c_k[:2])
            d_o = np.linalg.norm(p_wo[:2] - c_k[:2])
        else:
            # Sphere: 3-D Euclidean distance
            d_c = np.linalg.norm(p_wc - c_k)
            d_o = np.linalg.norm(p_wo - c_k)

        if d_c < 1e-6 or d_o < 1e-6:      # degenerate: inside or on obstacle
            return None, None

        l_c_k = d_c - r_k
        l_o_k = d_o - r_k

        p_co_norm = np.linalg.norm(p_co)   # ‖p_co‖ (3-D, rotation-invariant)

        if p_co_norm < 1e-6:
            return None, None

        h = l_c_k + l_o_k - p_co_norm - self._gamma / p_co_norm - self._delta_occ

        # ── Gradient e_{c,k}: depends on obstacle shape ──────────────────
        obs_type = obs.get('type', 'sphere')

        if obs_type == 'cylinder':
            # Vertical cylinder: distance = 2-D XY horizontal distance.
            # Gradient has zero Z component (height direction irrelevant).
            #   ∂l_c,k/∂p_wc = [ex, ey, 0]   (world frame)
            diff_c_xy  = p_wc[:2] - c_k[:2]
            d_c_xy     = np.linalg.norm(diff_c_xy)
            if d_c_xy < 1e-6:
                return None, None
            e_c_k = np.array([diff_c_xy[0] / d_c_xy,
                               diff_c_xy[1] / d_c_xy,
                               0.0])                  # world frame, Z=0
        else:
            # Sphere: 3-D distance gradient
            e_c_k = (p_wc - c_k) / d_c              # world frame

        hat_p_co = p_co / p_co_norm             # camera frame: camera → object
        beta     = 1.0 - self._gamma / p_co_norm ** 2

        # L_g h_occ,k w.r.t. V^b_wc = [v^b_wc; ω^b_wc]
        #   translational: e_{c,k}^T R_wc + β p̂_co^T  (both are coeff of v^b_wc)
        #   angular:       0₃^T
        a        = e_c_k @ R_wc + beta * hat_p_co   # (3,) in camera frame
        Lgh_Vwc  = np.concatenate([a, np.zeros(3)])  # (6,)

        # Transform: V^b_wc = −Ad(g_d) u_c
        A = Lgh_Vwc @ (-Ad(self._g_d))              # (6,)
        return h, A

    # ------------------------------------------------------------------
    # CBF-QP safety filter
    # ------------------------------------------------------------------

    def _collect_constraints(self, g_bar: np.ndarray) -> list:
        """Collect all active CBF constraints.

        Returns list of dicts with:
            'h'    : float  — barrier value
            'A'    : (6,)   — gradient s.t. constraint is  A @ u_c + α h ≥ 0
            'name' : str
        """
        p_co = g_bar[:3, 3]

        if np.linalg.norm(p_co) < 1e-6:
            return []

        cons = []

        # [1] Field-of-view maintenance
        h_fov, A_fov = self._barrier_fov(p_co)
        cons.append({'h': h_fov, 'A': A_fov, 'name': 'h_fov'})

        # [2] Occlusion avoidance (one constraint per obstacle)
        for i, obs in enumerate(self._obstacles):
            h_occ, A_occ = self._barrier_occ_k(p_co, obs)
            if h_occ is not None:
                cons.append({'h': h_occ, 'A': A_occ, 'name': f'h_occ_{i}'})

        return cons

    def _apply_cbf_qp_with_cons(self, u_c_nom: np.ndarray,
                                 cons: list) -> np.ndarray:
        """CBF-QP safety filter  (eq. 42, Kanno et al. 2024).

        cons は _collect_constraints() の戻り値を渡す（呼び出し元で計算済み）。
        Solves:
            min_{u_c}  ½ ‖u_c − u_c_nom‖²
            s.t.       A_i @ u_c + α h_i(x) ≥ 0   for each barrier i

        Falls back to u_c_nom if all constraints are satisfied or solver fails.
        """
        if not cons:
            return u_c_nom

        # Fast path: nominal already safe
        alpha = self._cbf_alpha
        if all(c['A'] @ u_c_nom + alpha * c['h'] >= 0.0 for c in cons):
            return u_c_nom

        # Log violations
        for c in cons:
            val = c['A'] @ u_c_nom + alpha * c['h']
            if val < 0.0:
                self.get_logger().warn(
                    f"[CBF] '{c['name']}' violated  "
                    f"h={c['h']:.4f}  A·u+αh={val:.4f}",
                    throttle_duration_sec=0.5,
                )

        sc_cons = [
            {
                'type': 'ineq',
                'fun':  lambda uc, c=c: c['A'] @ uc + alpha * c['h'],
                'jac':  lambda uc, c=c: c['A'],
            }
            for c in cons
        ]

        result = opt.minimize(
            fun=lambda uc: 0.5 * float(np.dot(uc - u_c_nom, uc - u_c_nom)),
            x0=u_c_nom.copy(),
            jac=lambda uc: uc - u_c_nom,
            constraints=sc_cons,
            method='SLSQP',
            options={'ftol': 1e-9, 'maxiter': 200, 'disp': False},
        )

        if result.success or result.status in (0, 1):
            return result.x

        self.get_logger().warn(
            f'[CBF-QP] solver failed (status={result.status}): {result.message}',
            throttle_duration_sec=1.0,
        )
        return u_c_nom   # fallback

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def _compute_and_publish(self):
        # Control error  g_ce = g_d^{-1} ḡ
        g_ce = inv_se3(self._g_d) @ self._g_bar
        e_c  = vec(g_ce)

        # Error vector and output  (eq 7.9)
        e  = np.concatenate([e_c, self._e_e])
        N  = N_matrix(g_ce)
        nu = N @ e

        # Nominal control law  (eq 7.15):  u_nom = −K ν
        u_nom   = -self._K @ nu
        u_c_nom = u_nom[:6]

        # Observer correction u_e — published to VMO.
        #
        # The full N-matrix design gives:  u_e = ke · Ad_rot · e_c − ke · e_e
        # The ke·Ad_rot·e_c term compensates for the camera's NOMINAL control
        # motion (designed for V_wc = −Ad(g_d)·u_c_nom).  When CBF-QP modifies
        # u_c → u_c_cbf, V_wc changes but this term does not — creating a
        # persistent disturbance in ġ̄ that causes a constant steady-state
        # observer error.
        #
        # Root cause: the observer is at the correct pose (g_bar = g_true) only
        # when u_e = 0.  With the coupling term, u_e = ke·Ad_rot·e_c ≠ 0
        # whenever the camera is still moving (e_c ≠ 0), which CBF guarantees
        # permanently.
        #
        # Fix: decouple u_e from e_c.  The prediction step already uses the
        # true (CBF-modified) V_wc, so the correction needs only the estimation
        # error — giving u_e = 0 exactly when g_bar = g_true.
        
        # u_e = u_nom[6:] # theoretical input
        u_e = -self._ke * self._e_e   # pure estimation feedback, CBF-robust
        
        # ------ CBF-QP safety filter on u_c --------------------------
        if self._cbf_enabled:
            t0 = time.perf_counter()
            cbf_cons = self._collect_constraints(self._g_bar)
            t1 = time.perf_counter()
            u_c = self._apply_cbf_qp_with_cons(u_c_nom, cbf_cons)
            t2 = time.perf_counter()

            self.get_logger().info(
                f'[CBF timing] constraints={1000*(t1-t0):.2f} ms  '
                f'QP={1000*(t2-t1):.2f} ms  '
                f'total={1000*(t2-t0):.2f} ms',
                throttle_duration_sec=3.0,
            )

            cbf_msg      = Float64MultiArray()
            cbf_msg.data = [float(c['h']) for c in cbf_cons]
            self._pub_cbf.publish(cbf_msg)
        else:
            u_c = u_c_nom
        # --------------------------------------------------------------

        # V^b_wc = −Ad(g_d) u_c  (eq 7.7 / Fig 7.3)
        V_wc = -Ad(self._g_d) @ u_c

        vc_msg      = Float64MultiArray()
        vc_msg.data = V_wc.tolist()
        self._pub_uc.publish(vc_msg)

        # camera/body_velocity (Twist) — same V^b_wc for VMO
        twist           = Twist()
        twist.linear.x  = V_wc[0]
        twist.linear.y  = V_wc[1]
        twist.linear.z  = V_wc[2]
        twist.angular.x = V_wc[3]
        twist.angular.y = V_wc[4]
        twist.angular.z = V_wc[5]
        self._pub_bv.publish(twist)

        ue_msg      = Float64MultiArray()
        ue_msg.data = u_e.tolist()
        self._pub_ue.publish(ue_msg)

        nu_msg      = Float64MultiArray()
        nu_msg.data = nu.tolist()
        self._pub_nu.publish(nu_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ErrorControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
