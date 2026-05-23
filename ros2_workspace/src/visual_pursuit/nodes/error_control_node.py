import os

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from modules.jacobian import N_matrix
from modules.se3 import inv_se3, vec


class ErrorControlNode(Node):
    """
    Estimation/Control Error System (Fig 7.4).

    Computes errors, assembles ν = N e, and applies the control law
        u = -K ν   (eq 7.15)
    where K = diag(K_c, k_e I_6) and ν = [ν_c; ν_e] ∈ R^12.

    Only u_c = -K_c ν_c (first 6 components) is published as the robot
    velocity command. The VMO handles u_e independently.

    Subscribes
    ----------
    visual_pursuit/estimated_pose   : Float64MultiArray  4×4 row-major (16)
    visual_pursuit/estimation_error : Float64MultiArray  e_e ∈ R^6

    Publishes
    ---------
    visual_pursuit/control_output : Float64MultiArray  u_c ∈ R^6
    visual_pursuit/output_nu      : Float64MultiArray  ν ∈ R^12  (for monitoring)
    """

    def __init__(self):
        super().__init__('error_control_node')
        self._load_params()

        self._g_bar = np.eye(4)
        self._e_e = np.zeros(6)

        self.create_subscription(
            Float64MultiArray,
            'visual_pursuit/estimated_pose', self._cb_pose, 10)
        self.create_subscription(
            Float64MultiArray,
            'visual_pursuit/estimation_error', self._cb_ee, 10)

        self._pub_uc = self.create_publisher(
            Float64MultiArray, 'visual_pursuit/control_output', 10)
        self._pub_nu = self.create_publisher(
            Float64MultiArray, 'visual_pursuit/output_nu', 10)

    def _load_params(self):
        pkg = get_package_share_directory('visual_pursuit')
        with open(os.path.join(pkg, 'config', 'control.yaml')) as f:
            ctrl = yaml.safe_load(f)

        self._g_d = np.array(ctrl['g_d'], dtype=float)   # 4×4

        Kc_param = ctrl['Kc']
        if isinstance(Kc_param, list):
            self._Kc = np.diag([float(v) for v in Kc_param])
        else:
            self._Kc = float(Kc_param) * np.eye(6)

        ke = float(ctrl['ke'])
        self._K = np.block([
            [self._Kc,              np.zeros((6, 6))],
            [np.zeros((6, 6)), ke * np.eye(6)],
        ])  # 12×12

    def _cb_pose(self, msg: Float64MultiArray):
        self._g_bar = np.array(msg.data).reshape(4, 4)
        self._compute_and_publish()

    def _cb_ee(self, msg: Float64MultiArray):
        self._e_e = np.array(msg.data)

    def _compute_and_publish(self):
        # Control error: g_ce = g_d^{-1} ḡ
        g_ce = inv_se3(self._g_d) @ self._g_bar
        e_c = vec(g_ce)

        # Error vector and output (eq 7.9)
        e = np.concatenate([e_c, self._e_e])
        N = N_matrix(g_ce)
        nu = N @ e

        # Control law (eq 7.15): u = -K ν
        u = -self._K @ nu
        u_c = u[:6]

        uc_msg = Float64MultiArray()
        uc_msg.data = u_c.tolist()
        self._pub_uc.publish(uc_msg)

        nu_msg = Float64MultiArray()
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
