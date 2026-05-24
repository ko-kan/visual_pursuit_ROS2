import os

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class VMOFeedbackNode(Node):
    """
    Simple VMO feedback controller (Chapter 6, eq 6.22).

    Closes the feedback loop of the Visual Motion Observer without any
    camera motion control.  Use this node when you want to run the VMO
    in estimation-only mode (no robot/camera velocity command).

    Control law:
        u_e = -k_e * e_e          (eq 6.22, Hatanaka et al. 2015)

    This is the minimum counterpart to VMONode: it receives the
    estimation error published by the observer and returns the
    correction input u_e needed to drive e_e → 0.

    For the full visual-pursuit configuration (Chapter 7), use
    ErrorControlNode instead — it computes both u_c (camera command)
    and u_e (observer correction) from the combined error system.

    Subscribes
    ----------
    visual_pursuit/estimation_error : Float64MultiArray  e_e ∈ R^6

    Publishes
    ---------
    visual_pursuit/u_e : Float64MultiArray  u_e ∈ R^6
    """

    def __init__(self):
        super().__init__('vmo_feedback_node')
        self._load_params()

        self.create_subscription(
            Float64MultiArray,
            'visual_pursuit/estimation_error', self._cb_ee, 10)

        self._pub_ue = self.create_publisher(
            Float64MultiArray, 'visual_pursuit/u_e', 10)

    def _load_params(self):
        pkg = get_package_share_directory('visual_pursuit')
        with open(os.path.join(pkg, 'config', 'control.yaml')) as f:
            ctrl = yaml.safe_load(f)
        self._ke = float(ctrl['ke'])

    def _cb_ee(self, msg: Float64MultiArray):
        e_e = np.array(msg.data, dtype=float)
        # eq 6.22: u_e = -k_e * e_e
        u_e = -self._ke * e_e
        out = Float64MultiArray()
        out.data = u_e.tolist()
        self._pub_ue.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = VMOFeedbackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
