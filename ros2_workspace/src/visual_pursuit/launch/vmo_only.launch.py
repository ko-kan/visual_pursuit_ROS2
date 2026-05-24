"""
VMO-only launch (Chapter 6 — estimation without camera control).

Nodes started:
  vmo_node          : Visual Motion Observer (pure observer)
  vmo_feedback_node : Simple feedback  u_e = -k_e * e_e  (eq 6.22)

For the full visual-pursuit configuration (Chapter 7, with camera
velocity control) use visual_pursuit.launch.py instead.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='visual_pursuit',
            executable='vmo_node',
            name='vmo_node',
            output='screen',
        ),
        Node(
            package='visual_pursuit',
            executable='vmo_feedback_node',
            name='vmo_feedback_node',
            output='screen',
        ),
    ])
