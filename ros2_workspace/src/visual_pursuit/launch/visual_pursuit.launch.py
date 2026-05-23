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
            executable='error_control_node',
            name='error_control_node',
            output='screen',
        ),
    ])
