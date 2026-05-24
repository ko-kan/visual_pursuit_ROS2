import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'visual_pursuit'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'data'),   glob('data/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TODO',
    maintainer_email='todo@example.com',
    description='Visual pursuit via estimation/control error system (Hatanaka et al. 2015)',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'vmo_node = nodes.vmo_node:main',
            'vmo_feedback_node = nodes.vmo_feedback_node:main',
            'error_control_node = nodes.error_control_node:main',
        ],
    },
)
