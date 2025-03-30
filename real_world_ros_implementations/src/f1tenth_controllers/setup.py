from setuptools import setup
import os
from glob import glob

package_name = 'f1tenth_controllers'

def get_model_paths():
    model_paths = []
    # Walk through the "models" directory recursively
    for root, _, files in os.walk("models"):
        # For each file, determine its target installation path
        files_list = [
            os.path.join(root, file)
            for file in files
        ]
        if files_list:
            # Target directory: share/<package_name>/<root>
            target_dir = os.path.join("share", package_name, root)
            model_paths.append((target_dir, files_list))
    return model_paths

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        *get_model_paths(),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yukang',
    maintainer_email='cao00125@umn.edu',
    description= 'ROS2 controllers for CU-MPPI. See [https://arxiv.org/pdf/2503.05819]',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'path_follower_node = f1tenth_controllers.p_controller:main',
            'mppi_costmap_node = f1tenth_controllers.mppi_ros2_kinematic:main',
            'cumppi_costmap_node = f1tenth_controllers.cumppi_ros2_kinematic:main',
            'pose_collector_node = f1tenth_controllers.amcl_pose_collector_node:main',
            'ackermann_to_twist = f1tenth_controllers.ackermann_to_twist:main',
        ],
    },
)
