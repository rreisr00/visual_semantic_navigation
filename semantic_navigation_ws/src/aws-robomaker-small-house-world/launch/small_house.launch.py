# /*******************************************************************************
# * Copyright 2019 ROBOTIS CO., LTD.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

# /* Author: Darby Lim */

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    world_file_name = 'small_house.world'
    package_dir = get_package_share_directory('aws_robomaker_small_house_world')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world = LaunchConfiguration('world')
    gui = LaunchConfiguration('gui')

    # Set GZ_SIM_RESOURCE_PATH to include AWS models directory for Harmonic
    model_path = os.path.join(package_dir, 'models')
    set_model_path = SetEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        model_path
    )

    gazebo_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        condition=IfCondition(gui),
        launch_arguments={
            'gz_args': ['-r ', world],
            'on_exit_shutdown': 'true',
        }.items(),
    )

    gazebo_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        condition=UnlessCondition(gui),
        launch_arguments={
            'gz_args': ['-r -s ', world],
            'on_exit_shutdown': 'true',
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
          'world',
          default_value=os.path.join(package_dir, 'worlds', world_file_name),
          description='SDF world file'),
        DeclareLaunchArgument(
            name='gui',
            default_value='true'
        ),
        DeclareLaunchArgument(
            name='use_sim_time',
            default_value='true'
        ),
        set_model_path,
        gazebo_gui,
        gazebo_headless,
    ])


if __name__ == '__main__':
    generate_launch_description()
