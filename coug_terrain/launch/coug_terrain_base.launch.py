# Copyright 2026 BYU FROST Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.action import Action
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node


def load_launch_params(path: str, top_key: str) -> dict[str, Any]:
    try:
        with open(path) as config_file:
            config = yaml.safe_load(config_file)
        params = config[top_key]["coug_terrain_launch"]["ros__parameters"]
        return dict(params)
    except (KeyError, TypeError, OSError):
        return {}


def launch_setup(context: LaunchContext, *args: Any, **kwargs: Any) -> list[Action]:
    use_sim_time = LaunchConfiguration("use_sim_time")

    scenario_param_path = LaunchConfiguration("scenario_param_file").perform(context)

    config_dir = os.environ["CONFIG_DIR"]
    coug_terrain_dir = get_package_share_directory("coug_terrain")

    fleet_param_file = PathJoinSubstitution(
        [EnvironmentVariable("CONFIG_DIR"), "fleet", "coug_terrain_params.yaml"]
    )
    scenario_param_file = scenario_param_path or fleet_param_file

    fleet_param_path = os.path.join(config_dir, "fleet", "coug_terrain_params.yaml")

    launch_params = {
        **load_launch_params(fleet_param_path, "/**"),
        **load_launch_params(scenario_param_path, "/**"),
    }
    dem_filename = launch_params.get("dem_file")
    if not dem_filename:
        return []
    dem_file = os.path.join(coug_terrain_dir, "dem", dem_filename)

    return [
        Node(
            package="coug_terrain",
            executable="dem_global_costmap",
            name="dem_global_costmap_node",
            parameters=[
                fleet_param_file,
                scenario_param_file,
                {
                    "use_sim_time": use_sim_time,
                    "dem_file": dem_file,
                    "map_frame": "map",
                },
            ],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
            ),
            DeclareLaunchArgument(
                "agent_list",
                default_value="[auv0]",
            ),
            DeclareLaunchArgument(
                "scenario_param_file",
                default_value="",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
