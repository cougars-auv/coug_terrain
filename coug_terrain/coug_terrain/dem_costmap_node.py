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

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from osgeo import gdal
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from sensor_msgs.msg import NavSatFix

from coug_terrain.utils.terrain_grid import (
    FREE,
    LATCHED_QOS,
    LETHAL,
    UNKNOWN,
    flat_grid,
    occupancy_grid,
    warp_to_map,
)


class DemCostmapNode(Node):
    def __init__(self) -> None:
        super().__init__("dem_costmap_node")

        self.declare_parameter("dem_file", "")
        self.declare_parameter("max_slope_degrees", 25.0)
        self.declare_parameter("resolution", 1.0)
        self.declare_parameter("outside_is_lethal", False)
        self.declare_parameter("fallback_size", 500.0)
        self.declare_parameter("origin_topic", "/origin")
        self.declare_parameter("output_topic", "terrain/dem/occupancy")
        self.declare_parameter("map_frame", "map")

        dem_file = self.get_parameter("dem_file").value
        self._max_slope_degrees = self.get_parameter("max_slope_degrees").value
        self._resolution = self.get_parameter("resolution").value
        self._outside_is_lethal = self.get_parameter("outside_is_lethal").value
        fallback_size = self.get_parameter("fallback_size").value
        origin_topic = self.get_parameter("origin_topic").value
        output_topic = self.get_parameter("output_topic").value
        self._map_frame = self.get_parameter("map_frame").value

        self._origin_sub = self.create_subscription(
            NavSatFix, origin_topic, self._origin_callback, qos_profile_system_default
        )
        self._output_pub = self.create_publisher(OccupancyGrid, output_topic, LATCHED_QOS)

        self._published = False

        if dem_file:
            dem_dataset = gdal.Open(dem_file, gdal.GA_ReadOnly)
            self._slope_dataset = gdal.DEMProcessing(
                "", dem_dataset, "slope", format="MEM", computeEdges=True
            )
            slope_min, slope_max = self._slope_dataset.GetRasterBand(1).ComputeRasterMinMax(False)
            self.get_logger().info(
                f"DEM loaded: {dem_dataset.RasterXSize}x{dem_dataset.RasterYSize} cells at "
                f"{dem_dataset.GetGeoTransform()[1]:.2f} m/cell, "
                f"slope {slope_min:.1f} to {slope_max:.1f} deg."
            )
        else:
            grid, min_east, min_north = flat_grid(fallback_size, self._resolution)
            self._output_pub.publish(
                occupancy_grid(
                    grid,
                    min_east,
                    min_north,
                    self._resolution,
                    self._map_frame,
                    self.get_clock().now().to_msg(),
                )
            )
            self._published = True
            self.get_logger().info(
                f"No 'dem_file' set; published a flat {grid.shape[1]}x{grid.shape[0]} costmap "
                f"at {self._resolution:.2f} m/cell with all cells free."
            )

        self.get_logger().info("Initialization complete.")

    def _origin_callback(self, msg: NavSatFix) -> None:
        if self._published:
            return

        grid_msg = self._convert_to_occupancy_grid(msg)
        if grid_msg is None:
            return

        self._output_pub.publish(grid_msg)
        self._published = True

        grid = np.asarray(grid_msg.data, dtype=np.int8)
        self.get_logger().info(
            f"Costmap published: {grid_msg.info.width}x{grid_msg.info.height} cells "
            f"at {self._resolution:.2f} m/cell "
            f"({np.count_nonzero(grid == LETHAL)} lethal, "
            f"{np.count_nonzero(grid == FREE)} free, "
            f"{np.count_nonzero(grid == UNKNOWN)} unknown), "
            f"anchored at lat {msg.latitude:.6f}, lon {msg.longitude:.6f}."
        )

    def _convert_to_occupancy_grid(self, msg: NavSatFix) -> OccupancyGrid | None:
        # Warp the slope raster into the map frame around the origin
        try:
            slope, min_east, min_north = warp_to_map(
                self._slope_dataset, msg, self._resolution, np.nan
            )
        except RuntimeError as e:
            self.get_logger().error(f"Failed to warp DEM into the map frame: {e}")
            return None

        grid = np.where(slope > self._max_slope_degrees, LETHAL, FREE).astype(np.int8)
        grid[np.isnan(slope)] = LETHAL if self._outside_is_lethal else UNKNOWN

        return occupancy_grid(
            grid,
            min_east,
            min_north,
            self._resolution,
            self._map_frame,
            self.get_clock().now().to_msg(),
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    dem_costmap_node = DemCostmapNode()
    try:
        rclpy.spin(dem_costmap_node)
    except KeyboardInterrupt:
        pass
    finally:
        dem_costmap_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
