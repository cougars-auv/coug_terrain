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


import math

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

_MASK_NODATA = 255
_BLOCK_ROWS = 512


class DsmCostmapNode(Node):
    def __init__(self) -> None:
        super().__init__("dsm_costmap_node")

        self.declare_parameter("dsm_file", "")
        self.declare_parameter("dem_file", "")
        self.declare_parameter("max_obstacle_height", 0.15)
        self.declare_parameter("min_obstacle_area", 0.5)
        self.declare_parameter("resolution", 0.5)
        self.declare_parameter("fallback_size", 500.0)
        self.declare_parameter("origin_topic", "/origin")
        self.declare_parameter("output_topic", "terrain/dsm/occupancy")
        self.declare_parameter("map_frame", "map")

        dsm_file = self.get_parameter("dsm_file").value
        dem_file = self.get_parameter("dem_file").value
        max_obstacle_height = self.get_parameter("max_obstacle_height").value
        min_obstacle_area = self.get_parameter("min_obstacle_area").value
        self._resolution = self.get_parameter("resolution").value
        fallback_size = self.get_parameter("fallback_size").value
        origin_topic = self.get_parameter("origin_topic").value
        output_topic = self.get_parameter("output_topic").value
        self._map_frame = self.get_parameter("map_frame").value

        self._origin_sub = self.create_subscription(
            NavSatFix, origin_topic, self._origin_callback, qos_profile_system_default
        )
        self._output_pub = self.create_publisher(OccupancyGrid, output_topic, LATCHED_QOS)

        self._published = False

        if dsm_file and dem_file:
            self._mask_dataset = self._load_mask(
                dsm_file, dem_file, max_obstacle_height, min_obstacle_area
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
                f"No 'dsm_file' or 'dem_file' set; published a flat "
                f"{grid.shape[1]}x{grid.shape[0]} costmap at {self._resolution:.2f} m/cell "
                f"with all cells free."
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
        # Warp the obstacle mask into the map frame around the origin
        try:
            mask, min_east, min_north = warp_to_map(
                self._mask_dataset, msg, self._resolution, _MASK_NODATA
            )
        except RuntimeError as e:
            self.get_logger().error(f"Failed to warp DSM into the map frame: {e}")
            return None

        grid = np.where(mask == _MASK_NODATA, UNKNOWN, mask).astype(np.int8)

        return occupancy_grid(
            grid,
            min_east,
            min_north,
            self._resolution,
            self._map_frame,
            self.get_clock().now().to_msg(),
        )

    def _load_mask(
        self, dsm_file: str, dem_file: str, max_obstacle_height: float, min_obstacle_area: float
    ) -> gdal.Dataset:
        dsm_dataset = gdal.Open(dsm_file, gdal.GA_ReadOnly)
        dem_dataset = gdal.Open(dem_file, gdal.GA_ReadOnly)
        width, height = dsm_dataset.RasterXSize, dsm_dataset.RasterYSize
        geo_transform = dsm_dataset.GetGeoTransform()
        if (width, height, geo_transform) != (
            dem_dataset.RasterXSize,
            dem_dataset.RasterYSize,
            dem_dataset.GetGeoTransform(),
        ):
            raise ValueError(f"'{dsm_file}' and '{dem_file}' must share the same grid.")

        mask_dataset = gdal.GetDriverByName("MEM").Create("", width, height, 1, gdal.GDT_Byte)
        mask_dataset.SetGeoTransform(geo_transform)
        mask_dataset.SetProjection(dsm_dataset.GetProjection())
        mask_band = mask_dataset.GetRasterBand(1)
        mask_band.SetNoDataValue(_MASK_NODATA)

        dsm_band = dsm_dataset.GetRasterBand(1)
        dem_band = dem_dataset.GetRasterBand(1)
        dsm_nodata = dsm_band.GetNoDataValue()
        dem_nodata = dem_band.GetNoDataValue()
        for start_row in range(0, height, _BLOCK_ROWS):
            block_height = min(_BLOCK_ROWS, height - start_row)
            dsm_block = dsm_band.ReadAsArray(0, start_row, width, block_height)
            dem_block = dem_band.ReadAsArray(0, start_row, width, block_height)
            obstacle_block = dsm_block - dem_block > max_obstacle_height
            mask_block = np.where(obstacle_block, LETHAL, FREE).astype(np.uint8)
            mask_block[(dsm_block == dsm_nodata) | (dem_block == dem_nodata)] = _MASK_NODATA
            mask_band.WriteArray(mask_block, 0, start_row)

        cell_area = abs(geo_transform[1] * geo_transform[5])
        min_cells = max(1, math.ceil(min_obstacle_area / cell_area))
        gdal.SieveFilter(mask_band, None, mask_band, min_cells, 8)

        mask = mask_band.ReadAsArray()
        obstacle_percent = 100.0 * np.mean(mask[mask != _MASK_NODATA] == LETHAL)
        self.get_logger().info(
            f"DSM loaded: {width}x{height} cells at {geo_transform[1]:.2f} m/cell, "
            f"{obstacle_percent:.2f}% above {max_obstacle_height:.2f} m."
        )
        return mask_dataset


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    dsm_costmap_node = DsmCostmapNode()
    try:
        rclpy.spin(dsm_costmap_node)
    except KeyboardInterrupt:
        pass
    finally:
        dsm_costmap_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
