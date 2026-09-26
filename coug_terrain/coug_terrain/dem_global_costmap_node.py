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

import array
import math
import time

import numpy as np
import numpy.typing as npt
import rclpy
from nav_msgs.msg import OccupancyGrid
from osgeo import gdal, osr
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_system_default,
)
from sensor_msgs.msg import NavSatFix

_UNKNOWN = -1
_FREE = 0
_LETHAL = 100

_WGS84_A = 6378137.0  # [m]
_WGS84_B = 6356752.314245  # [m]
_SECONDS_PER_YEAR = 365.25 * 86400.0

_EPSG_WGS84 = 4326
_EPSG_NAD83_2011_3D = 6319
_EPSG_ITRF2014_3D = 7912

gdal.UseExceptions()
osr.UseExceptions()


class DemGlobalCostmapNode(Node):
    def __init__(self) -> None:
        super().__init__("dem_global_costmap_node")

        self.declare_parameter("dem_file", "")
        self.declare_parameter("max_slope_degrees", 25.0)
        self.declare_parameter("resolution", 1.0)
        self.declare_parameter("outside_is_lethal", False)
        self.declare_parameter("fallback_size", 500.0)
        self.declare_parameter("origin_topic", "/origin")
        self.declare_parameter("output_topic", "terrain/occupancy")
        self.declare_parameter("map_frame", "map")

        dem_file = self.get_parameter("dem_file").value
        self._max_slope_degrees = self.get_parameter("max_slope_degrees").value
        self._resolution = self.get_parameter("resolution").value
        self._outside_is_lethal = self.get_parameter("outside_is_lethal").value
        fallback_size = self.get_parameter("fallback_size").value
        origin_topic = self.get_parameter("origin_topic").value
        output_topic = self.get_parameter("output_topic").value
        self._map_frame = self.get_parameter("map_frame").value

        self._published = False

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._output_pub = self.create_publisher(OccupancyGrid, output_topic, latched_qos)

        if dem_file:
            self._slope_dataset = self._load_dem(dem_file)
            self._origin_sub = self.create_subscription(
                NavSatFix, origin_topic, self._origin_callback, qos_profile_system_default
            )
        else:
            width = height = max(1, math.ceil(fallback_size / self._resolution))
            grid = np.full((height, width), _FREE, dtype=np.int8)
            self._publish_grid(
                grid, -0.5 * width * self._resolution, -0.5 * height * self._resolution
            )
            self.get_logger().info(
                f"No 'dem_file' set; published a flat {width}x{height} costmap at "
                f"{self._resolution:.2f} m/cell with all cells free."
            )

        self.get_logger().info("Initialization complete.")

    def _load_dem(self, path: str) -> gdal.Dataset:
        dem_dataset = gdal.Open(path, gdal.GA_ReadOnly)
        slope_dataset = gdal.DEMProcessing(
            "", dem_dataset, "slope", format="MEM", computeEdges=True
        )
        slope_min, slope_max = slope_dataset.GetRasterBand(1).ComputeRasterMinMax(False)

        self.get_logger().info(
            f"DEM loaded: {dem_dataset.RasterXSize}x{dem_dataset.RasterYSize} cells at "
            f"{dem_dataset.GetGeoTransform()[1]:.2f} m/cell, "
            f"slope {slope_min:.1f} to {slope_max:.1f} deg."
        )
        return slope_dataset

    @staticmethod
    def _geographic_crs(epsg: int) -> osr.SpatialReference:
        crs = osr.SpatialReference()
        crs.ImportFromEPSG(epsg)
        crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        return crs

    @staticmethod
    def _map_crs(origin: NavSatFix) -> osr.SpatialReference:
        # Raise the ellipsoid to the origin altitude to match navsat_odom ENU
        crs = osr.SpatialReference()
        crs.ImportFromProj4(
            f"+proj=aeqd +lat_0={origin.latitude} +lon_0={origin.longitude} "
            f"+a={_WGS84_A + origin.altitude} +b={_WGS84_B + origin.altitude} +units=m"
        )
        crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        return crs

    def _nad83_offset(self, origin: NavSatFix) -> tuple[float, float]:
        # Shift USGS 3DEP data from NAD83(2011) onto ITRF2014 (~WGS 84) at the current epoch
        epoch = 1970.0 + time.time() / _SECONDS_PER_YEAR
        itrf2014_crs = self._geographic_crs(_EPSG_ITRF2014_3D)
        itrf2014_crs.SetCoordinateEpoch(epoch)
        itrf2014_T_nad83 = osr.CoordinateTransformation(
            self._geographic_crs(_EPSG_NAD83_2011_3D), itrf2014_crs
        )
        lon, lat, _, _ = itrf2014_T_nad83.TransformPoint(
            origin.longitude, origin.latitude, origin.altitude, epoch
        )
        map_T_wgs84 = osr.CoordinateTransformation(
            self._geographic_crs(_EPSG_WGS84), self._map_crs(origin)
        )
        east, north, _ = map_T_wgs84.TransformPoint(lon, lat)
        return east, north

    def _origin_callback(self, msg: NavSatFix) -> None:
        if self._published:
            return

        try:
            map_dataset = gdal.Warp(
                "",
                self._slope_dataset,
                format="MEM",
                dstSRS=self._map_crs(msg).ExportToWkt(),
                xRes=self._resolution,
                yRes=self._resolution,
                resampleAlg="max",
                dstNodata=np.nan,
            )
            east_offset, north_offset = self._nad83_offset(msg)
        except RuntimeError as e:
            self.get_logger().error(f"Failed to warp DEM into the map frame: {e}")
            return

        slope = np.flipud(map_dataset.GetRasterBand(1).ReadAsArray())
        height, width = slope.shape
        geo_transform = map_dataset.GetGeoTransform()
        min_east = geo_transform[0] + east_offset
        min_north = geo_transform[3] + height * geo_transform[5] + north_offset

        grid = np.where(slope > self._max_slope_degrees, _LETHAL, _FREE).astype(np.int8)
        grid[np.isnan(slope)] = _LETHAL if self._outside_is_lethal else _UNKNOWN

        self._publish_grid(grid, min_east, min_north)
        self._published = True

        self.get_logger().info(
            f"Costmap published: {width}x{height} cells at {self._resolution:.2f} m/cell "
            f"({np.count_nonzero(grid == _LETHAL)} lethal, "
            f"{np.count_nonzero(grid == _FREE)} free, "
            f"{np.count_nonzero(grid == _UNKNOWN)} unknown), "
            f"anchored at lat {msg.latitude:.6f}, lon {msg.longitude:.6f}."
        )

    def _publish_grid(self, grid: npt.NDArray[np.int8], min_east: float, min_north: float) -> None:
        height, width = grid.shape

        occupancy_grid_msg = OccupancyGrid()
        occupancy_grid_msg.header.stamp = self.get_clock().now().to_msg()
        occupancy_grid_msg.header.frame_id = self._map_frame
        occupancy_grid_msg.info.resolution = self._resolution
        occupancy_grid_msg.info.width = width
        occupancy_grid_msg.info.height = height
        occupancy_grid_msg.info.origin.position.x = min_east
        occupancy_grid_msg.info.origin.position.y = min_north
        occupancy_grid_msg.info.origin.orientation.w = 1.0
        occupancy_grid_msg.data = array.array("b", grid.tobytes())

        self._output_pub.publish(occupancy_grid_msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    dem_global_costmap_node = DemGlobalCostmapNode()
    try:
        rclpy.spin(dem_global_costmap_node)
    except KeyboardInterrupt:
        pass
    finally:
        dem_global_costmap_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
