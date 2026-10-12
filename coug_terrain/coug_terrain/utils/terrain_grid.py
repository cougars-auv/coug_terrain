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
from builtin_interfaces.msg import Time
from nav_msgs.msg import OccupancyGrid
from osgeo import gdal, osr
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix

UNKNOWN = -1
FREE = 0
LETHAL = 100

LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
)

_WGS84_A = 6378137.0  # [m]
_WGS84_B = 6356752.314245  # [m]
_SECONDS_PER_YEAR = 365.25 * 86400.0

_EPSG_WGS84 = 4326
_EPSG_NAD83_2011_3D = 6319
_EPSG_ITRF2020_3D = 9989

gdal.UseExceptions()
osr.UseExceptions()


def _geographic_crs(epsg: int) -> osr.SpatialReference:
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(epsg)
    crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return crs


def _map_crs(origin: NavSatFix) -> osr.SpatialReference:
    crs = osr.SpatialReference()
    crs.ImportFromProj4(
        f"+proj=aeqd +lat_0={origin.latitude} +lon_0={origin.longitude} "
        f"+a={_WGS84_A + origin.altitude} +b={_WGS84_B + origin.altitude} +units=m"
    )
    crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return crs


def warp_to_map(
    dataset: gdal.Dataset, origin: NavSatFix, resolution: float, nodata: float
) -> tuple[npt.NDArray[np.float64], float, float]:
    map_dataset = gdal.Warp(
        "",
        dataset,
        format="MEM",
        dstSRS=_map_crs(origin).ExportToWkt(),
        xRes=resolution,
        yRes=resolution,
        resampleAlg="max",
        dstNodata=nodata,
    )

    # Shift USGS 3DEP data from NAD83(2011) onto ITRF2020 (~WGS 84) at the current epoch
    epoch = 1970.0 + time.time() / _SECONDS_PER_YEAR
    itrf2020_crs = _geographic_crs(_EPSG_ITRF2020_3D)
    itrf2020_crs.SetCoordinateEpoch(epoch)
    itrf2020_T_nad83 = osr.CoordinateTransformation(
        _geographic_crs(_EPSG_NAD83_2011_3D), itrf2020_crs
    )
    lon, lat, _, _ = itrf2020_T_nad83.TransformPoint(
        origin.longitude, origin.latitude, origin.altitude, epoch
    )
    map_T_wgs84 = osr.CoordinateTransformation(_geographic_crs(_EPSG_WGS84), _map_crs(origin))
    east_offset, north_offset, _ = map_T_wgs84.TransformPoint(lon, lat)

    map_values = np.flipud(map_dataset.GetRasterBand(1).ReadAsArray())
    height = map_values.shape[0]
    geo_transform = map_dataset.GetGeoTransform()
    min_east = geo_transform[0] + east_offset
    min_north = geo_transform[3] + height * geo_transform[5] + north_offset
    return map_values, min_east, min_north


def flat_grid(size: float, resolution: float) -> tuple[npt.NDArray[np.int8], float, float]:
    width = max(1, math.ceil(size / resolution))
    min_corner = -0.5 * width * resolution
    return np.full((width, width), FREE, dtype=np.int8), min_corner, min_corner


def occupancy_grid(
    grid: npt.NDArray[np.int8],
    min_east: float,
    min_north: float,
    resolution: float,
    frame_id: str,
    stamp: Time,
) -> OccupancyGrid:
    height, width = grid.shape

    occupancy_grid_msg = OccupancyGrid()
    occupancy_grid_msg.header.stamp = stamp
    occupancy_grid_msg.header.frame_id = frame_id
    occupancy_grid_msg.info.resolution = resolution
    occupancy_grid_msg.info.width = width
    occupancy_grid_msg.info.height = height
    occupancy_grid_msg.info.origin.position.x = min_east
    occupancy_grid_msg.info.origin.position.y = min_north
    occupancy_grid_msg.info.origin.orientation.w = 1.0
    occupancy_grid_msg.data = array.array("b", grid.tobytes())
    return occupancy_grid_msg
