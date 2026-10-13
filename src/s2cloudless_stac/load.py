import math
import os
from typing import Any

import numpy as np
import odc.stac
import pystac
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox

from s2cloudless_stac.radiometry import BAND_ORDER, NODATA_DN

# B01 is native 60 m, its grid is the 60 m tile grid (also of the L2A *_60m)
GRID_ASSET = "B01"
# GDAL settings for CDSE S3. Keys (https://eodata-s3keysmanager.dataspace.copernicus.eu/)
# come from the environment: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.
# JP2 tiles are decoded single-threaded: the multi-threaded JP2OpenJPEG decoder
# on CDSE S3 fails on some tiles ("Stream too short") and, without raising,
# returns zero pixels that then pass as nodata.
CDSE_GDAL = {
    "AWS_S3_ENDPOINT": "eodata.dataspace.copernicus.eu",
    "AWS_VIRTUAL_HOSTING": "FALSE",
    "AWS_HTTPS": "YES",
    "GDAL_NUM_THREADS": "1",
}


def configure_gdal(opts: dict[str, str] = CDSE_GDAL) -> None:
    rio_opts: dict[str, Any] = dict(opts)
    odc.stac.configure_rio(cloud_defaults=True, **rio_opts)
    # the rasterio env is thread-local and threads opening files without it
    # would read from AWS; GDAL falls back to process env vars
    os.environ.update(opts)


def tile_geobox(item: pystac.Item) -> GeoBox:
    """The 60 m tile grid from the projection fields of the `B01` asset."""
    f = item.assets[GRID_ASSET].extra_fields
    crs = f.get("proj:code") or f"EPSG:{f['proj:epsg']}"
    height, width = f["proj:shape"]
    return GeoBox((height, width), Affine(*f["proj:transform"][:6]), crs)


def crop_geobox(tile: GeoBox, target: GeoBox, buffer_px: int) -> GeoBox:
    """Part of the `tile` grid covering `target` plus `buffer_px` pixels, clipped.

    The buffer keeps the s2cloudless averaging/dilation at the target edges
    unaffected by the crop.
    """
    if tile.crs is None:
        raise ValueError("tile grid has no CRS")
    left, bottom, right, top = target.extent.to_crs(tile.crs).boundingbox
    inv = ~tile.affine
    cols, rows = zip(*(inv @ (x, y) for x, y in ((left, top), (right, bottom))))
    # tolerance so target edges on tile pixel edges do not grow the crop by one
    x0 = max(math.floor(min(cols) + 1e-6) - buffer_px, 0)
    x1 = min(math.ceil(max(cols) - 1e-6) + buffer_px, tile.width)
    y0 = max(math.floor(min(rows) + 1e-6) - buffer_px, 0)
    y1 = min(math.ceil(max(rows) - 1e-6) + buffer_px, tile.height)
    if x0 >= x1 or y0 >= y1:
        raise ValueError(
            f"target {target.extent.boundingbox} does not overlap the tile"
        )
    return tile[y0:y1, x0:x1]


def load_bands(
    item: pystac.Item, geobox: GeoBox | None = None
) -> tuple[xr.Dataset, GeoBox]:
    """The s2cloudless bands of the L1C `item` as raw DN on the 60 m tile grid.

    `geobox` restricts the read to part of that grid (see `crop_geobox`),
    default the whole tile. 10 and 20 m bands are area-averaged, nodata
    (DN 0) is left out of the mean.
    """
    gbox = geobox if geobox is not None else tile_geobox(item)
    ds = odc.stac.load(
        [item],
        bands=list(BAND_ORDER),
        geobox=gbox,
        resampling="average",
        nodata=NODATA_DN,
    ).isel(time=0)
    check_loaded(ds, item.id)
    return ds, gbox


def check_loaded(ds: xr.Dataset, item_id: str) -> None:
    """Raise if a band is all zero (nodata).

    odc-stac fills missing or unreadable assets with 0 without raising.
    """
    empty = [b for b in BAND_ORDER if not np.any(ds[b].to_numpy() != NODATA_DN)]
    if empty:
        raise ValueError(f"{item_id}: bands {empty} loaded as all nodata")
