import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox

from s2cloudless_stac.load import check_loaded, crop_geobox, tile_geobox
from s2cloudless_stac.radiometry import BAND_ORDER

L1C = "S2C_MSIL1C_20260517T002711_N0512_R016_T55MGS_20260517T020317"


def test_tile_geobox_from_60m_asset(item_factory):
    item = item_factory(L1C)
    item.assets["B01"].extra_fields |= {
        "proj:code": "EPSG:32755",
        "proj:transform": [60, 0, 699960, 0, -60, 9700000],
        "proj:shape": [1830, 1830],
    }
    gbox = tile_geobox(item)
    assert gbox.shape == (1830, 1830)
    assert gbox.affine == Affine(60, 0, 699960, 0, -60, 9700000)
    assert gbox.crs == "EPSG:32755"


def test_tile_geobox_accepts_proj_epsg(item_factory):
    item = item_factory(L1C)
    item.assets["B01"].extra_fields |= {
        "proj:epsg": 32755,
        "proj:transform": [60, 0, 699960, 0, -60, 9700000],
        "proj:shape": [1830, 1830],
    }
    assert tile_geobox(item).crs == "EPSG:32755"


def _dataset(zero_band: str | None = None) -> xr.Dataset:
    ds = xr.Dataset(
        {b: (("y", "x"), np.full((2, 2), 1500, dtype=np.uint16)) for b in BAND_ORDER}
    )
    if zero_band:
        ds[zero_band][:] = 0
    return ds


def test_check_loaded_passes_with_data():
    check_loaded(_dataset(), "id")


def test_check_loaded_raises_on_all_zero_band():
    with pytest.raises(ValueError, match="B10"):
        check_loaded(_dataset("B10"), "id")


TILE = GeoBox((1830, 1830), Affine(60, 0, 699960, 0, -60, 9700000), "EPSG:32755")


def test_crop_geobox_covers_target_on_tile_grid_with_buffer():
    # 20 km window at 10 m inside the tile, same CRS
    target = GeoBox((2000, 2000), Affine(10, 0, 720000, 0, -10, 9660000), "EPSG:32755")
    crop = crop_geobox(TILE, target, buffer_px=6)
    # x 720000..740000 -> tile cols 334..668, y rows 666..1000, +-6 px buffer
    assert crop.affine == Affine(60, 0, 699960 + 328 * 60, 0, -60, 9700000 - 660 * 60)
    assert crop.shape == (346, 346)
    assert crop.crs == "EPSG:32755"


def test_crop_geobox_clips_to_tile():
    target = GeoBox((100, 100), Affine(60, 0, 699960, 0, -60, 9700000), "EPSG:32755")
    crop = crop_geobox(TILE, target, buffer_px=6)
    assert crop.affine == TILE.affine
    assert crop.shape == (106, 106)


def test_crop_geobox_rejects_target_outside_tile():
    target = GeoBox((10, 10), Affine(60, 0, 100000, 0, -60, 9700000), "EPSG:32755")
    with pytest.raises(ValueError, match="does not overlap"):
        crop_geobox(TILE, target, buffer_px=6)
