import numpy as np
import pytest
import rasterio
from affine import Affine
from odc.geo.geobox import GeoBox

from s2cloudless_stac.io import write_layers

L2A = "S2C_MSIL2A_20260517T002711_N0512_R016_T55MGS_20260517T031408"
GEOBOX = GeoBox((4, 4), Affine(60, 0, 699960, 0, -60, 9700000), "EPSG:32755")


@pytest.fixture
def layers():
    clp = np.array([[0.0, 0.5, 1.0, 0.2]] * 4, dtype=np.float32)
    clm = np.array([[0, 1, 1, 0]] * 4, dtype=np.uint8)
    valid = np.ones((4, 4), dtype=bool)
    valid[0, 0] = False
    return clp, clm, valid


@pytest.fixture
def written(tmp_path, layers):
    clp, clm, valid = layers
    return write_layers(
        tmp_path, L2A, clp, clm, valid, GEOBOX, {"L1C_ID": "x", "DN_OFFSET": -1000}
    )


def test_file_names(tmp_path, written):
    assert written == (tmp_path / f"{L2A}_CLP.tif", tmp_path / f"{L2A}_CLM.tif")


def test_clp_scaled_to_uint8_with_internal_mask(written):
    with rasterio.open(written[0]) as src:
        assert src.dtypes == ("uint8",)
        assert src.nodata is None
        np.testing.assert_array_equal(src.read(1)[1], [0, 128, 255, 51])
        mask = src.dataset_mask()
        assert mask[0, 0] == 0
        assert (mask != 0).sum() == 15


def test_clm_nodata_255(written):
    with rasterio.open(written[1]) as src:
        assert src.dtypes == ("uint8",)
        assert src.nodata == 255
        data = src.read(1)
        assert data[0, 0] == 255
        np.testing.assert_array_equal(data[1], [0, 1, 1, 0])


def test_grid_and_tags(written):
    for path in written:
        with rasterio.open(path) as src:
            assert src.crs.to_epsg() == 32755
            assert src.transform == GEOBOX.affine
            tags = src.tags()
            assert tags["L2A_ID"] == L2A
            assert tags["L1C_ID"] == "x"
            assert tags["DN_OFFSET"] == "-1000"
