import numpy as np
from affine import Affine
from odc.geo.geobox import GeoBox

from s2cloudless_stac.masks import CloudMasks, to_grid

SRC = GeoBox((2, 2), Affine(60, 0, 699960, 0, -60, 9700000), "EPSG:32755")


def _masks() -> CloudMasks:
    return CloudMasks(
        clp=np.array([[0.1, 0.9], [0.5, 0.2]], dtype=np.float32),
        clm=np.array([[0, 1], [1, 0]], dtype=np.uint8),
        valid=np.array([[True, True], [True, False]]),
        geobox=SRC,
        l1c_id="L1C",
        baseline="05.12",
        dn_offset=-1000,
    )


def test_to_grid_nearest_upsampling_with_nan_nodata():
    target = GeoBox((12, 12), Affine(10, 0, 699960, 0, -10, 9700000), "EPSG:32755")
    clp, clm = to_grid(_masks(), target)
    assert clp.shape == clm.shape == (12, 12)
    assert clp.dtype == clm.dtype == np.float32
    # each 60 m pixel becomes a 6 x 6 block
    np.testing.assert_allclose(clp[:6, 6:], 0.9)
    np.testing.assert_array_equal(clm[6:, :6], 1.0)
    assert np.isnan(clp[6:, 6:]).all()
    assert np.isnan(clm[6:, 6:]).all()


def test_to_grid_outside_source_is_nan():
    target = GeoBox((2, 2), Affine(60, 0, 699960 + 600, 0, -60, 9700000), "EPSG:32755")
    clp, clm = to_grid(_masks(), target)
    assert np.isnan(clp).all()
    assert np.isnan(clm).all()
