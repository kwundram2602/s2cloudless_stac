import numpy as np
import pytest

from s2cloudless_stac.cloud import cloud_layers

# B01 B02 B04 B05 B08 B8A B09 B10 B11 B12
CLOUD = [0.45, 0.45, 0.45, 0.45, 0.5, 0.5, 0.2, 0.05, 0.35, 0.25]
WATER = [0.08, 0.06, 0.03, 0.02, 0.01, 0.01, 0.003, 0.001, 0.005, 0.003]


@pytest.fixture
def scene() -> np.ndarray:
    """(band, y, x) reflectance: left half cloud, right half water."""
    refl = np.empty((10, 20, 20), dtype=np.float32)
    refl[:, :, :10] = np.array(CLOUD, dtype=np.float32)[:, None, None]
    refl[:, :, 10:] = np.array(WATER, dtype=np.float32)[:, None, None]
    return refl


def test_shapes_and_dtypes(scene):
    clp, clm, valid = cloud_layers(scene)
    assert clp.shape == clm.shape == valid.shape == (20, 20)
    assert clp.dtype == np.float32
    assert clm.dtype == np.uint8
    assert valid.dtype == bool


def test_cloud_more_probable_than_water(scene):
    clp, clm, _ = cloud_layers(scene, average_over=None, dilation_size=None)
    assert clp[:, :10].min() > clp[:, 10:].max()
    assert clm[:, :10].all()
    assert not clm[:, 10:].any()


def test_nan_in_any_band_is_invalid(scene):
    scene[7, 3, 15] = np.nan
    clp, clm, valid = cloud_layers(scene)
    assert not valid[3, 15]
    assert valid.sum() == 20 * 20 - 1
    assert np.isfinite(clp).all()
    assert set(np.unique(clm)) <= {0, 1}


def test_rejects_wrong_band_count():
    with pytest.raises(ValueError, match="10 bands"):
        cloud_layers(np.zeros((13, 4, 4), dtype=np.float32))
