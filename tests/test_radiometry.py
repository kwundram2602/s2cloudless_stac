import numpy as np
import pytest

from s2cloudless_stac.radiometry import check_offset, dn_offset, to_reflectance

L1C = "S2C_MSIL1C_20260517T002711_N0512_R016_T55MGS_20260517T020317"


@pytest.mark.parametrize(
    ("baseline", "offset"),
    [("02.14", 0), ("03.01", 0), ("04.00", -1000), ("05.12", -1000), ("10.00", -1000)],
)
def test_dn_offset_from_baseline_04_00(baseline, offset):
    assert dn_offset(baseline) == offset


def test_to_reflectance_applies_offset_and_scale():
    dn = np.array([1000, 2000, 11000], dtype=np.uint16)
    np.testing.assert_allclose(to_reflectance(dn, -1000), [0.0, 0.1, 1.0])


def test_to_reflectance_without_offset_for_old_baselines():
    np.testing.assert_allclose(
        to_reflectance(np.array([1000], dtype=np.uint16), 0), [0.1]
    )


def test_to_reflectance_nodata_is_nan_and_negatives_clip_to_zero():
    out = to_reflectance(np.array([0, 500], dtype=np.uint16), -1000)
    assert out.dtype == np.float32
    assert np.isnan(out[0])
    assert out[1] == 0.0


def test_check_offset_accepts_matching_stac_offset(item_factory):
    check_offset(item_factory(L1C, raster_offset=-0.1), -1000)


def test_check_offset_rejects_contradicting_stac_offset(item_factory):
    with pytest.raises(ValueError, match="raster:offset"):
        check_offset(item_factory(L1C, raster_offset=0.0), -1000)


def test_check_offset_passes_without_stac_offset(item_factory):
    check_offset(item_factory(L1C, raster_offset=None), -1000)
