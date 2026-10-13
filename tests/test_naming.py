from datetime import UTC, datetime

import pytest

from s2cloudless_stac.naming import S2ProductId

L2A = "S2C_MSIL2A_20260517T002711_N0512_R016_T55MGS_20260517T031408"
L1C = "S2C_MSIL1C_20260517T002711_N0512_R016_T55MGS_20260517T020317"


def test_parse_l2a_fields():
    p = S2ProductId.parse(L2A)
    assert p.platform == "S2C"
    assert p.level == "2A"
    assert p.sensing == datetime(2026, 5, 17, 0, 27, 11, tzinfo=UTC)
    assert p.baseline == "05.12"
    assert p.rel_orbit == 16
    assert p.tile == "55MGS"
    assert p.discriminator == datetime(2026, 5, 17, 3, 14, 8, tzinfo=UTC)


def test_parse_accepts_safe_suffix():
    assert S2ProductId.parse(L1C + ".SAFE").level == "1C"


def test_l1c_and_l2a_of_one_scene_share_scene_key():
    assert S2ProductId.parse(L1C).scene_key() == S2ProductId.parse(L2A).scene_key()


def test_scene_key_differs_by_tile():
    other = L2A.replace("T55MGS", "T55MGR")
    assert S2ProductId.parse(other).scene_key() != S2ProductId.parse(L2A).scene_key()


@pytest.mark.parametrize("bad", ["", "LC08_L2SP_001", L2A.replace("_N0512", "_X0512")])
def test_parse_rejects_invalid_ids(bad):
    with pytest.raises(ValueError, match="Sentinel-2 product id"):
        S2ProductId.parse(bad)
