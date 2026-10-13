import pytest

from s2cloudless_stac.stac import l1c_search_filter, select_l1c

L2A = "S2C_MSIL2A_20260517T002711_N0512_R016_T55MGS_20260517T031408"
L1C = "S2C_MSIL1C_20260517T002711_N0512_R016_T55MGS_20260517T020317"


def test_select_single_match(item_factory):
    l1c = item_factory(L1C)
    assert select_l1c(item_factory(L2A), [l1c]) is l1c


def test_select_ignores_other_scenes(item_factory):
    other_tile = item_factory(L1C.replace("T55MGS", "T55MGR"))
    other_orbit = item_factory(L1C.replace("R016", "R116"))
    l1c = item_factory(L1C)
    assert select_l1c(item_factory(L2A), [other_tile, other_orbit, l1c]) is l1c


def test_select_without_candidates_raises(item_factory):
    with pytest.raises(LookupError, match="no L1C"):
        select_l1c(item_factory(L2A), [])


def test_select_baseline_mismatch_raises(item_factory):
    old = item_factory(L1C.replace("N0512", "N0511"), baseline="05.11")
    with pytest.raises(LookupError, match="05.11"):
        select_l1c(item_factory(L2A), [old])


def test_select_baseline_mismatch_allowed(item_factory):
    old = item_factory(L1C.replace("N0512", "N0511"), baseline="05.11")
    assert select_l1c(item_factory(L2A), [old], allow_baseline_mismatch=True) is old


def test_select_prefers_same_baseline_when_mismatch_allowed(item_factory):
    old = item_factory(L1C.replace("N0512", "N0511"), baseline="05.11")
    same = item_factory(L1C)
    assert (
        select_l1c(item_factory(L2A), [old, same], allow_baseline_mismatch=True) is same
    )


def test_select_latest_l1c_processed_before_l2a(item_factory):
    early = item_factory(L1C.replace("T020317", "T010000"))
    latest = item_factory(L1C)
    after = item_factory(L1C.replace("_20260517T020317", "_20260518T000000"))
    assert select_l1c(item_factory(L2A), [early, after, latest]) is latest


def test_select_only_l1c_processed_after_l2a_raises(item_factory):
    after = item_factory(L1C.replace("_20260517T020317", "_20260518T000000"))
    with pytest.raises(LookupError, match="processed before"):
        select_l1c(item_factory(L2A), [after])


def test_search_filter_on_tile_and_platform(item_factory):
    assert l1c_search_filter(item_factory(L2A)) == {
        "op": "and",
        "args": [
            {"op": "=", "args": [{"property": "grid:code"}, "MGRS-55MGS"]},
            {"op": "=", "args": [{"property": "platform"}, "sentinel-2c"]},
        ],
    }
