from datetime import UTC, datetime

import pystac
import pytest

from s2cloudless_stac.radiometry import BAND_ORDER


def make_item(
    item_id: str,
    *,
    baseline: str = "05.12",
    raster_offset: float | None = -0.1,
) -> pystac.Item:
    """Minimal CDSE-like S2 STAC item with the s2cloudless band assets."""
    item = pystac.Item(
        id=item_id,
        geometry=None,
        bbox=None,
        datetime=datetime(2026, 5, 17, 0, 27, 11, 25000, tzinfo=UTC),
        properties={
            "platform": "sentinel-2c",
            "grid:code": "MGRS-55MGS",
            "processing:version": baseline,
        },
    )
    for band in BAND_ORDER:
        extra = {} if raster_offset is None else {"raster:offset": raster_offset}
        item.add_asset(
            band, pystac.Asset(href=f"s3://eodata/{band}.jp2", extra_fields=extra)
        )
    return item


@pytest.fixture
def item_factory():
    return make_item
