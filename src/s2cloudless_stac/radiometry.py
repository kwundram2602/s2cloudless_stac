import logging

import numpy as np
import pystac

log = logging.getLogger(__name__)

# s2cloudless input bands and their order with `all_bands=False`
BAND_ORDER = ("B01", "B02", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12")
QUANTIFICATION = 10_000
NODATA_DN = 0


def dn_offset(baseline: str) -> int:
    """L1C RADIO_ADD_OFFSET: -1000 DN from processing baseline 04.00 on, else 0."""
    major, minor = (int(p) for p in baseline.split("."))
    return -1000 if (major, minor) >= (4, 0) else 0


def check_offset(item: pystac.Item, offset: int) -> None:
    """Raise if a band's STAC `raster:offset` contradicts the baseline offset.

    CDSE publishes the offset in reflectance units (-0.1 for -1000 DN).
    """
    expected = offset / QUANTIFICATION
    for band in BAND_ORDER:
        stac_offset = item.assets[band].extra_fields.get("raster:offset")
        if stac_offset is None:
            log.info("%s %s has no raster:offset, using baseline offset", item.id, band)
        elif not np.isclose(stac_offset, expected):
            raise ValueError(
                f"{item.id} {band}: raster:offset {stac_offset} contradicts the "
                f"baseline offset {offset} DN ({expected})"
            )


def to_reflectance(dn: np.ndarray, offset: int) -> np.ndarray:
    """TOA reflectance (DN + offset) / 10000 as float32; nodata DN 0 -> NaN.

    Dark pixels below the offset come out negative and are clipped to 0.
    """
    refl = np.clip((dn.astype(np.float32) + offset) / QUANTIFICATION, 0, None)
    return np.where(dn == NODATA_DN, np.float32(np.nan), refl).astype(np.float32)
