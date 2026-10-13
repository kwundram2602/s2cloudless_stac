from dataclasses import dataclass

import numpy as np
import pystac
import pystac_client
from odc.geo.geobox import GeoBox
from odc.geo.xr import wrap_xr

from s2cloudless_stac import cloud, stac
from s2cloudless_stac.load import crop_geobox, load_bands, tile_geobox
from s2cloudless_stac.naming import S2ProductId
from s2cloudless_stac.radiometry import (
    BAND_ORDER,
    check_offset,
    dn_offset,
    to_reflectance,
)


@dataclass(frozen=True)
class CloudMasks:
    """s2cloudless layers of one scene on a 60 m grid, with their L1C source."""

    clp: np.ndarray  # float32 probability 0..1
    clm: np.ndarray  # uint8 0/1
    valid: np.ndarray  # bool, False where any input band is nodata
    geobox: GeoBox
    l1c_id: str
    baseline: str
    dn_offset: int


def compute_masks(
    client: pystac_client.Client,
    l2a: pystac.Item,
    target: GeoBox | None = None,
    threshold: float = cloud.THRESHOLD,
    average_over: int | None = cloud.AVERAGE_OVER,
    dilation_size: int | None = cloud.DILATION_SIZE,
    allow_baseline_mismatch: bool = False,
) -> CloudMasks:
    """CLP/CLM for the L2A item `l2a` from its L1C counterpart.

    With `target`, only the part of the 60 m tile around it is read and
    computed, buffered by the averaging and dilation radii.
    """
    l1c = stac.find_l1c(client, l2a, allow_baseline_mismatch)
    baseline = S2ProductId.parse(l1c.id).baseline
    offset = dn_offset(baseline)
    check_offset(l1c, offset)

    gbox = None
    if target is not None:
        buffer_px = (average_over or 0) + (dilation_size or 0)
        gbox = crop_geobox(tile_geobox(l1c), target, buffer_px)
    ds, gbox = load_bands(l1c, gbox)
    refl = np.stack([to_reflectance(ds[b].to_numpy(), offset) for b in BAND_ORDER])
    clp, clm, valid = cloud.cloud_layers(refl, threshold, average_over, dilation_size)
    return CloudMasks(clp, clm, valid, gbox, l1c.id, baseline, offset)


def to_grid(masks: CloudMasks, target: GeoBox) -> tuple[np.ndarray, np.ndarray]:
    """CLP and CLM nearest-resampled onto `target`, float32 with NaN as nodata."""
    nan = np.float32(np.nan)
    layers = (
        np.where(masks.valid, masks.clp, nan).astype(np.float32),
        np.where(masks.valid, masks.clm, nan).astype(np.float32),
    )
    clp, clm = (
        wrap_xr(layer, masks.geobox, nodata=float("nan"))
        .odc.reproject(target, resampling="nearest", dst_nodata=nan)
        .to_numpy()
        for layer in layers
    )
    return clp, clm
