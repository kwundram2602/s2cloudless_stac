from collections.abc import Mapping
from pathlib import Path

import numpy as np
import rasterio
from odc.geo.geobox import GeoBox

CLM_NODATA = 255


def write_layers(
    out_dir: Path,
    l2a_id: str,
    clp: np.ndarray,
    clm: np.ndarray,
    valid: np.ndarray,
    geobox: GeoBox,
    tags: Mapping[str, object],
) -> tuple[Path, Path]:
    """`<l2a_id>_CLP.tif` and `<l2a_id>_CLM.tif` in the Sentinel Hub convention.

    CLP is the probability * 255 as uint8; 255 is a valid value, so nodata is
    an internal mask. CLM is 0/1 with nodata 255. `tags` are written to both.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": geobox.height,
        "width": geobox.width,
        "count": 1,
        "dtype": "uint8",
        "crs": str(geobox.crs),
        "transform": geobox.affine,
        "compress": "deflate",
        "tiled": True,
    }
    all_tags = {"L2A_ID": l2a_id, **{k: str(v) for k, v in tags.items()}}

    clp_path = out_dir / f"{l2a_id}_CLP.tif"
    with rasterio.open(clp_path, "w", **profile) as dst:
        dst.write(np.round(clp * 255).astype(np.uint8), 1)
        dst.write_mask(valid)
        dst.update_tags(**all_tags)

    clm_path = out_dir / f"{l2a_id}_CLM.tif"
    with rasterio.open(clm_path, "w", nodata=CLM_NODATA, **profile) as dst:
        dst.write(np.where(valid, clm, CLM_NODATA).astype(np.uint8), 1)
        dst.update_tags(**all_tags)
    return clp_path, clm_path
