import numpy as np
from s2cloudless import S2PixelCloudDetector

from s2cloudless_stac.radiometry import BAND_ORDER

# s2cloudless README setup; the library defaults are 1 and 1
THRESHOLD = 0.4
AVERAGE_OVER = 4
DILATION_SIZE = 2


def cloud_layers(
    refl: np.ndarray,
    threshold: float = THRESHOLD,
    average_over: int | None = AVERAGE_OVER,
    dilation_size: int | None = DILATION_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cloud probability (float32 0..1), cloud mask (uint8 0/1) and valid pixels.

    `refl` is (band, y, x) TOA reflectance in `BAND_ORDER`, NaN for nodata.
    Pixels with NaN in any band go into the model as 0 and are invalid.
    """
    if refl.ndim != 3 or refl.shape[0] != len(BAND_ORDER):
        raise ValueError(
            f"expected (band, y, x) with {len(BAND_ORDER)} bands, got {refl.shape}"
        )
    valid = np.isfinite(refl).all(axis=0)
    data = np.nan_to_num(np.moveaxis(refl, 0, -1), nan=0.0)[np.newaxis]
    detector = S2PixelCloudDetector(
        threshold=threshold,
        all_bands=False,
        average_over=average_over,
        dilation_size=dilation_size,
    )
    clp = detector.get_cloud_probability_maps(data)
    # mask from the same probabilities, the model runs once
    clm = detector.get_mask_from_prob(clp).astype(np.uint8)
    return clp[0], clm[0], valid
