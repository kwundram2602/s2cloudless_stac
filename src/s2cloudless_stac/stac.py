import logging
from collections.abc import Iterable
from datetime import timedelta

import pystac
import pystac_client
from pystac_client.stac_api_io import StacApiIO
from urllib3 import Retry

from s2cloudless_stac.naming import S2ProductId

log = logging.getLogger(__name__)

CDSE_STAC = "https://stac.dataspace.copernicus.eu/v1"
L2A_COLLECTION = "sentinel-2-l2a"
L1C_COLLECTION = "sentinel-2-l1c"
# retry rate limits (CDSE answers 429) and transient server errors with
# exponential backoff, honouring Retry-After; POST searches are retried too
RETRY = Retry(
    total=8,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=None,
    respect_retry_after_header=True,
)
# search window around the L2A sensing time, both levels share `datetime`
TIME_TOLERANCE = timedelta(seconds=1)


def open_client(url: str = CDSE_STAC) -> pystac_client.Client:
    return pystac_client.Client.open(url, stac_io=StacApiIO(max_retries=RETRY))


def get_l2a_item(client: pystac_client.Client, item_id: str) -> pystac.Item:
    items = list(client.search(collections=[L2A_COLLECTION], ids=[item_id]).items())
    if not items:
        raise LookupError(f"L2A item {item_id} not found in {L2A_COLLECTION}")
    return items[0]


def l1c_search_filter(l2a: pystac.Item) -> dict:
    """CQL2-JSON filter on the tile and platform of `l2a`."""
    return {
        "op": "and",
        "args": [
            {"op": "=", "args": [{"property": p}, l2a.properties[p]]}
            for p in ("grid:code", "platform")
        ],
    }


def find_l1c(
    client: pystac_client.Client,
    l2a: pystac.Item,
    allow_baseline_mismatch: bool = False,
) -> pystac.Item:
    """The L1C item `l2a` was processed from, see `select_l1c`."""
    if l2a.datetime is None:
        raise ValueError(f"{l2a.id} has no datetime")
    start = l2a.datetime - TIME_TOLERANCE
    end = l2a.datetime + TIME_TOLERANCE
    search = client.search(
        collections=[L1C_COLLECTION],
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        filter=l1c_search_filter(l2a),
        filter_lang="cql2-json",
    )
    l1c = select_l1c(l2a, search.items(), allow_baseline_mismatch)
    log.info(
        "%s (baseline %s) -> %s (baseline %s)",
        l2a.id,
        S2ProductId.parse(l2a.id).baseline,
        l1c.id,
        S2ProductId.parse(l1c.id).baseline,
    )
    return l1c


def select_l1c(
    l2a: pystac.Item,
    candidates: Iterable[pystac.Item],
    allow_baseline_mismatch: bool = False,
) -> pystac.Item:
    """L1C of the same acquisition (platform, sensing time, orbit, tile) as `l2a`.

    The baseline must match unless `allow_baseline_mismatch`, then matching
    baselines are still preferred. Of several (reprocessed) L1C, the one
    processed last before `l2a` is taken, as the L2A is derived from it.
    """
    target = S2ProductId.parse(l2a.id)
    scene = [
        (pid, c)
        for c in candidates
        if (pid := S2ProductId.parse(c.id)).level == "1C"
        and pid.scene_key() == target.scene_key()
    ]
    if not scene:
        raise LookupError(f"no L1C found for {l2a.id}")

    same_baseline = [(pid, c) for pid, c in scene if pid.baseline == target.baseline]
    if not same_baseline:
        found = sorted({pid.baseline for pid, _ in scene})
        if not allow_baseline_mismatch:
            raise LookupError(
                f"no L1C with baseline {target.baseline} for {l2a.id}, found {found}"
            )
        log.warning(
            "%s: no L1C with baseline %s, using baselines %s",
            l2a.id,
            target.baseline,
            found,
        )
        same_baseline = scene

    before = [
        (pid, c)
        for pid, c in same_baseline
        if pid.discriminator <= target.discriminator
    ]
    if not before:
        raise LookupError(f"no L1C for {l2a.id} processed before the L2A")
    return max(before, key=lambda pc: pc[0].discriminator)[1]
