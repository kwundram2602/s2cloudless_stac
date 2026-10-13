import argparse
import logging
from collections.abc import Callable, Iterable
from pathlib import Path

import pystac_client
from dotenv import find_dotenv, load_dotenv

from s2cloudless_stac import cloud, stac
from s2cloudless_stac.io import write_layers
from s2cloudless_stac.load import configure_gdal
from s2cloudless_stac.masks import compute_masks

log = logging.getLogger("s2cl")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="s2cl",
        description="s2cloudless CLP/CLM (60 m tile grid) for Sentinel-2 L2A "
        "scenes, computed from their L1C counterpart on CDSE STAC.",
    )
    p.add_argument("ids", nargs="+", metavar="L2A_ID", help="L2A STAC item ids")
    p.add_argument("--out-dir", type=Path, default=Path("data/out"))
    p.add_argument("--threshold", type=float, default=cloud.THRESHOLD)
    p.add_argument("--average-over", type=int, default=cloud.AVERAGE_OVER)
    p.add_argument("--dilation-size", type=int, default=cloud.DILATION_SIZE)
    p.add_argument(
        "--allow-baseline-mismatch",
        action="store_true",
        help="use an L1C of another processing baseline if none matches",
    )
    p.add_argument("--stac-url", default=stac.CDSE_STAC)
    return p.parse_args(argv)


def process(
    client: pystac_client.Client, l2a_id: str, args: argparse.Namespace
) -> tuple[Path, Path]:
    """Find the L1C of `l2a_id`, run s2cloudless on its tile, write CLP and CLM."""
    l2a = stac.get_l2a_item(client, l2a_id)
    m = compute_masks(
        client,
        l2a,
        threshold=args.threshold,
        average_over=args.average_over,
        dilation_size=args.dilation_size,
        allow_baseline_mismatch=args.allow_baseline_mismatch,
    )
    tags = {
        "L1C_ID": m.l1c_id,
        "PROCESSING_BASELINE": m.baseline,
        "DN_OFFSET": m.dn_offset,
        "S2CLOUDLESS_THRESHOLD": args.threshold,
        "S2CLOUDLESS_AVERAGE_OVER": args.average_over,
        "S2CLOUDLESS_DILATION_SIZE": args.dilation_size,
    }
    return write_layers(args.out_dir, l2a.id, m.clp, m.clm, m.valid, m.geobox, tags)


def run_all(ids: Iterable[str], process_one: Callable[[str], object]) -> int:
    """Process each id, logging failures instead of stopping; 1 if any failed."""
    failed = []
    for item_id in ids:
        try:
            process_one(item_id)
        except Exception:
            log.exception("%s failed", item_id)
            failed.append(item_id)
    if failed:
        log.error("%d failed: %s", len(failed), ", ".join(failed))
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # CDSE S3 keys (AWS_*) from the nearest .env upward of the working directory
    load_dotenv(find_dotenv(usecwd=True))
    args = parse_args(argv)
    configure_gdal()
    client = stac.open_client(args.stac_url)

    def process_one(item_id: str) -> None:
        clp, clm = process(client, item_id, args)
        log.info("wrote %s, %s", clp, clm)

    return run_all(args.ids, process_one)
