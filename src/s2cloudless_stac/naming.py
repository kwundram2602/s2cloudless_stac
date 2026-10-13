import re
from dataclasses import dataclass
from datetime import UTC, datetime

# S2C_MSIL2A_20260517T002711_N0512_R016_T55MGS_20260517T031408[.SAFE]
_PATTERN = re.compile(
    r"^(?P<platform>S2[A-D])_MSIL(?P<level>1C|2A)_(?P<sensing>\d{8}T\d{6})"
    r"_N(?P<baseline>\d{4})_R(?P<orbit>\d{3})_T(?P<tile>\d{2}[A-Z]{3})"
    r"_(?P<discriminator>\d{8}T\d{6})(\.SAFE)?$"
)
_TIME = "%Y%m%dT%H%M%S"


@dataclass(frozen=True)
class S2ProductId:
    """Sentinel-2 product name: platform, sensing time, baseline, orbit, tile.

    The discriminator is the processing time, it differs between an L2A and
    the L1C it was derived from.
    """

    platform: str
    level: str
    sensing: datetime
    baseline: str
    rel_orbit: int
    tile: str
    discriminator: datetime

    @classmethod
    def parse(cls, product_id: str) -> "S2ProductId":
        m = _PATTERN.match(product_id)
        if m is None:
            raise ValueError(f"not a Sentinel-2 product id: {product_id!r}")
        b = m["baseline"]
        return cls(
            platform=m["platform"],
            level=m["level"],
            sensing=_utc(m["sensing"]),
            baseline=f"{b[:2]}.{b[2:]}",
            rel_orbit=int(m["orbit"]),
            tile=m["tile"],
            discriminator=_utc(m["discriminator"]),
        )

    def scene_key(self) -> tuple[str, datetime, int, str]:
        """Identifies the acquisition, equal for the L1C and L2A of one scene."""
        return (self.platform, self.sensing, self.rel_orbit, self.tile)


def _utc(stamp: str) -> datetime:
    """Product name timestamps are UTC."""
    return datetime.strptime(stamp, _TIME).replace(tzinfo=UTC)
