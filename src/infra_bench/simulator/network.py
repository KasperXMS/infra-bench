from ..schemas import InfraState, NetworkLink


def transfer_time_s(size_mb: float, rtt_ms: float, bandwidth_mbps: float) -> float:
    if size_mb < 0:
        raise ValueError("size_mb must be non-negative")
    if rtt_ms < 0:
        raise ValueError("rtt_ms must be non-negative")
    if bandwidth_mbps <= 0:
        raise ValueError("bandwidth_mbps must be positive")
    return rtt_ms / 1000.0 + 8.0 * size_mb / bandwidth_mbps


def find_link(infra: InfraState, src_site: str, dst_site: str) -> NetworkLink | None:
    if src_site == dst_site:
        return None
    exact = next(
        (
            link
            for link in infra.links
            if link.src_site == src_site and link.dst_site == dst_site
        ),
        None,
    )
    if exact is not None:
        return exact
    return next(
        (
            link
            for link in infra.links
            if link.src_site == dst_site and link.dst_site == src_site
        ),
        None,
    )


def transfer_between_sites_s(
    infra: InfraState, src_site: str, dst_site: str, size_mb: float
) -> float:
    if src_site == dst_site:
        return 0.0
    link = find_link(infra, src_site, dst_site)
    if link is None:
        raise ValueError(f"no network link from {src_site!r} to {dst_site!r}")
    return transfer_time_s(size_mb, link.rtt_ms, link.bandwidth_mbps)

