from ..schemas import InfraState


def perturb_bandwidth(
    infra: InfraState, src_site: str, dst_site: str, bandwidth_mbps: float, *, infra_id: str
) -> InfraState:
    if bandwidth_mbps <= 0:
        raise ValueError("bandwidth_mbps must be positive")
    updated = infra.model_copy(deep=True, update={"infra_id": infra_id})
    matched = False
    for link in updated.links:
        if {link.src_site, link.dst_site} == {src_site, dst_site}:
            link.bandwidth_mbps = bandwidth_mbps
            matched = True
    if not matched:
        raise ValueError(f"no link between {src_site!r} and {dst_site!r}")
    return updated


def perturb_executor_load(
    infra: InfraState, executor_id: str, load: float, *, infra_id: str
) -> InfraState:
    if not 0.0 <= load < 1.0:
        raise ValueError("load must be in [0, 1)")
    updated = infra.model_copy(deep=True, update={"infra_id": infra_id})
    for executor in updated.executors():
        if executor.executor_id == executor_id:
            executor.load = load
            return updated
    raise ValueError(f"unknown executor: {executor_id}")

