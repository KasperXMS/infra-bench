from ..schemas import ExecutionResult


def metric_cost(result: ExecutionResult, objective: str) -> float:
    if not result.feasible or result.metrics is None:
        return float("inf")
    if objective not in {"latency_s", "wan_mb", "monetary_cost"}:
        raise ValueError(f"unsupported objective: {objective}")
    return float(getattr(result.metrics, objective))


def normalized_regret(selected_cost: float, oracle_cost: float, epsilon: float = 1e-9) -> float:
    return (selected_cost - oracle_cost) / max(oracle_cost, epsilon)
