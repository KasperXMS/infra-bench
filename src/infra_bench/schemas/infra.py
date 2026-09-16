from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Executor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    executor_id: str
    site_id: str
    capabilities: list[str] = Field(default_factory=list)
    speed_factors: dict[str, float] = Field(default_factory=dict)
    load: float = Field(ge=0.0, lt=1.0)
    monetary_cost_per_s: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Site(BaseModel):
    model_config = ConfigDict(extra="forbid")

    site_id: str
    executors: list[Executor] = Field(default_factory=list)

    @model_validator(mode="after")
    def executor_sites_match(self) -> "Site":
        if any(executor.site_id != self.site_id for executor in self.executors):
            raise ValueError("executor site_id must match its containing site")
        return self


class NetworkLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    src_site: str
    dst_site: str
    rtt_ms: float = Field(ge=0.0)
    bandwidth_mbps: float = Field(gt=0.0)


class DataArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    site_id: str
    size_mb: float = Field(ge=0.0)
    kind: str


class InfraState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    infra_id: str
    sites: list[Site]
    links: list[NetworkLink] = Field(default_factory=list)
    artifacts: list[DataArtifact] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self) -> "InfraState":
        site_ids = [site.site_id for site in self.sites]
        if len(site_ids) != len(set(site_ids)):
            raise ValueError("site IDs must be unique")
        known_sites = set(site_ids)
        executor_ids = [executor.executor_id for site in self.sites for executor in site.executors]
        if len(executor_ids) != len(set(executor_ids)):
            raise ValueError("executor IDs must be unique")
        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("artifact IDs must be unique")
        if any(artifact.site_id not in known_sites for artifact in self.artifacts):
            raise ValueError("artifact references unknown site")
        if any(link.src_site not in known_sites or link.dst_site not in known_sites for link in self.links):
            raise ValueError("network link references unknown site")
        return self

    def executors(self) -> list[Executor]:
        return [executor for site in self.sites for executor in site.executors]

