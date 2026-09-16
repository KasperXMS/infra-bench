import json
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class HuggingFaceRowsClient:
    """Small dependency-free client for metadata-only benchmark ingestion."""

    rows_endpoint = "https://datasets-server.huggingface.co/rows"
    api_endpoint = "https://huggingface.co/api/datasets"

    def __init__(self, *, timeout_s: float = 30.0, page_size: int = 100) -> None:
        self.timeout_s = timeout_s
        self.page_size = page_size

    def _json(self, url: str) -> dict[str, Any]:
        request = Request(url, headers={"User-Agent": "infra-bench/0.1.0"})
        with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
            return json.load(response)

    def revision(self, dataset: str) -> str:
        payload = self._json(f"{self.api_endpoint}/{dataset}")
        revision = payload.get("sha")
        if not revision:
            raise ValueError(f"dataset API returned no revision for {dataset}")
        return str(revision)

    def rows(
        self,
        dataset: str,
        *,
        config: str = "default",
        split: str = "test",
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        offset = 0
        while limit is None or offset < limit:
            length = self.page_size if limit is None else min(self.page_size, limit - offset)
            query = urlencode(
                {
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                    "offset": offset,
                    "length": length,
                }
            )
            payload = self._json(f"{self.rows_endpoint}?{query}")
            page = [item["row"] for item in payload.get("rows", [])]
            if not page:
                break
            yield from page
            offset += len(page)
            if len(page) < length:
                break
