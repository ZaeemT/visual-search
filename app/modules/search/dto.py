"""The request/response contract for visual search.

Written as models rather than plain dicts so the shape is validated on the way
out and published in the OpenAPI schema at /docs — which is also the contract
the Postman collection is built against.

The response serves three readers at once: a frontend (image_url, category,
colors), a non-technical stakeholder (explanation), and an engineer checking
performance (timings_ms). Nothing model-internal leaks into `explanation`.
"""

from pydantic import BaseModel, Field


class QueryEcho(BaseModel):
    """What the server understood the request to be, after defaults and clamping."""

    top_k: int
    explain: bool


class SearchResult(BaseModel):
    rank: int = Field(description="1 is the closest match")
    item_id: str
    image_url: str = Field(description="GET this path to view the catalogue image")
    category: str = Field(description="item type as annotated in the catalogue")
    colors: list[str]
    similarity: float = Field(description="cosine similarity, 0-1; higher is closer")
    explanation: str | None = Field(
        default=None, description="plain-text reason this item was returned"
    )


class Timings(BaseModel):
    """Milliseconds per stage. `explain` is 0 when explanations are off."""

    preprocess: float
    embed: float
    search: float
    explain: float = 0.0
    total: float


class SearchResponse(BaseModel):
    query: QueryEcho
    model: str = Field(description="the checkpoint that produced these results")
    catalogue_size: int
    results: list[SearchResult]
    timings_ms: Timings


class IndexInfo(BaseModel):
    """What the running server has loaded — useful for debugging a deployment."""

    model: str
    revision: str
    dim: int
    preprocessing: dict
    catalogue_size: int
    device: str
    built_at: str
