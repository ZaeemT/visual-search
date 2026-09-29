"""Search service: owns the loaded retriever and shapes responses.

Loaded once at startup (see the lifespan in app/main.py) and reused for every
request. Loading costs seconds — model weights onto the GPU, a 12 MB matrix and
the metadata table into memory — while a query costs tens of milliseconds, so
doing this per request would dominate the endpoint entirely.

A FastAPI app is single-process here and the retriever is read-only after load,
so sharing one instance across requests is safe.
"""

import logging
import time
from pathlib import Path

from PIL import Image

from ...core.config import get_settings
from ...search.explain import Explainer
from ...search.retriever import Retriever
from ...search.signals import QuerySignals, build_match_signals
from .dto import IndexInfo, QueryEcho, SearchResponse, SearchResult, Timings

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self, retriever: Retriever, explainer: Explainer):
        self.retriever = retriever
        self.explainer = explainer

    @classmethod
    def load(cls) -> "SearchService":
        """Load the index and the model that built it, then warm them up."""
        settings = get_settings()

        started = time.perf_counter()
        # Retriever.open reads the manifest and refuses to run if the configured
        # model does not match the weights, revision or preprocessing the index
        # was built with — a mismatch would return plausible nonsense.
        retriever = Retriever.open(
            model=settings.EMBEDDING_MODEL,
            index_root=settings.INDEX_ROOT,
            catalogue_dir=settings.CATALOGUE_DIR,
            device=settings.EMBEDDING_DEVICE,
        )
        logger.info(
            "loaded %s on %s: %d items, dim %d (%.1fs)",
            retriever.embedder.name,
            retriever.embedder.device,
            retriever.index.size,
            retriever.index.dim,
            time.perf_counter() - started,
        )

        explainer = Explainer(
            model=settings.OLLAMA_MODEL,
            host=settings.OLLAMA_HOST,
            timeout=settings.OLLAMA_TIMEOUT,
            enabled=settings.EXPLAIN_WITH_LLM,
        )
        if settings.EXPLAIN_WITH_LLM:
            logger.info(
                "explanations: %s",
                f"ollama {settings.OLLAMA_MODEL} (category translation)"
                if explainer.available()
                else "original category names (ollama unreachable)",
            )

        service = cls(retriever, explainer)
        service._warm_up(settings.SEARCH_WARMUP_QUERIES)
        return service

    def _warm_up(self, count: int) -> None:
        """Run throwaway queries so the first real request is not the slow one.

        The first pass through the model allocates GPU buffers and compiles
        kernels, and the first touch of the embedding matrix faults it into
        memory. Measured cold, that costs roughly 10 ms extra on embed and 15x
        on search; measured warm, it disappears. Better a user never sees it.
        """
        if count < 1:
            return
        blank = Image.new("RGB", (512, 512), (128, 128, 128))
        for i in range(count):
            timings = self.retriever.search(blank, top_k=5).timings_ms
            logger.info("warm-up %d/%d: %.2f ms", i + 1, count, timings["total"])

    def search(self, data: bytes, top_k: int, explain: bool) -> SearchResponse:
        """One query image in, a ranked response out."""
        response = self.retriever.search_bytes(data, top_k=top_k)

        explain_ms = 0.0
        explanations: list[str | None] = [None] * len(response.results)
        if explain:
            started = time.perf_counter()
            query_signals = response.query_signals or QuerySignals()
            # Signals first, wording second: the sentence is assembled from
            # these facts, so it can only say what they contain.
            signals = [
                build_match_signals(query_signals, r.metadata, r.score)
                for r in response.results
            ]
            explanations = self.explainer.explain_many(signals)
            explain_ms = round((time.perf_counter() - started) * 1000, 2)

        results = [
            SearchResult(
                rank=result.rank,
                item_id=result.item_id,
                image_url=f"/search/images/{result.item_id}",
                category=result.metadata["name"],
                # The annotation repeats a colour once per detected region, so
                # "green, green, green" is common. Collapse it, keeping order.
                colors=list(dict.fromkeys(result.metadata["colors"])),
                similarity=round(result.score, 4),
                explanation=explanation,
            )
            for result, explanation in zip(response.results, explanations)
        ]

        timings = response.timings_ms
        return SearchResponse(
            query=QueryEcho(top_k=top_k, explain=explain),
            model=self.retriever.embedder.model_id,
            catalogue_size=response.catalogue_size,
            results=results,
            timings_ms=Timings(
                preprocess=timings["preprocess"],
                embed=timings["embed"],
                search=timings["search"],
                explain=explain_ms,
                total=round(timings["total"] + explain_ms, 2),
            ),
        )

    def image_path(self, item_id: str) -> Path | None:
        """The file for a catalogue ID, or None if that ID is not in the catalogue.

        The path comes from the catalogue table rather than from the request, so
        a crafted item_id cannot walk out of the images directory.
        """
        item = self.retriever.catalogue.get(item_id)
        if item is None:
            return None
        path = Path(item["file"])
        return path if path.exists() else None

    def info(self) -> IndexInfo:
        manifest = self.retriever.index.manifest
        embedder = manifest["embedder"]
        return IndexInfo(
            model=embedder["model_id"],
            revision=embedder["revision"],
            dim=embedder["dim"],
            preprocessing=embedder["preprocessing"],
            catalogue_size=self.retriever.index.size,
            device=self.retriever.embedder.device,
            built_at=manifest["built_at"],
        )
