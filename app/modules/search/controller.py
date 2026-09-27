"""HTTP endpoints for visual search.

    POST /search/                 upload an image, get ranked similar items
    GET  /search/images/{id}      serve a catalogue image so results can be seen
    GET  /search/info             what index and model the server has loaded

Multipart upload is used for the image because it is what browsers, Postman and
curl all do natively — no base64 wrapping, no JSON envelope around binary.
"""

import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from ...core.config import get_settings
from ...search.retriever import BadImage
from .dto import IndexInfo, SearchResponse
from .service import SearchService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_service(request: Request) -> SearchService:
    """The service loaded at startup. 503 rather than a cold load per request."""
    service = getattr(request.app.state, "search_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="search index is not loaded; check the server logs for the startup error",
        )
    return service


@router.post("/", response_model=SearchResponse)
async def search(
    request: Request,
    image: UploadFile = File(..., description="the query image (jpeg/png/webp)"),
    top_k: int = Form(None, description="how many results to return"),
    explain: bool = Form(True, description="generate an explanation per result"),
) -> SearchResponse:
    """Find catalogue items that look like the uploaded image."""
    settings = get_settings()
    service = get_service(request)

    # Clamp rather than reject: asking for 4,000 results should not 400, but it
    # should not be honoured either.
    if top_k is None:
        top_k = settings.SEARCH_TOP_K_DEFAULT
    top_k = max(1, min(top_k, settings.SEARCH_TOP_K_MAX))

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    if len(data) > settings.SEARCH_MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"image exceeds {settings.SEARCH_MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    try:
        return service.search(data, top_k=top_k, explain=explain)
    except BadImage as exc:
        # The upload arrived fine but is not a usable image — the client's problem.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/images/{item_id}")
async def get_image(request: Request, item_id: str) -> FileResponse:
    """Serve one catalogue image by ID, as referenced by `image_url` in results."""
    service = get_service(request)
    path = service.image_path(item_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"no catalogue item {item_id!r}")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/info", response_model=IndexInfo)
async def info(request: Request) -> IndexInfo:
    """What is loaded: model, revision, preprocessing, catalogue size, device."""
    return get_service(request).info()
