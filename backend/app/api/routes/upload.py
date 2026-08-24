import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.api.dependencies import get_ingestion_service
from app.services.ingestion_service import (
    EmbeddingGenerationError,
    IndexingFailedError,
    IngestionService,
    InvalidInputError,
    UnsupportedFileTypeError,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingestion"])


class UploadResponse(BaseModel):
    status: str
    document_id: str
    filename: str
    chunk_count: int
    indexed_chunk_count: int


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_document(
    file: UploadFile,
    uploaded_by: str | None = Form(default=None),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> UploadResponse:
    """Accepts one file, runs it through the existing IngestionService,
    and reports the result. This route only handles HTTP concerns
    (validation, temp-file lifecycle, status-code mapping) -- loading,
    normalization, chunking, embedding, and indexing all remain
    IngestionService's responsibility.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    original_filename = file.filename
    # Only the suffix is taken from the client-supplied filename (needed
    # so loader dispatch by extension works); it is never used to build
    # the actual temp filesystem path, which tempfile generates itself.
    suffix = Path(original_filename).suffix.lower()

    if suffix not in ingestion_service.loaders:
        supported = ", ".join(sorted(ingestion_service.loaders))
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type {suffix!r}. Supported types: {supported}.",
        )

    fd, tmp_path_str = tempfile.mkstemp(suffix=suffix)
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            tmp_file.write(await file.read())

        result = ingestion_service.ingest(tmp_path, uploaded_by=uploaded_by)

        return UploadResponse(
            status=result.status,
            document_id=result.document_id,
            filename=original_filename,
            chunk_count=result.chunk_count,
            indexed_chunk_count=result.indexed_chunk_count,
        )

    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=415, detail="Unsupported file type.") from exc
    except InvalidInputError as exc:
        raise HTTPException(status_code=400, detail="Invalid upload request.") from exc
    except EmbeddingGenerationError as exc:
        logger.exception("Embedding generation failed for upload %r", original_filename)
        raise HTTPException(status_code=500, detail="Embedding generation failed.") from exc
    except IndexingFailedError as exc:
        logger.exception("Indexing failed for upload %r", original_filename)
        raise HTTPException(status_code=500, detail="Indexing failed.") from exc
    except FileNotFoundError as exc:
        logger.exception("Internal file-handling error for upload %r", original_filename)
        raise HTTPException(
            status_code=500, detail="Internal error while processing the uploaded file."
        ) from exc
    except ValueError as exc:
        # A loader-raised ValueError (e.g. corrupt/unparseable file
        # content) -- not an IngestionError, since loader exceptions are
        # intentionally left unwrapped. Treated as a client-input problem.
        raise HTTPException(
            status_code=400, detail="The uploaded file could not be parsed."
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during upload of %r", original_filename)
        raise HTTPException(status_code=500, detail="Unexpected error during ingestion.") from exc
    finally:
        tmp_path.unlink(missing_ok=True)
        await file.close()
