from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.query import router as query_router
from app.api.routes.upload import router as upload_router

app = FastAPI(
    title="Enterprise RAG Platform",
    description=(
        "Backend for a production-oriented, organization-aware, "
        "multi-format Retrieval-Augmented Generation platform."
    ),
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(upload_router)
app.include_router(query_router)
