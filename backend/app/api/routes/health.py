from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness check. Intentionally has no dependencies on the database
    or any embedding provider -- it only confirms the application
    process itself is running and routable.
    """
    return {"status": "healthy"}
