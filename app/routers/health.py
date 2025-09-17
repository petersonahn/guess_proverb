from fastapi import APIRouter
from app.includes.dbconn import healthcheck

router = APIRouter(prefix="/health", tags=["health"])

@router.get("")
def health():
    ok = False
    try:
        ok = healthcheck()
    except Exception:
        ok = False
    return {"ok": ok}
