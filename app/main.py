"""Local app entry point: two fully independent routers, one per track,
plus a small browser UI (app/static/index.html) with one tab per track.

Run with: uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/ for the UI, or /docs for the raw API.
"""
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.routers.track_a import router as track_a_router
from app.routers.track_a import warm_up as warm_up_track_a
from app.routers.track_b import router as track_b_router

STATIC_DIR = Path(__file__).resolve().parent / "static"
log = logging.getLogger("uvicorn.error")


def _warm_up():
    try:
        warm_up_track_a()
        log.info("Track A models loaded.")
    except Exception as e:  # the endpoints will surface the error on use
        log.warning("Model warm-up failed: %s", e)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load models in the background so the server (and Track B) is usable at once.
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Mental-Health Support Prototype (Track A + Track B)",
    description=(
        "Educational, inference-only prototype. Track A (conversation + chatbot) and "
        "Track B (accelerometer + activity) are independent: neither reads the other's "
        "data, and neither produces a depression diagnosis or severity score."
    ),
)

app.include_router(track_a_router)
app.include_router(track_b_router)


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api")
def api_index():
    return {
        "tracks": ["/track-a", "/track-b"],
        "docs": "/docs",
        "note": "Inference-only educational prototype; not a clinical tool.",
    }
