"""FastAPI server for Local Pixel-RAG — fancy UI, PDF ingest, streaming chat.

Run:  python server.py     (or)  uvicorn server:app --port 8000
Then open http://localhost:8000

Everything is local: the UI is served from ./static, retrieval/answering go to
Ollama on localhost. No external network is used.
"""
import json
import os
import re
import threading

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import ask as ask_mod
import config
import ingest as ingest_mod

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "static")
DATA = os.path.join(BASE, "data")
UPLOADS = os.path.join(DATA, "uploads")
INDEXES = os.path.join(DATA, "indexes")
ACTIVE_FILE = os.path.join(DATA, "active.json")
for d in (DATA, UPLOADS, INDEXES):
    os.makedirs(d, exist_ok=True)

app = FastAPI(title="MCU-RAG")

# Single-user app: one active index + one in-flight generation we can cancel.
_STATE = {"cancel": None}


# --- active-index helpers ---------------------------------------------------
def load_active():
    try:
        with open(ACTIVE_FILE, encoding="utf-8") as f:
            a = json.load(f)
        if os.path.isdir(a.get("index_dir", "")):
            return a
    except (OSError, json.JSONDecodeError):
        pass
    return None


def save_active(info):
    with open(ACTIVE_FILE, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def _safe_name(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "doc"


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


# --- pages / static ---------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/status")
def status():
    return JSONResponse({"active": load_active()})


@app.get("/api/page/{page_num}")
def page_image(page_num: int):
    active = load_active()
    if not active:
        return JSONResponse({"error": "no active index"}, status_code=404)
    path = os.path.join(active["index_dir"], config.PAGES_SUBDIR, f"page_{page_num:04d}.png")
    if not os.path.isfile(path):
        return JSONResponse({"error": "page not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


# --- ingest (SSE progress) --------------------------------------------------
@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...), mode: str = Form("text"), max_pages: str = Form("")):
    safe = _safe_name(file.filename or "doc")
    pdf_path = os.path.join(UPLOADS, safe + ".pdf")
    with open(pdf_path, "wb") as out:
        out.write(await file.read())
    out_dir = os.path.join(INDEXES, safe)
    mp = int(max_pages) if max_pages.strip().isdigit() else None
    mode = "vlm" if mode == "vlm" else "text"

    def gen():
        try:
            for ev in ingest_mod.build_index_stream(pdf_path, out_dir, mp, mode):
                if ev["event"] == "done":
                    info = {
                        "index_dir": ev["out_dir"],
                        "name": file.filename or safe,
                        "pages": ev["pages"],
                        "dim": ev["dim"],
                    }
                    save_active(info)
                    yield _sse({"type": "done", **info})
                else:
                    yield _sse({"type": ev["event"], **{k: v for k, v in ev.items() if k != "event"}})
        except Exception as e:  # noqa: BLE001 - surface any failure to the UI
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


# --- chat (SSE streaming answer, cancellable) -------------------------------
@app.post("/api/ask")
async def ask(req: Request):
    body = await req.json()
    question = (body.get("question") or "").strip()
    top_k = body.get("top_k") or None
    active = load_active()

    def gen():
        if not active:
            yield _sse({"type": "error", "message": "Ingest a PDF first."})
            return
        if not question:
            yield _sse({"type": "error", "message": "Empty question."})
            return
        cancel = threading.Event()
        _STATE["cancel"] = cancel
        try:
            for ev in ask_mod.ask_stream(question, active["index_dir"], top_k, cancel):
                yield _sse(ev)
        except Exception as e:  # noqa: BLE001
            yield _sse({"type": "error", "message": str(e)})
        finally:
            if _STATE.get("cancel") is cancel:
                _STATE["cancel"] = None

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.post("/api/stop")
def stop():
    cancel = _STATE.get("cancel")
    if cancel is not None:
        cancel.set()
        return JSONResponse({"stopped": True})
    return JSONResponse({"stopped": False})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn

    print("MCU-RAG UI -> http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
