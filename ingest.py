"""Ingest: PDF -> page images -> searchable text -> embeddings -> local index.

Pixel-RAG keeps the *answering* step on pixels: every page is rendered to an
image and the reader VLM literally looks at it, so tables, pinout diagrams and
figures are never lost to a lossy text parse.

The *retrieval* step just needs to surface the right page. Two modes:

  text  (default) -- embed the PDF's own text layer. Instant, no VLM. Any page
                     whose text layer is empty/sparse (a scanned page or a pure
                     diagram) automatically falls back to a VLM description so it
                     is still findable.
  vlm              -- describe every page with the VLM. Slow, but best for
                     scanned documents with no usable text layer.

Either way the page images are stored and used for answering.
"""
import json
import os

import numpy as np
import fitz  # PyMuPDF
from tqdm import tqdm

import config
import ollama_client as oc

# Minimum chars of real text before we trust a page's text layer for retrieval.
MIN_TEXT_CHARS = 40

# Description the VLM writes when it has to look at a page (vlm mode, or the
# fallback for image-only pages). Tuned for embedded datasheets.
CAPTION_PROMPT = (
    "You are indexing a page from an electronics datasheet or reference manual. "
    "Describe this page in detail so it can be found later by a search query. "
    "Capture: the section/title, every part number and component name, all pin "
    "names/numbers and connector labels, the contents of any tables (row and "
    "column headers plus key values), what each figure, diagram, schematic or "
    "pinout shows, and any electrical specs, voltages, or jumper settings. "
    "Be thorough and factual. Do not invent details that are not on the page."
)


def _render_page(page, matrix, pages_dir, page_num):
    pix = page.get_pixmap(matrix=matrix)
    img_path = os.path.join(pages_dir, f"page_{page_num:04d}.png")
    pix.save(img_path)
    return img_path


def build_index_stream(pdf_path, out_dir=None, max_pages=None, mode="text"):
    """Build an index, yielding progress events. Use build_index() for a blocking call.

    Yields dicts: {"event": "start"|"page"|"done", ...}.
    """
    pdf_path = os.path.abspath(pdf_path)
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(pdf_path), config.INDEX_DIR_NAME)
    pages_dir = os.path.join(out_dir, config.PAGES_SUBDIR)
    os.makedirs(pages_dir, exist_ok=True)

    if mode == "vlm":
        oc.ensure_models(config.VLM_MODEL, config.EMBED_MODEL)
    else:
        oc.ensure_models(config.EMBED_MODEL)

    doc = fitz.open(pdf_path)
    n = doc.page_count if not max_pages else min(max_pages, doc.page_count)
    zoom = config.RENDER_DPI / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    yield {"event": "start", "total": n, "mode": mode, "name": os.path.basename(pdf_path)}

    records, vectors = [], []
    for i in range(n):
        page = doc.load_page(i)
        page_num = i + 1
        img_path = _render_page(page, matrix, pages_dir, page_num)

        source = mode
        if mode == "vlm":
            search_text = oc.caption_image(img_path, CAPTION_PROMPT)
        else:
            search_text = page.get_text().strip()
            if len(search_text) < MIN_TEXT_CHARS:  # image-only page -> let the VLM see it
                search_text = oc.caption_image(img_path, CAPTION_PROMPT)
                source = "vlm-fallback"

        vectors.append(oc.embed(search_text))
        records.append(
            {
                "page": page_num,
                "image": os.path.relpath(img_path, out_dir),
                "source": source,
                "text": search_text,
            }
        )
        yield {"event": "page", "done": page_num, "total": n, "page": page_num, "source": source}
    doc.close()

    embeddings = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-8, None)

    np.save(os.path.join(out_dir, config.EMBEDDINGS_NPY), embeddings)
    with open(os.path.join(out_dir, config.INDEX_JSON), "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_pdf": pdf_path,
                "mode": mode,
                "vlm_model": config.VLM_MODEL,
                "embed_model": config.EMBED_MODEL,
                "dpi": config.RENDER_DPI,
                "records": records,
            },
            f,
            indent=2,
        )
    yield {"event": "done", "out_dir": out_dir, "pages": len(records), "dim": int(embeddings.shape[1])}


def build_index(pdf_path, out_dir=None, max_pages=None, mode="text"):
    """Blocking build with a tqdm progress bar (used by the CLI)."""
    out_dir_final = None
    bar = None
    for ev in build_index_stream(pdf_path, out_dir, max_pages, mode):
        if ev["event"] == "start":
            print(f"Indexing '{ev['name']}' ({ev['total']} pages, mode={ev['mode']}, {config.RENDER_DPI} DPI)")
            bar = tqdm(total=ev["total"], desc="Pages")
        elif ev["event"] == "page":
            bar.update(1)
        elif ev["event"] == "done":
            if bar:
                bar.close()
            out_dir_final = ev["out_dir"]
            print(f"\nIndex built: {ev['out_dir']}")
            print(f"  {ev['pages']} pages, embedding dim = {ev['dim']}")
    return out_dir_final
