# MCU-RAG — ask your embedded datasheet anything

*Your embedded datasheet assistant — local, pixel-based, and grounded.*

A local, pixel-based RAG for MCU datasheets and reference manuals

A fully local Retrieval-Augmented Generation pipeline for datasheets / reference
manuals, inspired by [PixelRAG](https://github.com/StarTrail-org/PixelRAG).
Pages are rendered to **images** and the reader model **looks at the pixels** to
answer — so pinout diagrams, configuration tables, and figures that lossy text
extraction would mangle stay intact.

Everything runs on your machine through **[Ollama](https://ollama.com)**:

| Role | Model | Why |
|------|-------|-----|
| Reader (sees page images, answers) | `qwen3-vl:8b` | Vision-language model reads tables/diagrams off the page |
| Retrieval embeddings | `nomic-embed-text` | Small, fast text embedder to find the right page |

## How it differs from the original PixelRAG

The Berkeley PixelRAG paper retrieves over images directly using a LoRA-tuned
`Qwen3-VL-Embedding` model (torch + FAISS + GPU). That doesn't run through
Ollama. This project keeps PixelRAG's **core idea — answer from pixels, never a
lossy text parse —** but adapts retrieval to be practical on a single local GPU:

- **Answering is always visual.** The reader VLM is handed the actual rendered
  page image(s).
- **Retrieval is text-layer by default** (born-digital PDFs have an excellent
  text layer), with **automatic VLM-caption fallback** for any scanned /
  image-only page so it stays findable. A full `--mode vlm` is available for
  documents with no usable text layer.

## Setup

```powershell
# 1. Models (one-time)
ollama pull qwen3-vl:8b
ollama pull nomic-embed-text

# 2. Python deps
pip install -r requirements.txt
```

## Web UI (recommended)

A FastAPI app with an animated landing page, drag-and-drop PDF ingest (live
progress), and an in-page streaming chat with a working **Stop** button.

```powershell
python server.py
# open http://localhost:8000
```

Drop any PDF onto the landing page → watch it render/index → chat. Each answer
streams live; click **Stop** to cancel and free the GPU; click a page-citation
chip to view that page image. Nothing leaves your machine (UI is served locally;
retrieval/answering go to Ollama on localhost).

> **Stop latency:** Stop takes full effect once the answer *starts streaming*.
> While the model is still doing image prefill (the wait before the first token —
> ~1–2 min/image on an 8 GB GPU), it finishes prefill before cancelling.

## Run on Google Colab / Jupyter

`mcu_rag_colab.ipynb` runs the whole stack on a Colab GPU: it installs Ollama,
pulls the models, ingests an uploaded PDF, and can even expose the full web UI
through a free Cloudflare tunnel. A 16 GB Colab T4 fits the model entirely in
VRAM, so it is much faster than an 8 GB laptop. Open the notebook in Colab, set
`REPO_URL` to this repo, and run the cells top to bottom.

## Command line

```powershell
# Build a visual index from a datasheet PDF (renders pages + builds retrieval index)
python pixelrag.py ingest "um1724-stm32-nucleo64-boards-mb1136-stmicroelectronics.pdf"

# Ask a question — the reader looks at the retrieved page images
python pixelrag.py ask "Which STM32 pin is the user LED LD2 connected to?"

# Show which pages were retrieved + their text snippets
python pixelrag.py ask "What are the JP5 power jumper positions?" --captions

# Interactive loop
python pixelrag.py chat
```

### Useful flags

| Flag | Meaning |
|------|---------|
| `-n, --max-pages N` | Index only the first N pages (quick trial) |
| `-m, --mode {text,vlm}` | `text` (fast, default) or `vlm` (describe every page with the VLM — slow, for scanned docs) |
| `-k, --top-k N` | How many page images to feed the reader (default 2) |
| `-i, --index DIR` | Use a specific index directory |

## What gets created

```
pixel_index/
  pages/           page_0001.png ... one PNG per rendered page
  index.json       page records: page #, image path, retrieval text, source
  embeddings.npy   normalized embedding matrix (cosine search = dot product)
```

## Configuration

Edit `config.py` or set `PIXELRAG_*` environment variables:
`PIXELRAG_VLM_MODEL`, `PIXELRAG_EMBED_MODEL`, `PIXELRAG_DPI`, `PIXELRAG_TOP_K`,
`PIXELRAG_OLLAMA_HOST`, `PIXELRAG_TIMEOUT`.

## Performance notes

`qwen3-vl:8b` needs ~8 GB and won't fully fit an 8 GB GPU alongside its vision
encoder + context, so it partially offloads to CPU. Consequences on such a box:

- **Ingest (text mode):** ~2 s/page — fast, because the VLM isn't used.
- **Answering:** a few minutes per question (the VLM does heavy image prefill).
  Keep `--top-k` at 1–2. Lower `PIXELRAG_DPI` (e.g. 130) to speed prefill at the
  cost of fine-text legibility.
- `--mode vlm` captions every page with the VLM and is very slow on this
  hardware (minutes/page); use it only for scanned PDFs with no text layer, or on
  a bigger GPU.

## Files

- `server.py` — FastAPI app: serves the UI, PDF ingest (SSE progress), streaming chat, stop
- `static/` — the web UI (`index.html`, `style.css`, `app.js`)
- `mcu_rag_colab.ipynb` — Google Colab / Jupyter notebook version (GPU, optional tunneled UI)
- `pixelrag.py` — CLI (`ingest` / `ask` / `chat`)
- `ingest.py` — PDF → images → retrieval text → embeddings → index (with streaming progress)
- `ask.py` — embed query → cosine retrieve → answer from page images (blocking + streaming)
- `ollama_client.py` — Ollama HTTP client (embeddings + vision, blocking + streaming)
- `config.py` — models, DPI, top-k, paths
- `data/` — uploaded PDFs, per-document indexes, and the active-index pointer (created at runtime)
