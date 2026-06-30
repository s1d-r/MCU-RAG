"""Central configuration for the local Pixel-RAG pipeline.

Everything is local: rendering with PyMuPDF, retrieval + answering through Ollama.
Override any value with an environment variable of the same name (PIXELRAG_*).
"""
import os

# --- Ollama -----------------------------------------------------------------
OLLAMA_HOST = os.environ.get("PIXELRAG_OLLAMA_HOST", "http://localhost:11434")

# Vision-language model: writes page descriptions during ingest AND reads the
# actual page images to answer questions at query time.
VLM_MODEL = os.environ.get("PIXELRAG_VLM_MODEL", "qwen3-vl:8b")

# Small text-embedding model used to make page descriptions searchable.
EMBED_MODEL = os.environ.get("PIXELRAG_EMBED_MODEL", "nomic-embed-text")

# Request timeout (seconds). VLM calls on CPU can be slow, so keep this generous.
REQUEST_TIMEOUT = int(os.environ.get("PIXELRAG_TIMEOUT", "600"))

# --- Rendering --------------------------------------------------------------
# DPI for PDF -> PNG. 150-200 keeps small datasheet text legible to the VLM.
RENDER_DPI = int(os.environ.get("PIXELRAG_DPI", "170"))

# --- Retrieval --------------------------------------------------------------
# How many page images to feed the reader model when answering. Keep this small
# on GPUs that can't fully fit the VLM (each image adds significant prefill cost).
TOP_K = int(os.environ.get("PIXELRAG_TOP_K", "1"))

# Max long-side pixels for images sent to the VLM. The rendered pages are kept at
# full DPI for the lightbox, but a memory-constrained GPU chokes on big images
# (vision-token prefill is the main latency). Downscaling here is the biggest
# speed lever; raise it on a larger GPU for sharper fine-text reading.
MAX_IMAGE_PX = int(os.environ.get("PIXELRAG_MAX_IMAGE_PX", "1280"))

# Context window for VLM calls. Smaller -> more of the model fits on the GPU
# (less CPU offload) -> faster tokens. 4096 is plenty for one page + prompt.
NUM_CTX = int(os.environ.get("PIXELRAG_NUM_CTX", "4096"))

# --- Storage ----------------------------------------------------------------
INDEX_DIR_NAME = "pixel_index"   # created next to the source PDF / chosen output
PAGES_SUBDIR = "pages"
INDEX_JSON = "index.json"
EMBEDDINGS_NPY = "embeddings.npy"
