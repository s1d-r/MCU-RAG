"""Thin client over the local Ollama HTTP API — embeddings + vision generation."""
import base64
import json
import sys

import requests

import config


def _url(path):
    return f"{config.OLLAMA_HOST.rstrip('/')}{path}"


def _downscaled_bytes(image_path):
    """Return PNG bytes, shrunk so the long side <= config.MAX_IMAGE_PX.

    Big images dominate VLM prefill latency on a memory-constrained GPU, so we
    downscale before sending. Falls back to the raw file if Pillow is missing.
    """
    try:
        import io

        from PIL import Image

        with Image.open(image_path) as im:
            w, h = im.size
            longest = max(w, h)
            im = im.convert("RGB")
            if longest > config.MAX_IMAGE_PX:
                scale = config.MAX_IMAGE_PX / longest
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:  # noqa: BLE001 - any failure -> send the original
        with open(image_path, "rb") as f:
            return f.read()


def _b64(image_path):
    return base64.b64encode(_downscaled_bytes(image_path)).decode("ascii")


def list_models():
    """Return the set of model names currently available in Ollama."""
    try:
        r = requests.get(_url("/api/tags"), timeout=30)
        r.raise_for_status()
        return {m["name"] for m in r.json().get("models", [])}
    except requests.RequestException as e:
        raise SystemExit(
            f"Could not reach Ollama at {config.OLLAMA_HOST}. Is it running?\n  {e}"
        )


def ensure_models(*models):
    """Fail early with a helpful message if a required model isn't pulled."""
    have = list_models()
    # Ollama reports tags as 'name:tag'; accept a bare name too (':latest').
    norm = {m.split(":")[0] for m in have} | have
    missing = [m for m in models if m not in have and m.split(":")[0] not in norm]
    if missing:
        hint = "\n".join(f"  ollama pull {m}" for m in missing)
        raise SystemExit(
            f"Missing Ollama model(s): {', '.join(missing)}\nPull them first:\n{hint}"
        )


def embed(text):
    """Return the embedding vector (list[float]) for a string."""
    r = requests.post(
        _url("/api/embed"),
        json={"model": config.EMBED_MODEL, "input": text},
        timeout=config.REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    embs = data.get("embeddings") or ([data["embedding"]] if "embedding" in data else None)
    if not embs:
        raise RuntimeError(f"No embedding returned: {data}")
    return embs[0]


def caption_image(image_path, prompt):
    """Ask the VLM to describe a single page image (used during ingest)."""
    return _generate(prompt, [image_path])


def answer_with_images(prompt, image_paths):
    """Ask the VLM a question grounded in one or more page images."""
    return _generate(prompt, image_paths)


def _generate(prompt, image_paths):
    payload = {
        "model": config.VLM_MODEL,
        "prompt": prompt,
        "images": [_b64(p) for p in image_paths],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": config.NUM_CTX},
    }
    r = requests.post(_url("/api/generate"), json=payload, timeout=config.REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def stream_answer_with_images(prompt, image_paths, cancel_event=None):
    """Yield the VLM answer in chunks. Stops early if cancel_event is set.

    Exiting the loop closes the HTTP connection, which tells Ollama to stop
    generating — so the Stop button actually frees the GPU.
    """
    payload = {
        "model": config.VLM_MODEL,
        "prompt": prompt,
        "images": [_b64(p) for p in image_paths],
        "stream": True,
        "options": {"temperature": 0.1, "num_ctx": config.NUM_CTX},
    }
    with requests.post(
        _url("/api/generate"), json=payload, stream=True, timeout=config.REQUEST_TIMEOUT
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if cancel_event is not None and cancel_event.is_set():
                break
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk = obj.get("response", "")
            if chunk:
                yield chunk
            if obj.get("done"):
                break


if __name__ == "__main__":
    # Quick connectivity smoke test: python ollama_client.py
    print("Models available:", ", ".join(sorted(list_models())) or "(none)")
    sys.exit(0)
