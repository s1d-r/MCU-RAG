"""Query: embed the question, retrieve top-k page images, answer from pixels."""
import json
import os

import numpy as np

import config
import ollama_client as oc

ANSWER_PROMPT = (
    "You are a precise hardware engineering assistant. The attached image(s) are, "
    "in order, the following pages of the datasheet: {pages}. Answer the question "
    "using ONLY what is visible in those page images. Read tables, pinout diagrams, "
    "and figures directly from the image. If the answer is a pin, value, or setting, "
    "state it exactly. Cite page numbers using ONLY the numbers listed above — do "
    "not guess any other page number. If the pages do not contain the answer, say "
    "so plainly.\n\n"
    "Question: {question}"
)


def load_index(index_dir):
    with open(os.path.join(index_dir, config.INDEX_JSON), encoding="utf-8") as f:
        meta = json.load(f)
    embeddings = np.load(os.path.join(index_dir, config.EMBEDDINGS_NPY))
    return meta, embeddings


def retrieve(question, index_dir, top_k=None):
    top_k = top_k or config.TOP_K
    meta, embeddings = load_index(index_dir)
    q = np.asarray(oc.embed(question), dtype=np.float32)
    q /= np.clip(np.linalg.norm(q), 1e-8, None)
    scores = embeddings @ q
    order = np.argsort(-scores)[:top_k]
    hits = []
    for rank, idx in enumerate(order, 1):
        rec = meta["records"][idx]
        hits.append(
            {
                "rank": rank,
                "page": rec["page"],
                "score": float(scores[idx]),
                "image_path": os.path.join(index_dir, rec["image"]),
                "text": rec["text"],
                "source": rec.get("source", "text"),
            }
        )
    return meta, hits


def ask(question, index_dir, top_k=None):
    _meta, hits = retrieve(question, index_dir, top_k)
    image_paths = [h["image_path"] for h in hits]
    pages = ", ".join(str(h["page"]) for h in hits)
    prompt = ANSWER_PROMPT.format(pages=pages, question=question)
    answer = oc.answer_with_images(prompt, image_paths)
    return {
        "question": question,
        "answer": answer,
        "pages_used": [h["page"] for h in hits],
        "hits": hits,
    }


def ask_stream(question, index_dir, top_k=None, cancel_event=None):
    """Yield progress events for the UI: a 'pages' event, then 'token' events, then 'done'."""
    _meta, hits = retrieve(question, index_dir, top_k)
    image_paths = [h["image_path"] for h in hits]
    pages = [h["page"] for h in hits]
    yield {
        "type": "pages",
        "pages": pages,
        "hits": [
            {"page": h["page"], "score": round(h["score"], 3), "source": h["source"]}
            for h in hits
        ],
    }
    prompt = ANSWER_PROMPT.format(pages=", ".join(str(p) for p in pages), question=question)
    for chunk in oc.stream_answer_with_images(prompt, image_paths, cancel_event):
        yield {"type": "token", "text": chunk}
    if cancel_event is not None and cancel_event.is_set():
        yield {"type": "cancelled"}
    else:
        yield {"type": "done"}
