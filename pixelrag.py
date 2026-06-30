"""Local Pixel-RAG for embedded datasheets — CLI entry point.

Examples
--------
  # 1. Build a visual index from a datasheet PDF
  python pixelrag.py ingest "um1724-stm32-nucleo64-boards-mb1136-stmicroelectronics.pdf"

  # 2. Ask questions (reader looks at the actual page images)
  python pixelrag.py ask "What is the default jumper configuration for JP5 (power)?"

  # 3. One-off interactive session
  python pixelrag.py chat
"""
import argparse
import os
import sys

import config


def _default_index_dir():
    return os.path.join(os.getcwd(), config.INDEX_DIR_NAME)


def cmd_ingest(args):
    import ingest

    ingest.build_index(args.pdf, args.out, args.max_pages, args.mode)


def _print_result(res, show_captions=False):
    print("\n" + "=" * 70)
    print(res["answer"])
    print("=" * 70)
    pages = ", ".join(str(p) for p in res["pages_used"])
    print(f"[retrieved pages: {pages}]")
    for h in res["hits"]:
        print(f"  - p.{h['page']:<4} score={h['score']:.3f}  [{h['source']}]  {os.path.basename(h['image_path'])}")
        if show_captions:
            snippet = " ".join(h["text"].split())[:200]
            print(f"      {snippet}...")


def cmd_ask(args):
    import ask as ask_mod

    res = ask_mod.ask(args.question, args.index, args.top_k)
    _print_result(res, args.captions)


def cmd_chat(args):
    import ask as ask_mod

    print("Pixel-RAG chat. Ask about the datasheet. Ctrl-C or 'exit' to quit.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        res = ask_mod.ask(q, args.index, args.top_k)
        _print_result(res, args.captions)
        print()


def main():
    p = argparse.ArgumentParser(description="Local Pixel-RAG for embedded datasheets (Ollama + qwen3-vl).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="Render a PDF to page images and build a visual index.")
    pi.add_argument("pdf", help="Path to the datasheet PDF.")
    pi.add_argument("-o", "--out", default=None, help="Output index dir (default: ./pixel_index next to PDF).")
    pi.add_argument("-n", "--max-pages", type=int, default=None, help="Only index the first N pages (quick trial).")
    pi.add_argument(
        "-m", "--mode", choices=["text", "vlm"], default="text",
        help="Retrieval indexing: 'text' (fast, PDF text layer + VLM fallback) or 'vlm' (describe every page, slow).",
    )
    pi.set_defaults(func=cmd_ingest)

    pa = sub.add_parser("ask", help="Ask a single question against an index.")
    pa.add_argument("question", help="Your question.")
    pa.add_argument("-i", "--index", default=_default_index_dir(), help="Index dir (default: ./pixel_index).")
    pa.add_argument("-k", "--top-k", type=int, default=None, help="Pages to feed the reader (default: config.TOP_K).")
    pa.add_argument("--captions", action="store_true", help="Also print retrieved page captions.")
    pa.set_defaults(func=cmd_ask)

    pc = sub.add_parser("chat", help="Interactive Q&A loop.")
    pc.add_argument("-i", "--index", default=_default_index_dir(), help="Index dir (default: ./pixel_index).")
    pc.add_argument("-k", "--top-k", type=int, default=None, help="Pages to feed the reader.")
    pc.add_argument("--captions", action="store_true", help="Also print retrieved page captions.")
    pc.set_defaults(func=cmd_chat)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
