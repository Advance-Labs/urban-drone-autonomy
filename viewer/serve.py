#!/usr/bin/env python3
"""Open the viewer with a flight record already loaded.

    python viewer/serve.py flight.json

Writes a self-contained copy of the viewer with the record embedded, then
opens it. Embedding rather than fetching avoids the file:// CORS restriction
that blocks a plain `fetch` from a local page.
"""

from __future__ import annotations

import json
import sys
import tempfile
import webbrowser
from pathlib import Path

VIEWER = Path(__file__).parent / "index.html"


def embed_json(payload: dict) -> str:
    """Serialise for embedding inside a <script> tag.

    JSON is not safe in an HTML script context as-is: a string containing
    `</script>` closes the block early and everything after it is parsed as
    markup, which turns a flight record into an injection vector. The HTML
    parser looks for that literal sequence before any JavaScript runs, so the
    fix is to break it at the source level -- `<\\/script>` is an identical
    string to JavaScript and invisible to the parser.

    `<!--` gets the same treatment for the legacy comment-in-script rule.
    """
    return (
        json.dumps(payload)
        .replace("</", "<\\/")
        .replace("<!--", "<\\!--")
        .replace(" ", "\\u2028")  # line separators are literal newlines
        .replace(" ", "\\u2029")  # inside JS strings, breaking the parse
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2

    record_path = Path(argv[1])
    if not record_path.exists():
        print(f"No such flight record: {record_path}", file=sys.stderr)
        return 1

    try:
        record = json.loads(record_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"{record_path} is not valid JSON: {exc}", file=sys.stderr)
        return 1

    if not record.get("track"):
        print(f"{record_path} has no track data to display.", file=sys.stderr)
        return 1

    html = VIEWER.read_text().replace(
        "<script>",
        f"<script>window.EMBEDDED_FLIGHT = {embed_json(record)};</script>\n<script>",
        1,
    )

    # A predictable name in the shared temp directory is pre-creatable by
    # another user, who could plant a symlink and have this overwrite an
    # arbitrary file. A private directory created with 0700 avoids that.
    out_dir = Path(tempfile.mkdtemp(prefix="flight-replay-"))
    out = out_dir / f"{record_path.stem}.html"
    out.write_text(html)
    print(f"opening {out}")
    webbrowser.open(out.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
