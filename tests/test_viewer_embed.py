"""Security regression cover for embedding a flight record into the viewer.

A flight record is data from a file, and `serve.py` splices it into a
`<script>` block. Anything that can terminate that block early turns a JSON
file into script execution, so these pin the escaping.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SERVE = Path(__file__).resolve().parent.parent / "viewer" / "serve.py"
_spec = importlib.util.spec_from_file_location("viewer_serve", SERVE)
serve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serve)


class TestScriptContextEscaping:
    def test_closing_script_tag_cannot_break_out(self):
        """The HTML parser scans for `</script>` before any JS runs, so this
        is the whole attack: one such string ends the block early."""
        payload = {"aborted_reason": "</script><script>alert(1)</script>"}
        assert "</script>" not in serve.embed_json(payload)

    def test_html_comment_opener_is_escaped(self):
        assert "<!--" not in serve.embed_json({"note": "<!--"})

    @pytest.mark.parametrize("sep", [" ", " "])
    def test_js_line_separators_are_escaped(self, sep):
        """U+2028/9 are literal line terminators inside a JS string and would
        break the parse even though JSON considers them valid."""
        assert sep not in serve.embed_json({"note": f"a{sep}b"})

    def test_ordinary_payloads_survive_unchanged(self):
        payload = {
            "completed": True,
            "duration_s": 203.3,
            "track": [{"t": 1.0, "n": -2.5, "phase": "navigate"}],
        }
        assert json.loads(serve.embed_json(payload)) == payload

    def test_escaping_is_semantically_transparent(self):
        """`<\\/` is the same string to a JS parser, so the escape must not
        change what the page actually receives."""
        payload = {"s": "</div>"}
        restored = serve.embed_json(payload).replace("<\\/", "</")
        assert json.loads(restored) == payload


class TestTempFileHandling:
    def test_output_directory_is_private(self, tmp_path):
        """A predictable name in the shared temp dir is pre-creatable by
        another user, who could symlink it at a file worth clobbering."""
        record = tmp_path / "f.json"
        record.write_text(json.dumps({"track": [{"t": 0, "n": 0, "e": 0, "d": 0}]}))

        opened: list[str] = []
        original = serve.webbrowser.open
        serve.webbrowser.open = opened.append
        try:
            assert serve.main(["serve.py", str(record)]) == 0
        finally:
            serve.webbrowser.open = original

        assert opened, "should have opened a page"
        out = Path(opened[0].removeprefix("file://"))
        assert out.exists()
        # 0o700: owner only, which is what mkdtemp guarantees.
        assert out.parent.stat().st_mode & 0o077 == 0


class TestInputValidation:
    def test_missing_file_is_rejected(self, tmp_path):
        assert serve.main(["serve.py", str(tmp_path / "nope.json")]) == 1

    def test_malformed_json_is_rejected(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        assert serve.main(["serve.py", str(bad)]) == 1

    def test_record_without_a_track_is_rejected(self, tmp_path):
        empty = tmp_path / "empty.json"
        empty.write_text(json.dumps({"completed": True}))
        assert serve.main(["serve.py", str(empty)]) == 1
