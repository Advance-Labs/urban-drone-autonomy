#!/usr/bin/env python3
"""Compose the showcase page from the template, the viewer, and the flights.

The renderer is NOT duplicated here. `viewer/index.html` stays the single
source of truth: its `<script>` block is lifted verbatim and the template
supplies the same element ids it expects. That way a fix to the camera or the
drawing code reaches the showcase by rebuilding, rather than by remembering to
copy it across.

The template also declares the CSS custom properties the renderer reads
through `getComputedStyle`, so the 3D canvas picks up the showcase palette
instead of carrying its own.

    python showcase/build.py           # -> showcase/index.html

Output is one self-contained file: no external requests, which is what the
Artifact CSP requires and what makes the page work from a file:// URL.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
VIEWER = SIM / "viewer" / "index.html"
TEMPLATE = HERE / "template.html"
FLIGHTS = HERE / "flights"
OUTPUT = HERE / "index.html"

# Frame policy: CSP `frame-ancestors`, NOT `X-Frame-Options`.
#
# advancelabs.dev embeds this showcase in its research section, and X-Frame-Options cannot
# express an allowlist — its only useful values are DENY and SAMEORIGIN, so permitting one
# other origin is impossible with it. `frame-ancestors` takes a list and supersedes XFO in
# every browser that implements it, which is why XFO is dropped rather than kept alongside.
#
# This is still a real restriction: any origin NOT listed here is refused, so the page
# cannot be framed by an unrelated site. Widen the list only for origins we operate.
#
# A parent page gets NO readable signal when a frame is refused (no error event, nothing
# observable cross-origin), so a wrong value here shows up as a silently blank rectangle on
# the embedding page rather than as an error anywhere.
VERCEL_JSON = """{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "X-Content-Type-Options", "value": "nosniff" },
        { "key": "Referrer-Policy", "value": "strict-origin-when-cross-origin" },
        { "key": "Content-Security-Policy", "value": "frame-ancestors 'self' https://advancelabs.dev https://*.advancelabs.dev" }
      ]
    }
  ]
}
"""

# Order is the reading order of the selector: start with the clean synthetic
# case, then the real environments from easiest to most punishing.
SCENARIOS = [
    {
        "key": "grid",
        "label": "Test grid",
        "place": "Procedural city",
        "source": "Generated, seed 7",
        "tests": "Moving traffic · replanning on invalidation",
    },
    {
        "key": "princessanne",
        "label": "Princess Anne Manor",
        "place": "Etobicoke, Toronto, ON",
        "source": "OpenStreetMap",
        "tests": "5 m AGL · threading under distribution wire",
    },
    {
        "key": "scarborough",
        "label": "Scarborough",
        "place": "Toronto, ON",
        "source": "OpenStreetMap",
        "tests": "Car and pedestrian as legal keep-out volumes",
    },
    {
        "key": "northyork",
        "label": "North York Centre",
        "place": "Toronto, ON",
        "source": "OpenStreetMap",
        "tests": "Surveillance with real occlusion",
    },
    {
        "key": "downtown",
        "label": "Downtown",
        "place": "Toronto, ON",
        "source": "OpenStreetMap",
        "tests": "292 m towers exceed the ceiling · routes around, not over",
    },
]


def viewer_script() -> str:
    """Lift the renderer out of viewer/index.html, verbatim."""
    html = VIEWER.read_text()
    match = re.search(r"<script>\n(.*)\n</script>", html, re.DOTALL)
    if not match:
        raise SystemExit(f"no <script> block found in {VIEWER}")
    return match.group(1)


def embed_json(payload) -> str:
    """Serialise for embedding inside a <script> tag.

    `</script>` inside a string closes the block early and everything after it
    parses as markup. The HTML parser looks for that literal sequence before
    any JavaScript runs, so it has to be broken at the source level -- and
    `<\\/` is an identical string to JavaScript.
    """
    return (
        json.dumps(payload, separators=(",", ":"))
        .replace("</", "<\\/")
        .replace("<!--", "<\\!--")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )


def summarise(payload: dict) -> dict:
    """Numbers for the selector card and the environment table."""
    ctx = payload.get("context", {})
    city = ctx.get("city", {})
    footprints = city.get("footprints") or city.get("buildings") or []
    heights = [f.get("h") for f in footprints if isinstance(f, dict) and f.get("h")]
    heights += [b.get("h") for b in footprints if isinstance(b, dict) and b.get("h")]
    track = payload.get("track", [])
    alts = [-s["d"] for s in track] or [0.0]
    cov = ctx.get("coverage_summary")
    return {
        "buildings": len(footprints),
        "tallest": round(max(heights), 0) if heights else 0,
        "duration": payload.get("duration_s", 0),
        "samples": len(track),
        "peak_alt": round(max(alts), 1),
        "min_alt": round(min(a for a in alts if a > 0.2), 1) if any(a > 0.2 for a in alts) else 0.0,
        "completed": payload.get("completed", False),
        "coverage": round(cov["fraction"] * 100) if cov else None,
        "conductors": len(ctx.get("true_wires") or []),
    }


def main() -> int:
    if not FLIGHTS.exists():
        raise SystemExit("no flights: run `python showcase/generate.py` first")

    bundle, missing = {}, []
    for scenario in SCENARIOS:
        path = FLIGHTS / f"{scenario['key']}.json"
        if not path.exists():
            missing.append(scenario["key"])
            continue
        payload = json.loads(path.read_text())
        bundle[scenario["key"]] = payload
        scenario["stats"] = summarise(payload)

    if missing:
        raise SystemExit(f"missing flights: {', '.join(missing)}")

    manifest = [
        {k: v for k, v in s.items()} for s in SCENARIOS if s["key"] in bundle
    ]

    template = TEMPLATE.read_text()
    viewer = viewer_script()
    controller = (HERE / "controller.js").read_text()

    def compose(flights_literal: str) -> str:
        return (
            template
            + "<script>\n"
            + f"window.SCENARIOS = {embed_json(manifest)};\n"
            + f"window.FLIGHTS = {flights_literal};\n"
            + "</script>\n"
            + "<script>\n" + viewer + "\n</script>\n"
            + "<script>\n" + controller + "\n</script>\n"
        )

    # Two builds, because the two destinations want opposite things.
    #
    # Self-contained: every flight inlined. Required for the Artifact, whose
    # CSP blocks fetching anything the page did not ship with, and for opening
    # the file directly from disk.
    OUTPUT.write_text(compose(embed_json(bundle)))

    # Split: the page ships with no flight data and fetches each on selection.
    # First paint drops from over a megabyte to tens of kilobytes, which is
    # the difference between usable and not on a phone over cellular.
    dist = HERE / "dist"
    (dist / "f").mkdir(parents=True, exist_ok=True)
    dist_page = compose("{}")
    (dist / "index.html").write_text(dist_page)
    (dist / "vercel.json").write_text(VERCEL_JSON)
    # Also ship the icon as a file. The page inlines it, but a standalone
    # /favicon.svg is what some crawlers and link unfurlers look for.
    (dist / "favicon.svg").write_text((HERE / "assets" / "favicon.svg").read_text())
    for key, payload in bundle.items():
        (dist / "f" / f"{key}.json").write_text(
            json.dumps(payload, separators=(",", ":"))
        )

    inline_mb = OUTPUT.stat().st_size / 1024 / 1024
    split_kb = (dist / "index.html").stat().st_size / 1024
    largest_kb = max(
        (dist / "f" / f"{k}.json").stat().st_size for k in bundle
    ) / 1024
    print(
        f"self-contained  {OUTPUT.relative_to(SIM)}  {inline_mb:.2f} MB\n"
        f"split           dist/index.html  {split_kb:.0f} KB"
        f"  + {len(bundle)} flights (largest {largest_kb:.0f} KB)"
    )
    if inline_mb > 15:
        print("  WARNING: approaching the 16 MB artifact limit", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
