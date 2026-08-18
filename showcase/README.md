# Showcase

A self-contained page presenting the work, with a selector to fly five
different environments.

Live at **[drone.advancelabs.dev](https://drone.advancelabs.dev)**.

```bash
python showcase/generate.py     # one flight recording per scenario
python showcase/build.py        # -> showcase/index.html and showcase/dist/
open showcase/index.html

cd showcase/dist && vercel deploy --prod --scope advancelabs
```

The page is the work, not a description of it: one full-viewport deck, a
channel bar to switch environment, and no prose. Everything it says about a
flight is a number the simulator produced, so there is no copy to keep in sync.

## Mobile

Verified down to 320 px: no horizontal scroll, no clipped readouts, and the
rail wraps to two rows rather than three.

Touch gestures are handled explicitly because `touch-action: none` — required
for single-finger look to work — also disables the browser's own pinch. Without
the handlers in `viewer/index.html` there is simply no way to zoom on a phone.

| gesture | action |
|---|---|
| one finger | look around |
| two fingers, spread | zoom (chase distance while following) |
| two fingers, slide | pan |

Below 560 px the camera presets and speed control are hidden: those are
reachable by gesture, while play, loop and scrub are not.

## Favicon

The site's own HUD corner brackets framing the aircraft — drawn from the
subject rather than generic drone clipart, and the one mark already on every
page. Sized for 16 px first, which is the size that actually matters: an
earlier draft carried the heading stalk the viewer draws, and at 16 px it
merged with the dot into an ambiguous blob, so the brackets got heavier and
the stalk went.

Amber `#c9871f` sits between the light and dark accents, so one file holds on
a white tab strip and a dark one without needing a second.

Inlined as a data URI because the self-contained build has to carry everything
it needs; the same URI serves the deploy, so there is one icon rather than two
that can drift apart. `build.py` also writes `dist/favicon.svg`, which is what
some crawlers and link unfurlers look for.

## Attribution

City data is derived from OpenStreetMap, which is ODbL-licensed and requires
attribution on anything produced from it. The credit is in the top bar and is
not optional on a public deployment.

## How it is put together

**The renderer is not duplicated.** `build.py` lifts the `<script>` block out
of `viewer/index.html` verbatim, and `template.html` supplies the element ids
it expects. A fix to the camera or the drawing code reaches the showcase by
rebuilding, rather than by remembering to copy it across.

**The palette flows the other way.** The template declares the CSS custom
properties the renderer reads through `getComputedStyle`, so the 3D canvas
picks up the page's colours instead of carrying its own.

**Two builds, because the destinations want opposite things.**
`showcase/index.html` inlines every flight — required for the Artifact, whose
CSP blocks fetching anything the page did not ship with, and for opening the
file straight off disk. `showcase/dist/` ships the page with no flight data and
fetches each on selection, so first paint is **70 KB** rather than a megabyte.
Records cache after the first fetch, so switching back is instant, and the
download streams against `content-length` so the progress bar reports real
bytes rather than a guess.

### Record size

Flights are thinned on export, which took the largest from 406 KB to 138 KB:

| what | how |
|---|---|
| track | Ramer-Douglas-Peucker at 0.4 m, preserving phase boundaries |
| conductors | 5 display points per catenary instead of 9 |
| roads, sidewalks, footprints | RDP at 1.5 m / 0.8 m |
| coordinates | decimetres rather than centimetres |

RDP rather than a uniform stride because stride is blind to shape — it throws
away as much of a tight turn as of a straight transit. A 313 s survey survives
at **41 samples** with its path length within 0.08% of the original.

That is only safe because playback is driven by **time**, not by sample index.
The recorder already skipped stationary samples, so the track was never
uniform in time; advancing an index per tick spent as long crossing a forty-
second straight as rounding a corner. Positions interpolate between the two
samples bracketing the current time.

| file | role |
|---|---|
| `generate.py` | flies each scenario and writes `flights/<key>.json` |
| `template.html` | page shell: design, copy, and the ids the renderer needs |
| `controller.js` | the scenario selector |
| `build.py` | composes the three into `index.html` |
| `assets/favicon.svg` | the icon, inlined into the page and shipped as a file |

## Scenarios

Chosen so the selector shows genuinely different planning problems rather than
one map recoloured — a 9 m residential street and a 292 m tower district break
a planner in opposite directions.

| key | environment | what it exercises |
|---|---|---|
| `grid` | procedural city | moving traffic, replanning, coverage scoring |
| `princessanne` | Princess Anne Manor | 5 m AGL survey threading under distribution wire |
| `scarborough` | Scarborough | a car and a pedestrian as legal keep-out volumes |
| `northyork` | North York Centre | surveillance with real occlusion |
| `downtown` | Downtown Toronto | 292 m towers exceed the ceiling — route around, not over |

Regenerating is cheap: OSM responses are cached to `.osm-cache/`, so only the
first fetch of an area touches the network.

## Note on the reference models

`reference-models/` is deliberately **not** part of the showcase. Those are
third-party assets with no licence, and the page is built to be shareable.
