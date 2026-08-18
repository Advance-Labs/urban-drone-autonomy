"use strict";

/* Environment selector.

   The viewer script (inlined above) already knows how to load and render a
   flight record -- it exposes `load(data)` plus the transport controls. This
   builds the channel bar and hands the right record over on selection.

   Everything the page says about a flight is a number that came out of the
   simulator. There is no copy here to keep in sync with the work. */

(function () {
  const scenarios = window.SCENARIOS || [];
  const flights = window.FLIGHTS || {};
  const strip = document.getElementById("strip");
  const ctx = document.getElementById("ctx");
  if (!strip || !scenarios.length) return;

  /* Built as DOM nodes, not markup. Every string comes from the build
     manifest, and a selector assembled by concatenation is an injection path
     the moment a place name contains an angle bracket. */
  function text(tag, cls, value) {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    el.textContent = value;
    return el;
  }

  const gap = strip.querySelector(".gap");
  let current = null;
  let pending = 0;

  /* The page is built two ways. Self-contained, every flight is already in
     `window.FLIGHTS` -- required for the Artifact, whose CSP blocks fetching
     anything the page did not ship with. Split, `FLIGHTS` is empty and each
     record is fetched on selection, which takes first paint from over a
     megabyte to tens of kilobytes.

     Records are cached after the first fetch, so switching back is instant. */
  const bar = document.getElementById("loadbar");
  const fill = document.getElementById("loadfill");

  function showProgress(fraction) {
    if (!bar) return;
    bar.hidden = false;
    if (fraction === null) {
      bar.dataset.indeterminate = "1";
    } else {
      delete bar.dataset.indeterminate;
      fill.style.width = `${Math.round(fraction * 100)}%`;
    }
  }

  function hideProgress() {
    if (!bar) return;
    bar.hidden = true;
    delete bar.dataset.indeterminate;
    fill.style.width = "0";
  }

  /* Streamed so progress is real rather than a guess.

     `response.json()` gives no visibility into the download, so a slow
     connection would sit on "loading…" with nothing moving. Reading the body
     in chunks against content-length reports actual bytes; where the header is
     absent the bar says so by sweeping instead of pretending to know. */
  async function download(key, onProgress) {
    const res = await fetch(`f/${key}.json`);
    if (!res.ok) throw new Error(String(res.status));

    const total = Number(res.headers.get("content-length")) || 0;
    if (!res.body || !total) {
      onProgress(null);
      return res.json();
    }

    const reader = res.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress(Math.min(1, received / total));
    }
    const merged = new Uint8Array(received);
    let at = 0;
    for (const c of chunks) { merged.set(c, at); at += c.length; }
    return JSON.parse(new TextDecoder().decode(merged));
  }

  function flightFor(key) {
    if (flights[key]) return Promise.resolve(flights[key]);
    // Show the bar the moment the request starts, not when the first byte
    // arrives. On a high-latency connection those are seconds apart, and
    // waiting for bytes means the user sees nothing at all until it is
    // nearly done.
    showProgress(null);
    return download(key, showProgress).then((data) => {
      flights[key] = data;
      return data;
    });
  }

  function select(key, button, s) {
    if (key === current) return;
    current = key;
    strip.querySelectorAll(".chan").forEach((b) => {
      const on = b === button;
      b.setAttribute("aria-selected", String(on));
      b.tabIndex = on ? 0 : -1;
    });

    const token = ++pending;
    if (ctx) ctx.textContent = flights[key] ? ctx.textContent : "loading…";

    flightFor(key)
      .then((data) => {
        // A slow fetch must not overwrite a newer selection.
        if (token !== pending) return;
        hideProgress();
        // `load` comes from the viewer script and handles framing, HUD wiring
        // and autoplay; none of that is reimplemented here.
        window.load(data);
        describe(s);
      })
      .catch((err) => {
        if (token !== pending) return;
        hideProgress();
        if (ctx) ctx.textContent = `could not load ${s.label} (${err.message})`;
      });
  }

  function describe(s) {

    const st = s.stats || {};
    const bits = [s.tests];
    if (st.coverage !== null && st.coverage !== undefined) {
      bits.push(`${st.coverage}% coverage`);
    }
    if (st.conductors) bits.push(`${st.conductors} conductors`);
    if (ctx) ctx.textContent = bits.join("  ·  ");
  }

  scenarios.forEach((s, i) => {
    const st = s.stats || {};
    const btn = document.createElement("button");
    btn.className = "chan";
    btn.type = "button";
    btn.setAttribute("role", "tab");
    btn.setAttribute("aria-selected", "false");
    btn.tabIndex = -1;
    btn.append(
      text("span", "nm", s.label),
      text(
        "span",
        "sub",
        `${st.buildings} bldg · ${st.tallest} m · ${Math.round(st.duration)} s`
      )
    );
    btn.addEventListener("click", () => select(s.key, btn, s));

    // Arrow keys move between tabs, which is what a tablist owes a keyboard.
    btn.addEventListener("keydown", (e) => {
      const dir = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (!dir) return;
      e.preventDefault();
      const all = [...strip.querySelectorAll(".chan")];
      const next = all[(all.indexOf(btn) + dir + all.length) % all.length];
      next.focus();
      next.click();
    });

    strip.insertBefore(btn, gap);
    if (i === 0) queueMicrotask(() => select(s.key, btn, s));
  });
})();
