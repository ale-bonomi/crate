"""Build a self-contained listening page for the sliced hits.

usage: listening_page.py <slice out dir> <chops dir> <out html> [max hits]
"""

import base64
import html
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import lameenc
import numpy as np
import soundfile as sf

proto, chops, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 60
kept = json.loads((proto / "kept.json").read_text())
report = json.loads((proto / "report.json").read_text())

random.seed(11)
by_label = defaultdict(list)
for h in kept:
    by_label[h["label"]].append(h)
order = ["kick", "snare", "clap", "hat", "perc"]
per = max(1, limit // max(1, len(by_label)))
picked = []
for lab in order:
    hs = by_label.get(lab, [])
    picked += random.sample(hs, min(per, len(hs)))


def wav_uri(path: Path) -> str:
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode()


def mp3_uri(path: Path, seconds: float = 12.0) -> str:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    data = data[: int(sr * seconds)]
    pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16)
    enc = lameenc.Encoder()
    enc.set_bit_rate(96)
    enc.set_in_sample_rate(sr)
    enc.set_channels(pcm.shape[1])
    enc.set_quality(5)
    mp3 = enc.encode(pcm.tobytes()) + enc.flush()
    return "data:audio/mpeg;base64," + base64.b64encode(mp3).decode()


chop_names = sorted({h["chop"] for h in picked})[:6]
chop_rows = "".join(
    f'<div class="chop"><div class="name">{html.escape(c)}</div><audio controls preload="none" src="{mp3_uri(chops / c)}"></audio></div>'
    for c in chop_names
)

KINDS = [("kick", "Kick"), ("snare", "Snare"), ("rim", "Rim"), ("clap", "Clap"),
         ("hat", "Hat"), ("tom", "Tom"), ("perc", "Perc")]
cards = []
for k, h in enumerate(picked):
    bleed = "clean" if h["bleed"] < 0.5 else "a little of the song underneath" if h["bleed"] < 1.5 else "a lot of the song underneath"
    btns = "".join(f'<button type="button" id="h{k}-{v}" data-v="{v}">{t}</button>' for v, t in KINDS)
    cards.append(f"""
<article class="card" data-id="h{k}" data-file="{html.escape(h['file'])}" data-label="{h['label']}">
  <header class="top"><span class="tag">Crate says {h['label']}</span><span class="meta">{h['prob']:.0%} sure · {h.get('secs', '?')} s · {bleed}</span></header>
  <audio controls preload="none" src="{wav_uri(proto / 'hits' / h['file'])}"></audio>
  <p class="from">from {html.escape(h['chop'])}</p>
  <div class="votes" role="group" aria-label="Drums you hear (pick all)">{btns}<button type="button" id="h{k}-fill" class="flag" data-flag="fill">Fill</button></div>
  <p class="hint" hidden>You said two at once. Tap each drum you hear.</p>
  <div class="flags"><label class="short"><input type="checkbox" id="h{k}-short" data-flag="short"> Cut too short</label><button type="button" id="h{k}-junk" class="flag junk" data-flag="junk">Junk</button></div>
</article>""")

with_drums = [r for r in report if r.get("note") != "barely any drums"]
stats = (f"{len(report)} random chops · {len(with_drums)} had drums · "
         f"{sum(r['hits'] for r in report)} hits found · {len(kept)} kept · {len(picked)} here")

page = f"""<title>Chop Slicing Test</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@600&display=swap">
<style>
/* Layout: one listening column, a card per hit, a verdict bar pinned to the bottom. */
:root {{ --bg:#f5f6f8; --card:#ffffff; --ink:#1b1e24; --muted:#626a78; --line:#dde1e8; --accent:#3b5bdb;
        --ok:#2b8a3e; --warn:#c26a00; --bad:#c92a2a; --on:#ffffff;
        --body:"Inter",system-ui,-apple-system,sans-serif; --display:"Space Grotesk","Inter",system-ui,sans-serif; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#12141a; --card:#1b1e26; --ink:#e8eaf0;
        --muted:#9aa2b1; --line:#2b303b; --accent:#7f9cff; --ok:#51cf66; --warn:#ffa94d; --bad:#ff6b6b; --on:#12141a; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#12141a; --card:#1b1e26; --ink:#e8eaf0; --muted:#9aa2b1; --line:#2b303b; --accent:#7f9cff;
        --ok:#51cf66; --warn:#ffa94d; --bad:#ff6b6b; --on:#12141a; color-scheme:dark; }}
* {{ box-sizing:border-box }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.5 var(--body);
        padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px)); }}
main {{ max-width:720px; margin:0 auto; padding-block:24px 120px; display:flex; flex-direction:column; gap:12px }}
h1 {{ font:600 24px/1.2 var(--display); margin:0; text-wrap:balance }}
h2 {{ font:600 15px/1.3 var(--display); margin:16px 0 0; text-transform:uppercase; letter-spacing:.06em; color:var(--muted) }}
p {{ margin:0; max-width:65ch }}
.lede {{ color:var(--muted) }}
.stats {{ font-size:13px; color:var(--muted); font-variant-numeric:tabular-nums }}
.card,.chop {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px; display:flex; flex-direction:column; gap:6px; min-width:0 }}
.top {{ display:flex; gap:8px; align-items:baseline; flex-wrap:wrap }}
.tag {{ font:600 16px/1.2 var(--display); text-transform:capitalize; color:var(--accent) }}
.meta,.from,.name {{ font-size:13px; color:var(--muted); overflow-wrap:anywhere }}
audio {{ width:100%; height:40px }}
.votes {{ display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:6px }}
.short {{ display:flex; gap:8px; align-items:center; font-size:13.5px; color:var(--muted); min-height:32px }}
.short input {{ width:18px; height:18px; accent-color:var(--warn) }}
.votes button {{ min-height:44px; padding:0 2px; border-radius:10px; border:1px solid var(--line); background:transparent;
        color:var(--ink); font:500 14px var(--body); cursor:pointer }}
.votes button:focus-visible,.bar button:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px }}
.votes button.on {{ background:var(--accent); border-color:var(--accent); color:var(--on) }}
.votes button.on.match {{ background:var(--ok); border-color:var(--ok) }}
.votes button.flag.on {{ background:var(--warn); border-color:var(--warn) }}
.flags {{ display:flex; gap:10px; align-items:center; justify-content:space-between }}
.flags .junk {{ min-height:36px; padding:0 14px; border-radius:10px; border:1px solid var(--line); background:transparent; color:var(--ink); font:500 14px var(--body); cursor:pointer }}
.flags .junk.on {{ background:var(--bad); border-color:var(--bad); color:var(--on) }}
.hint {{ font-size:13px; color:var(--warn) }}
.bar {{ position:fixed; left:0; right:0; bottom:0; background:var(--card); border-top:1px solid var(--line);
        padding:10px max(16px, env(safe-area-inset-right, 0px)) calc(10px + env(safe-area-inset-bottom, 0px)) max(16px, env(safe-area-inset-left, 0px)) }}
.bar-inner {{ max-width:720px; margin:0 auto; display:flex; gap:12px; align-items:center; flex-wrap:wrap }}
.bar button {{ min-height:44px; padding:0 16px; border-radius:10px; border:0; background:var(--accent); color:var(--on); font:600 14px var(--body) }}
#count {{ font-size:13px; color:var(--muted); font-variant-numeric:tabular-nums; flex:1; min-width:0 }}
</style>
<main>
<h1>Chop slicing test · round 2</h1>
<p class="lede">Same 40 chops, new cuts. Hits now ring out (snares at least half a second) instead of stopping at the next hit,
and Crate prefers moments where the drum plays alone. Tap <b>every drum you hear</b>; green means Crate's label is among them.
Add <b>Fill</b> when it's several hits in a row, <b>Junk</b> when it isn't usable, and tick <b>Cut too short</b> if the tail stops early.
Your answers save as you go and become training examples, so rims and toms finally get their own.</p>
<p class="stats">Round 1: kicks 12 of 12 usable, snares 9 of 12, hats 10 of 12, perc 6 of 12, claps 1 of 12 (the rest were snares or rims). Only 1 of 60 was junk.</p>
<p class="stats">{stats}</p>
<h2>Original chops, for comparison</h2>{chop_rows}
<h2>Hits</h2>{''.join(cards)}
</main>
<div class="bar"><div class="bar-inner"><span id="count">Connecting…</span><button type="button" id="copy" hidden>Copy my answers</button></div></div>
<script>
const TOTAL = {len(picked)};
const KEY = "chop-slicing-round2b";
const cards = [...document.querySelectorAll(".card")];
let votes = {{}};
let db = null;
let saving = 0;
try {{ votes = JSON.parse(localStorage.getItem(KEY) || "{{}}") || {{}}; }} catch (e) {{ votes = {{}}; }}

function remember() {{ try {{ localStorage.setItem(KEY, JSON.stringify(votes)); }} catch (e) {{}} }}
const answered = x => x && ((x.kinds && x.kinds.length) || x.junk);
function fromDoc(x) {{
  // Earlier answers stored one choice in `is`; now several drums plus flags.
  const kinds = Array.isArray(x.kinds) ? x.kinds.filter(k => typeof k === "string") :
    (typeof x.is === "string" && !["two", "junk"].includes(x.is) ? [x.is] : []);
  return {{ kinds, fill: x.fill === true, junk: x.junk === true || x.is === "junk", short: x.short === true,
           two: x.is === "two" && !Array.isArray(x.kinds) }};
}}
function paint(note) {{
  cards.forEach(c => {{
    const mine = votes[c.dataset.id] || {{ kinds: [] }};
    c.querySelectorAll(".votes button[data-v]").forEach(b => {{
      const on = (mine.kinds || []).includes(b.dataset.v);
      b.classList.toggle("on", on); b.classList.toggle("match", on && b.dataset.v === c.dataset.label);
      b.setAttribute("aria-pressed", on ? "true" : "false");
    }});
    c.querySelectorAll("button.flag").forEach(b => {{
      const on = !!mine[b.dataset.flag];
      b.classList.toggle("on", on); b.setAttribute("aria-pressed", on ? "true" : "false");
    }});
    c.querySelector("input[data-flag=short]").checked = !!mine.short;
    c.querySelector(".hint").hidden = !(mine.two && !(mine.kinds || []).length);
  }});
  const n = Object.values(votes).filter(answered).length;
  document.getElementById("count").textContent = note || (n + " of " + TOTAL + " answered" + (db ? (saving ? " · saving…" : " · saved") : ""));
}}

async function save(c) {{
  const id = c.dataset.id, mine = votes[id];
  remember(); paint();
  if (!db) return;
  saving++; paint();
  try {{
    const ref = db.collection("round2").doc(id);
    if (mine && (answered(mine) || mine.short || mine.fill || mine.two))
      await ref.set({{ kinds: mine.kinds || [], fill: !!mine.fill, junk: !!mine.junk, short: !!mine.short,
                      label: c.dataset.label, file: c.dataset.file }});
    else await ref.delete();
  }} catch (e) {{
    paint("Couldn't save that tap (" + (e && e.code || "error") + "). Use Copy my answers instead.");
    document.getElementById("copy").hidden = false;
  }} finally {{ saving--; if (saving === 0) paint(); }}
}}
function current(c) {{
  const m = votes[c.dataset.id] || {{}};
  return {{ kinds: [...(m.kinds || [])], fill: !!m.fill, junk: !!m.junk, short: !!m.short, two: false }};
}}
cards.forEach(c => {{
  c.querySelectorAll(".votes button[data-v]").forEach(b => b.addEventListener("click", () => {{
    const mine = current(c), k = b.dataset.v;
    mine.kinds = mine.kinds.includes(k) ? mine.kinds.filter(x => x !== k) : [...mine.kinds, k];
    if (mine.kinds.length) mine.junk = false;
    votes[c.dataset.id] = mine; save(c);
  }}));
  c.querySelectorAll("button.flag").forEach(b => b.addEventListener("click", () => {{
    const mine = current(c), f = b.dataset.flag;
    mine[f] = !mine[f];
    if (f === "junk" && mine.junk) mine.kinds = [];
    votes[c.dataset.id] = mine; save(c);
  }}));
  c.querySelector("input[data-flag=short]").addEventListener("change", e => {{
    const mine = current(c); mine.short = e.target.checked; votes[c.dataset.id] = mine; save(c);
  }});
}});

document.getElementById("copy").addEventListener("click", async () => {{
  const lines = cards.filter(c => answered(votes[c.dataset.id])).map(c => {{ const v = votes[c.dataset.id];
    return c.dataset.id + " said " + c.dataset.label + " is " + (v.junk ? "junk" : v.kinds.join("+")) + (v.fill ? " fill" : "") + (v.short ? " short" : ""); }});
  const text = "chop slicing votes:\\n" + lines.join("\\n");
  try {{ await navigator.clipboard.writeText(text); paint("Copied. Paste it into the chat."); }}
  catch (e) {{ paint("Copy didn't work here. Tell Claude your answers in the chat."); }}
}});

paint();
(async () => {{
  const use = window.claude && window.claude.use;
  db = use ? await window.claude.use("db") : null;
  if (!db) {{ document.getElementById("copy").hidden = false; paint(Object.values(votes).filter(answered).length + " of " + TOTAL + " answered · tap Copy my answers when done"); return; }}
  let settled = false;
  const timer = setTimeout(() => {{ if (!settled) paint("Can't reach saved answers yet. Taps still save."); }}, 10000);
  db.collection("round2").onSnapshot(snap => {{
    if (!snap.metadata.fromCache) {{ settled = true; clearTimeout(timer); }}
    const fresh = {{}};
    snap.docs.forEach(d => {{ const x = d.data(); if (x) fresh[d.id] = fromDoc(x); }});
    votes = fresh; remember(); paint();
  }}, err => {{
    settled = true; clearTimeout(timer); db = null;
    document.getElementById("copy").hidden = false; paint("Saving isn't available. Tap Copy my answers when done.");
  }});
}})();
</script>"""
out.write_text(page)
print(out, f"{out.stat().st_size / 1e6:.1f} MB", len(picked), "hits", len(chop_names), "chops")
