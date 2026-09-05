#!/usr/bin/env python3
"""Render bench_autoencoders.py output into one self-contained HTML page.

    python make_ae_charts.py bench.json charts.html

Six charts -- VRAM, latency, accuracy, for the encoder and the decoder -- eight lines
each. No external assets: the SVG is inline, so the page works offline.
"""
import json, sys, math, html

if len(sys.argv) < 3:
    raise SystemExit("usage: make_ae_charts.py <bench.json> <out.html>")
rows = json.load(open(sys.argv[1]))
OUT = sys.argv[2]

SERIES = [  # (model, prec, chunked) -> label, colour, dash
    ("same-l", "16",  True ), ("same-l", "16",  False),
    ("same-l", "fp8", True ), ("same-l", "fp8", False),
    ("same-s", "16",  True ), ("same-s", "16",  False),
    ("same-s", "fp8", True ), ("same-s", "fp8", False),
]
COLOR = {("same-l","16"):"#e07b39", ("same-l","fp8"):"#a8341b",
         ("same-s","16"):"#3b9ad9", ("same-s","fp8"):"#1d5b86"}
def label(m,p,c):
    pn = {"16":"16-bit","fp8":"fp8"}[p]
    return f"{m.upper()} · {pn} · {'chunked' if c else 'single-shot'}"

def series_points(m,p,c,key):
    out=[]
    for r in rows:
        if (r["model"],r["prec"],r["chunked"])==(m,p,c) and key in r:
            v=r[key]
            if isinstance(v,(int,float)) and v==v and v>0:
                out.append((r["L"], v))
    return sorted(out)

W,H = 1180, 560
PAD = dict(l=92, r=28, t=26, b=64)
def chart(key, title, ylab, logy, unit, note=""):
    data = {s: series_points(*s, key) for s in SERIES}
    pts  = [v for s in data.values() for _,v in s]
    xs   = sorted({x for s in data.values() for x,_ in s})
    if not pts: return f"<p>no data for {key}</p>"
    lo, hi = min(pts), max(pts)
    if logy:
        lo = max(lo, hi/1e4); ylo, yhi = math.log10(lo*0.85), math.log10(hi*1.18)
        ytr = lambda v: math.log10(max(v,lo*0.85))
    else:
        span=hi-lo or 1; ylo, yhi = lo-span*0.10, hi+span*0.12; ytr = lambda v: v
    xlo, xhi = math.log10(min(xs)*0.85), math.log10(max(xs)*1.18)
    iw, ih = W-PAD['l']-PAD['r'], H-PAD['t']-PAD['b']
    X = lambda L: PAD['l'] + (math.log10(L)-xlo)/(xhi-xlo)*iw
    Y = lambda v: PAD['t'] + ih - (ytr(v)-ylo)/(yhi-ylo)*ih
    g=[f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{html.escape(title)}">']
    # y grid
    if logy:
        ticks=[]; e=math.floor(ylo)
        while e<=math.ceil(yhi):
            for mlt in (1,2,5):
                v=mlt*10**e
                if lo*0.85<=v<=hi*1.18: ticks.append(v)
            e+=1
    else:
        ticks=[ylo+(yhi-ylo)*i/6 for i in range(7)]
    for v in ticks:
        y=Y(v)
        g.append(f'<line x1="{PAD["l"]}" x2="{W-PAD["r"]}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        t=(f"{v:,.0f}" if v>=100 or (not logy and abs(v)>=10) else f"{v:,.2f}".rstrip("0").rstrip("."))
        g.append(f'<text x="{PAD["l"]-10}" y="{y+4:.1f}" class="ytick">{t}</text>')
    for L in [1,4,16,64,256,1024,4096,8192]:
        if not (min(xs)*0.85<=L<=max(xs)*1.18): continue
        x=X(L)
        g.append(f'<line y1="{PAD["t"]}" y2="{H-PAD["b"]}" x1="{x:.1f}" x2="{x:.1f}" class="grid"/>')
        g.append(f'<text x="{x:.1f}" y="{H-PAD["b"]+20}" class="xtick">{L}</text>')
    g.append(f'<text x="{W/2}" y="{H-PAD["b"]+46}" class="axlab">latent length L '
             f'&nbsp;(L×4096 samples ≈ L×0.093 s)</text>')
    g.append(f'<text transform="translate(22,{PAD["t"]+ih/2}) rotate(-90)" class="axlab">'
             f'{html.escape(ylab)}</text>')
    for s in SERIES:
        p=data[s]
        if not p: continue
        d=" ".join(("M" if i==0 else "L")+f"{X(L):.1f},{Y(v):.1f}" for i,(L,v) in enumerate(p))
        col=COLOR[(s[0],s[1])]; dash='6 4' if s[2] else 'none'
        g.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.4" '
                 f'stroke-dasharray="{dash}" stroke-linejoin="round"/>')
        for L,v in p:
            g.append(f'<circle cx="{X(L):.1f}" cy="{Y(v):.1f}" r="2.6" fill="{col}">'
                     f'<title>{label(*s)} — L={L}: {v:,.2f} {unit}</title></circle>')
    g.append("</svg>")
    leg="".join(
        f'<span class="key"><svg width="30" height="10"><line x1="1" y1="5" x2="29" y2="5" '
        f'stroke="{COLOR[(s[0],s[1])]}" stroke-width="2.6" '
        f'stroke-dasharray="{"6 4" if s[2] else "none"}"/></svg>{html.escape(label(*s))}</span>'
        for s in SERIES)
    return (f'<section><h2>{html.escape(title)}</h2>'
            + (f'<p class="note">{note}</p>' if note else "")
            + "".join(g) + f'<div class="legend">{leg}</div></section>')

CHARTS = [
  ("enc", "enc_scratch_mb", "Encoder — VRAM scratch reserved", "scratch (MB, log)", True, "MB",
   "TensorRT commits a context's scratch from its PROFILE CEILING, not from the shape you bind — "
   "so these are flat in L. That is the whole point of the low band. SAME-L's two precisions "
   "reserve identical scratch, so those lines overlap exactly."),
  ("enc", "enc_ms", "Encoder — latency", "milliseconds (log)", True, "ms",
   "Best of three, after warm-up. Single-shot lines stop at L=4096: that is the wide band's "
   "ceiling. Chunked keeps going because it windows."),
  ("enc", "enc_fix_db", "Encoder — accuracy vs the original music, content held fixed",
   "reconstruction SNR (dB)", False, "dB",
   "The SAME 190 s of real music at every point — reassembled from L-sized blocks, so only the "
   "processing length changes. This is the apples-to-apples question the TFLite rung table asks, "
   "and the answer is the same: flat. From L=16 up the whole curve moves 0.002 dB on SAME-L and "
   "0.020 dB on SAME-S. The fall below L≈16 is real but is not a seam: 0.09 s blocks encoded "
   "independently lose context at every boundary."),
  ("dec", "dec_scratch_mb", "Decoder — VRAM scratch reserved", "scratch (MB, log)", True, "MB", ""),
  ("dec", "dec_ms", "Decoder — latency", "milliseconds (log)", True, "ms", ""),
  ("dec", "dec_fix_db", "Decoder — accuracy vs the original music, content held fixed",
   "reconstruction SNR (dB)", False, "dB",
   "Same construction, decoder varying. Chunked and single-shot land on the same value to three "
   "decimals at every length, which is the direct evidence that the windowed overlap is exact."),
]
body = "".join(chart(k, t, y, ly, u, n) for _, k, t, y, ly, u, n in CHARTS)
n_ok = sum(1 for r in rows if "enc_ms" in r)
n_len = len({r['L'] for r in rows})
CSS = """
:root{--bg:#faf8f5;--fg:#1c1a17;--mut:#6b645c;--line:#e0dbd3;--card:#fff;--accent:#a8341b}
:root:not([data-theme="light"]){}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#14120f;--fg:#ece7df;--mut:#9a9086;--line:#2e2a25;--card:#1b1815;--accent:#e07b39}}
:root[data-theme="dark"]{--bg:#14120f;--fg:#ece7df;--mut:#9a9086;--line:#2e2a25;
  --card:#1b1815;--accent:#e07b39}
*{box-sizing:border-box}
body{margin:0;padding:38px 22px 90px;background:var(--bg);color:var(--fg);
  font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1240px;margin:0 auto}
h1{font-size:1.75rem;margin:0 0 6px;letter-spacing:-.02em;text-wrap:balance}
.sub{color:var(--mut);margin:0 0 30px;max-width:78ch}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:20px 22px 16px;margin:0 0 26px;overflow-x:auto}
h2{font-size:1.06rem;margin:0 0 4px;letter-spacing:-.01em}
.note{color:var(--mut);font-size:.86rem;margin:0 0 12px;max-width:92ch}
.chart{width:100%;height:auto;min-width:820px;display:block}
.grid{stroke:var(--line);stroke-width:1}
.ytick,.xtick{fill:var(--mut);font-size:12px;font-variant-numeric:tabular-nums}
.ytick{text-anchor:end}.xtick{text-anchor:middle}
.axlab{fill:var(--mut);font-size:12.5px;text-anchor:middle}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:12px;
  border-top:1px solid var(--line);padding-top:12px}
.key{display:flex;align-items:center;gap:7px;font-size:.83rem;color:var(--mut)}
footer{color:var(--mut);font-size:.85rem;max-width:88ch;margin-top:34px}
code{background:rgba(128,128,128,.14);padding:1px 5px;border-radius:4px;font-size:.9em}
"""
open(OUT, "w").write(f"""<meta charset="utf-8">
<title>SAME Autoencoder Sweep</title>
<style>{CSS}</style>
<main>
<h1>SAME autoencoder: VRAM, latency, accuracy</h1>
<p class="sub">Eight configurations — {{SAME-S, SAME-L}} × {{16-bit, fp8}} × {{chunked,
single-shot}} — measured across {n_len} latent lengths from
L=1 to L=8192 ({n_ok} points). Accuracy is a full round trip against <em>real music</em> — seven
mastered tracks from the 806k-track corpus (rock, jazz, classical, electronic, hip hop, piano,
folk) concatenated to 762&nbsp;s, because L=8192 is longer than any single song and tiling one
track would be artificially self-similar. Never noise: the autoencoder is content-dependent and
noise fabricates numbers that do not hold on music. 16-bit means fp16 on SAME-L and bf16 on
SAME-S. Hover any point for its value.</p>
{body}
<footer>
<p><strong>How accuracy is attributed.</strong> A round trip is a property of the encoder and
decoder together, so to read one stage the other is pinned to that model's 16-bit chunked
engine. The encoder chart varies the encoder; the decoder chart varies the decoder.</p>
<p><strong>Read these as comparisons, not verdicts.</strong> Compare lines to each other; do
not read the absolute height as a quality statement, and do not let any of it substitute for
listening.</p>
<p><strong>It is not the limiter.</strong> The decoders bake a 0.977 ceiling and real masters
here peak at 1.019, so level clipping was the obvious suspect. It is not: re-measured with the
input scaled from 0.9 down to 0.225 the SNR moves by 0.03&nbsp;dB, and an optimal scalar
gain-match recovers only 0.27&nbsp;dB.</p>
<p><strong>Why accuracy is plotted with content held fixed.</strong> The first version of
these charts measured one excerpt per length — the first L latents of the stream — and the lines
bounced by several dB, which looked like a length-dependent defect. It was not. At a fixed L,
moving the excerpt swings SNR by <strong>9.4&ndash;15.0&nbsp;dB</strong>, while the entire L axis
at a fixed offset moved only <strong>2.95&nbsp;dB</strong>: a longer L is simply a different,
longer piece of music, so one-excerpt-per-L plots the music rather than the model. The charts
above therefore hold the audio identical at every point and vary only the processing length —
the same thing the TFLite rung table does, and it comes out just as flat.</p>
<p><strong>The absolute level.</strong> Round-trip SNR on this real-music set is 4.2&ndash;4.9 dB
for SAME-L and 3.0&ndash;3.9 dB for SAME-S. Measured on our own generated audio the same pipeline
scores ~16 dB, and published figures for this family span roughly &minus;23 dB (solo piano) to
&minus;4 dB (dense circus). Waveform SNR systematically understates a perceptual autoencoder.</p>
<p><strong>SAME-S draw-to-draw noise.</strong> The shipped SAME-S decoder has a
<code>RandomNormalLike</code> at its bottleneck, so its round trip carries run-to-run variance
that belongs to neither precision nor chunking.</p>
</footer>
</main>""")
print(f"  wrote {OUT}  ({len(rows)} rows)")
