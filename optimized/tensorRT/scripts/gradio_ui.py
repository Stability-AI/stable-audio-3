"""Backend-agnostic gradio UI helpers, copied from optimized/mlx/scripts/sa3_gradio.py.

Forked rather than shared: the MLX and TensorRT trees install independently, with their own
requirements, and a common module would couple two deployment units that today have no reason to
know about each other. Nothing in here touches a backend -- these are player/history/queue
renderers and filename helpers -- so the two copies only need to be reconciled when the UI
itself changes.
"""
from __future__ import annotations
import base64
import html as html_lib
import math
import os
import re
import time
import urllib.parse
import wave
from pathlib import Path

import numpy as np

_JS_FIX_SLIDERS = (
    "() => setTimeout(() => {"
    "document.querySelectorAll('input[type=range]').forEach(r => {"
    "const min = +r.min || 0, max = +r.max || 100, v = +r.value;"
    "if (max > min) r.style.setProperty('--range_progress',"
    "((v - min) / (max - min) * 100) + '%');"
    "});}, 150)"
)

_JS_PAUSE_OTHERS = (
    "if(this.isConnected){"
    "var t=this;var L=window._sa3All=(window._sa3All||[]);"
    "if(L.indexOf(t)<0)L.push(t);"
    "L.forEach(function(o){if(o!==t){try{o.pause()}catch(e){}}});"
    "document.querySelectorAll('audio').forEach(function(o){if(o!==t)o.pause();});"
    "window._sa3All=L.filter(function(o){return o===t||o.isConnected;});}"
    "else{this.pause();}")

_JS_PLAYHEAD = ("var p=this.closest('.blk').querySelector('.ph');"
                "if(p&&this.duration)p.style.left=(this.currentTime/this.duration*100)+'%';")
# Main-player position ledger (for Hotswap): every timeupdate/play/pause stamps
# the current position so a freshly swapped-in element can resume exactly there.
# Guards (all verified via CDP against the live page):
#  - isConnected: teardown fires a pause AND a final timeupdate on the removed
#    element — detached elements never write the ledger.
#  - unresumed hotswap candidates (data-hs set, hsd not yet) don't write either:
#    with autoplay, the NEW element's 'play' event fires BEFORE loadedmetadata
#    and would stamp t≈0 over the old clip's position right before the resume
#    handler reads it.

_JS_POS_RECORD = ("if(this.isConnected&&!(this.dataset.hs==='1'&&!this.dataset.hsd))"
                  "{window._sa3Pos={t:this.currentTime,playing:!this.paused,ts:Date.now()};}")
# timeupdate variant: a PAUSED element only fires timeupdate on seeks — e.g. the
# resume handler's own currentTime assignment. Stamping playing:false there
# poisons the ledger for any second render of the same clip (gradio sometimes
# renders a component update twice), which froze the handoff chain. While
# paused, only real pause events may write.

_JS_POS_RECORD_TU = ("if(this.isConnected&&!this.paused&&"
                     "!(this.dataset.hs==='1'&&!this.dataset.hsd))"
                     "{window._sa3Pos={t:this.currentTime,playing:true,ts:Date.now()};}")
# Resilient play: Chrome's autoplay policy can reject programmatic play() once
# the transient user activation from the Generate click has expired (seconds),
# leaving the clip correctly positioned but paused. Try, retry shortly after,
# and as a last resort arm a one-shot listener so the user's next click or
# keypress anywhere resumes playback.
# Every attempt re-checks isConnected: a superseded (detached) element must
# never revive itself from a queued retry and fight the current player.

_JS_TRY_PLAY = (
    "var A=this;A.play().catch(function(){setTimeout(function(){"
    "if(!A.isConnected)return;"
    "A.play().catch(function(){console.warn('play blocked by autoplay policy — "
    "will resume on next interaction');"
    "var f=function(){if(A.isConnected)A.play();"
    "document.removeEventListener('pointerdown',f,true);"
    "document.removeEventListener('keydown',f,true);};"
    "document.addEventListener('pointerdown',f,true);"
    "document.addEventListener('keydown',f,true);});},150);});")
# Hotswap resume: if the previous main audio was playing when this one arrived,
# jump to its position (+ the split-second since the last stamp) and keep going;
# beyond the new clip's duration -> start at zero. Guarded (hsd) so it applies
# once, on whichever of loadedmetadata/canplay fires first.

_JS_SEEK = ("var a=this.closest('.blk').querySelector('audio');"
            "var r=this.getBoundingClientRect();"
            "if(a&&a.duration){a.currentTime=(event.clientX-r.left)/r.width*a.duration;a.play();}")

_JS_PROMOTE = ("var b=document.getElementById('sa3-promote');"
               "if(b){(b.tagName==='BUTTON'?b:b.querySelector('button')||b).click();}")
# Scroll anchoring for the history panel: onscroll continuously records which
# item sits at the viewport top (+offset); after every re-render a hidden
# bootstrap <img onerror> restores that anchor — stick-to-top when at top,
# otherwise keep hovering over the same old item as new ones prepend.

_JS_SCROLL_RECORD = (
    "var c=this,k=null,off=0,ch=c.querySelectorAll('[data-key]');"
    "for(var i=0;i<ch.length;i++){if(ch[i].offsetTop+ch[i].offsetHeight>c.scrollTop)"
    "{k=ch[i].getAttribute('data-key');off=ch[i].offsetTop-c.scrollTop;break}}"
    "window._sa3S={atTop:c.scrollTop<8,key:k,off:off};")

_JS_SCROLL_RESTORE = (
    "var c=document.getElementById('sa3-hist');var s=window._sa3S||{};"
    "if(c){var el=s.key?c.querySelector('[data-key=&quot;'+s.key+'&quot;]'):null;"
    "if(el&&!s.atTop){c.scrollTop=el.offsetTop-(s.off||0)}else{c.scrollTop=0}}"
    "this.remove();")


def _ago(ts) -> str:
    if not ts:
        return ""
    d = max(0.0, time.time() - ts)
    if d < 10:
        return "just now"
    if d < 60:
        return f"{int(d)}s ago"
    if d < 3600:
        return f"{int(d // 60)}m ago"
    if d < 86400:
        return f"{int(d // 3600)}h ago"
    return f"{int(d // 86400)}d ago"

def condense_prompt(prompt: str) -> str:
    """Prompt → filename fragment (the main repo's verbose-naming rule):
    filesystem-special characters become hyphens, capped at 150 chars."""
    prompt = re.sub(r'[\\/:*?"<>|]', '-', prompt)[:150]
    return prompt or "_"

def verbose_basename(prompt, negative_prompt, cfg, sigma_max, seed) -> str:
    """prompt[.neg-…].cfg{scale}[.smx{σ}].{seed} — matches the main repo's
    gradio 'verbose' file naming (cfg segment only when cfg != 1)."""
    base = condense_prompt(prompt)
    if negative_prompt and negative_prompt.strip():
        base += ".neg-" + condense_prompt(negative_prompt.strip())
    if cfg != 1.0:
        base += f".cfg{cfg:g}"
    if sigma_max != 1.0:
        base += f".smx{sigma_max:g}"
    return f"{base}.{seed}"

def _save_wav(pcm_int16, out_path):
    """pcm_int16: (T, 2) int16 interleaved."""
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm_int16.tobytes())


# ── Model caches (unified memory; ~0.9-2.8 GB per DiT, 0.2-1.7 GB per codec) ──
# The MLX DiT bakes RoPE/mask lengths at load, so the DiT cache key includes
# T_lat. Everything else is length-independent and cached by name.
_t5: T5Gemma | None = None
_dit_lru: list[tuple[str, int]] = []

def _meta_suffix(entry) -> str:
    """' · cfg 1.5 · noise 0.92 · audio2audio · inpainting' — only the non-defaults.
    The sigma tag reads 'noise' for a2a runs (it IS the init_noise_level there),
    'smx' otherwise."""
    parts = []
    cfg = entry.get("cfg", 1.0)
    smx = entry.get("smx", 1.0)
    mode = entry.get("mode", "")
    is_a2a = "a2a" in mode or mode == "audio-to-audio"
    dit = entry.get("dit", "")
    if dit:
        parts.append({"medium": "med", "sm-music": "sm-mus", "sm-sfx": "sm-sfx"}.get(dit, dit))
    if cfg != 1.0:
        parts.append(f"cfg {cfg:g}")
    if smx != 1.0:
        parts.append(f"{'noise' if is_a2a else 'smx'} {smx:g}")
    if is_a2a:
        parts.append("audio2audio")
    if "inpaint" in mode:
        parts.append("inpainting")
    if entry.get("lora"):
        parts.append(f"lora {entry['lora']}")
    return "".join(f" · {p}" for p in parts)

def _neg_disp(entry) -> str:
    """Labeled negative prompt, shown only when it actually acted (cfg != 1)."""
    neg = (entry.get("neg") or "").strip()
    if neg and entry.get("cfg", 1.0) != 1.0:
        return f' · <span style="opacity:0.75">neg: {html_lib.escape(neg)}</span>'
    return ""

def render_player(entry, *, small=False, autoplay=False, autodl=False, radio=False,
                  bg=None, advance=False, loop=False, hotswap=False):
    """One self-contained player block: audio + caption + seekable spectrogram
    with playhead. Global one-at-a-time playback via onplay pause-others.
    small: audio + spectrogram side by side (half width each), 'Xm ago' caption.
    advance: on ended, hop to the next history item's audio (Auto-play).
    loop: native loop attribute — finished audio restarts (suppresses ended).
    hotswap: main slot only — resume at the previous clip's position on arrival."""
    # assemble per-event handler bodies (a duplicated attribute name would
    # silently drop one handler, so each event is emitted exactly once)
    on_canplay = []
    if small:
        attrs = f'onplay="{_JS_PAUSE_OTHERS}" ontimeupdate="{_JS_PLAYHEAD}"'
    else:
        attrs = (f'onplay="{_JS_PAUSE_OTHERS}{_JS_POS_RECORD}" '
                 f'onpause="{_JS_POS_RECORD}" '
                 f'ontimeupdate="{_JS_PLAYHEAD}{_JS_POS_RECORD_TU}"')
    if hotswap and not small:
        attrs += f' data-hs="1" onloadedmetadata="{_JS_HOTSWAP}"'
        on_canplay.append(_JS_HOTSWAP)   # fallback if loadedmetadata already passed
    if autoplay and not small:
        # the autoplay attribute is subject to the same policy — rescue a
        # policy-paused clip (radio transitions, long generations)
        on_canplay.append("if(this.paused&&!(this.dataset.hs==='1'&&!this.dataset.hsd))"
                          "{" + _JS_TRY_PLAY + "}")
    if autodl:
        js_name = entry["name"].replace("\\", "").replace("'", "\\'")
        on_canplay.append("if(!this.dataset.dld){this.dataset.dld=1;"
                          "var l=document.createElement('a');l.href=this.src;"
                          f"l.download='{js_name}';l.click();}}")
    if on_canplay:
        attrs += f' oncanplay="{"".join(on_canplay)}"'
    if loop:
        attrs += " loop"
    elif radio:
        attrs += f' onended="{_JS_PROMOTE}"'
    elif advance:
        attrs += (' onended="var b=this.closest(\'.blk\');'
                  "var n=b?b.nextElementSibling:null;"
                  "while(n&&!n.classList.contains('blk'))n=n.nextElementSibling;"
                  "if(n){var a=n.querySelector('audio');if(a)a.play();}\"")
    auto = "autoplay " if autoplay else ""
    # Serve audio via gradio's file route instead of a data: URI — the URL ends
    # with the real (verbose) filename, so right-click "Save audio as…" offers
    # it instead of download.mp3, and the page stays light as history grows.
    src = "gradio_api/file=" + urllib.parse.quote(entry["path"], safe="/")
    audio_el = (f'<audio controls {auto}style="width:100%" {attrs} '
                f'src="{src}"></audio>')
    prompt_disp = html_lib.escape(entry["prompt"]) or "<i>(no prompt)</i>"

    spec_core = ""
    if entry.get("spec_b64"):
        height = "height:56px;" if small else ""
        tip = ("3-band tinted stereo mel · red=bass / green=mid / blue=high · "
               "L top, R bottom · click to seek")
        spec_core = (f'<div style="position:relative; cursor:pointer" title="{tip}" '
                     f'onclick="{_JS_SEEK}">'
                     f'<img src="data:image/png;base64,{entry["spec_b64"]}" '
                     f'style="width:100%; {height} display:block; image-rendering:pixelated; '
                     f'border:1px solid #333" alt="spectrogram"/>'
                     f'<div class="ph" style="position:absolute; top:0; bottom:0; left:0%; '
                     f'width:2px; background:#fff; pointer-events:none; '
                     f'box-shadow:0 0 4px rgba(0,0,0,.8)"></div>'
                     f'</div>')

    if small:
        row = (f'<div style="display:flex; gap:8px; align-items:center">'
               f'<div style="flex:1; min-width:0">{audio_el}</div>'
               f'<div style="flex:1; min-width:0">{spec_core}</div></div>')
        cap = (f'<div style="font-size:0.8em; margin:2px 0; color:#888">'
               f'{_ago(entry.get("ts"))} · {prompt_disp}{_neg_disp(entry)} · seed {entry["seed"]}'
               f'{_meta_suffix(entry)}</div>')
        style = f"padding:6px 8px;{' background:' + bg + ';' if bg else ''}"
        return (f'<div class="blk" data-key="{entry["key"]}" style="{style}">'
                f'{row}{cap}</div>')

    return (f'<div class="blk" data-key="{entry["key"]}">'
            f'{audio_el}{spec_core}</div>')

def render_history(hist, advance=False, loop=False):
    if not hist:
        return ""
    # zebra striping instead of separators: light grey vs medium grey rows
    items = "".join(
        render_player(e, small=True, advance=advance, loop=loop,
                      bg="rgba(127,127,127,0.24)" if i % 2 else "rgba(127,127,127,0.08)")
        for i, e in enumerate(hist))
    boot = f'<img src="data:," style="display:none" onerror="{_JS_SCROLL_RESTORE}"/>'
    return (f'<div style="font-weight:600; margin-top:14px">Previous generations ({len(hist)})</div>'
            f'<div id="sa3-hist" onscroll="{_JS_SCROLL_RECORD}" '
            f'style="max-height:480px; overflow-y:auto; position:relative; margin-top:6px; '
            f'padding-right:6px">{boot}{items}</div>')

def render_queue_status(entry=None, generating=False):
    """Subdued status chip for the Infinite Radio queue — no audio/spectrogram,
    just Generating…/Ready (the swap uses the entry held in server state)."""
    if generating:
        body = "generating…"
    elif entry is not None:
        p = html_lib.escape(entry["prompt"]) or "<i>(no prompt)</i>"
        body = f"ready — {p}{_neg_disp(entry)} · seed {entry['seed']}{_meta_suffix(entry)}"
    else:
        return ""
    return (f'<div style="margin-top:2px; padding:6px 10px; '
            f'background:rgba(127,127,127,0.12); border-radius:6px; '
            f'color:#888; font-size:0.85em">'
            f'<b>Queued next</b> · {body}</div>')


# ── Gradio UI ──────────────────────────────────────────────────────────────
