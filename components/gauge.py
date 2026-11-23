"""3D dial for the prediction page: tube-shaded coloured bands, tapered needle with hub, big score.

Built as inline SVG so the needle, bevel and shadows are fully controlled (a Plotly gauge cannot draw them).
"""
import math

CX, CY, R_OUT, R_IN = 200, 205, 165, 112
LOW = ["#0A7F6E", "#0F9D8A", "#2DBFAE", "#7ADFD0"]      # deep teal (far left) to light teal (near cutoff)
HIGH = ["#FF9B85", "#F77A62", "#EA533C", "#C92A18"]     # light coral (near cutoff) to deep red (far right)


def _pt(r, t):
    a = math.pi * (1 - t)
    return CX + r * math.cos(a), CY - r * math.sin(a)


def _arc(t0, t1, r_out=R_OUT, r_in=R_IN, dy=0):
    x0, y0 = _pt(r_out, t0); x1, y1 = _pt(r_out, t1)
    x2, y2 = _pt(r_in, t1); x3, y3 = _pt(r_in, t0)
    return (f"M{x0:.1f},{y0 + dy:.1f} A{r_out},{r_out} 0 0 1 {x1:.1f},{y1 + dy:.1f} "
            f"L{x2:.1f},{y2 + dy:.1f} A{r_in},{r_in} 0 0 0 {x3:.1f},{y3 + dy:.1f} Z")


def gauge_svg(score, cutoff, is_cnn):
    if is_cnn:
        lo, hi, mid, fmt = 0.0, 1.0, cutoff, "{:.3f}"
        ticks = [0, .25, .5, .75, 1]
        title = "Uncalibrated model score"
    else:
        hi = max(2.0, min(10.0, abs(score) * 1.25)); lo, mid, fmt = -hi, 0.0, "{:.2f}"
        ticks = [lo, lo / 2, 0, hi / 2, hi]
        title = "SVM decision value"
    span = hi - lo
    t = lambda v: min(1.0, max(0.0, (v - lo) / span))
    tm = t(mid)
    signal = score >= mid

    parts = []
    # extrusion (depth) under the dial
    # coloured bands with thin gaps
    gap = 0.006
    for i, c in enumerate(LOW):
        a, b = tm * i / 4, tm * (i + 1) / 4
        parts.append(f'<path d="{_arc(a + gap * (i > 0), b)}" fill="{c}" stroke="#fff" stroke-width="2"/>')
    for i, c in enumerate(HIGH):
        a, b = tm + (1 - tm) * i / 4, tm + (1 - tm) * (i + 1) / 4
        parts.append(f'<path d="{_arc(a + gap, b)}" fill="{c}" stroke="#fff" stroke-width="2"/>')
    # tube shading and glass highlight
    # cutoff notch
    x0, y0 = _pt(R_OUT + 6, tm); x1, y1 = _pt(R_IN - 6, tm)
    parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#0B2E3F" stroke-width="3"/>')
    # tick labels
    for v in ticks:
        x, y = _pt(R_OUT + 20, t(v))
        parts.append(f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" class="tk">{v:g}</text>')
    # zone captions
    parts.append('<text x="62" y="248" text-anchor="middle" class="zone" fill="#0A7F6E">NO SIGNAL</text>')
    parts.append('<text x="338" y="248" text-anchor="middle" class="zone" fill="#C92A18">CATARACT SIGNAL</text>')

    rot = -90 + 180 * t(score)
    needle = (
        f'<g transform="translate({CX} {CY})"><g class="needle" style="--rot:{rot:.1f}deg">'
        f'<path d="M-9,0 L0,-{R_OUT - 14} L9,0 Z" fill="#0B2E3F"/>'
        f'</g><circle r="17" fill="#0B2E3F"/><circle r="6" fill="#fff"/></g>'
    )
    color = "#C92A18" if signal else "#0A7F6E"
    word = "CATARACT SIGNAL" if signal else "NO CATARACT SIGNAL"
    return f"""<div class="gauge3d"><style>
.gauge3d{{text-align:center}} .gauge3d svg{{width:100%;max-width:430px;height:auto}}
.gauge3d .tk{{font:600 12px Inter,sans-serif;fill:#0B2E3F}} .gauge3d .zone{{font:800 11px 'Plus Jakarta Sans',Inter,sans-serif;letter-spacing:.08em}}
.gauge3d .needle{{animation:sweep 1.1s cubic-bezier(.2,.9,.25,1.15) both;transform:rotate({rot:.1f}deg)}}
@keyframes sweep{{from{{transform:rotate(-90deg)}}to{{transform:rotate({rot:.1f}deg)}}}}
.gauge3d .gt{{font:700 13px 'Plus Jakarta Sans',Inter,sans-serif;fill:#5B7480}}
.gauge3d .gs{{font:800 46px 'Plus Jakarta Sans',Inter,sans-serif}} .gauge3d .gv{{font:800 12px 'Plus Jakarta Sans',Inter,sans-serif;letter-spacing:.1em}}
</style><svg viewBox="0 0 400 330" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{title}: {fmt.format(score)}">

{''.join(parts)}
{needle}
<text x="200" y="272" text-anchor="middle" class="gs" fill="{color}">{fmt.format(score)}</text>
<text x="200" y="294" text-anchor="middle" class="gv" fill="{color}">{word}</text>
<text x="200" y="316" text-anchor="middle" class="gt">{title} · cutoff {fmt.format(mid)}</text>
</svg></div>""".replace("\n", "")
