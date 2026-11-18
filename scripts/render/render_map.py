# -*- coding: utf-8 -*-
"""
지도 렌더러 — 장면 명세 → Before/After 비교 지도, 단일 지도

■ Before / After 를 나란히 놓는 이유
  "우리 경로가 더 안전하다"는 말로는 전달되지 않는다.
  같은 출발점에서 두 선이 갈라지는 그림 한 장이면 설명이 필요 없다.
  왼쪽은 지금의 내비게이션이 안내하는 길, 오른쪽은 위험을 아는 길이다.

■ Before 에는 위험 정보를 그리지 않는다
  이게 핵심이다. Before 패널에 침수 구역을 그려 넣으면
  "왜 저길 지나가지?"가 되어 버린다. Before 는 위험을 **모르는** 화면이어야
  "기존 지도는 이걸 모른다"가 성립한다.

사용:
    python render_map.py scenes/underpass-busan_choryang.json
    python render_map.py <scene.json> --mode single --out out/
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from svgkit import (SVG, T, Projection, draw_basemap, draw_flood, draw_route,  # noqa: E402
                    marker_pin, marker_here, map_label, scale_bar, legend,
                    source_note, mockup_badge, card, chip, export_png)

sys.stdout.reconfigure(encoding="utf-8")

HEADER_H = 168
FOOTER_H = 132


# ════════════════════════════════════════════════════════════════════
# 지도 패널
# ════════════════════════════════════════════════════════════════════

def _basemap_layers(scene):
    """장면 명세의 정규화 레이어를 svgkit 이 먹는 형태로 되돌린다."""
    L = scene["layers"]
    return {
        "roads": L["roads"],
        "buildings": L["buildings"],
        "rail": L["rail"],
        "water": [{"pts": w["pts"],
                   "tags": {"natural": "water" if w.get("area") else None,
                            "waterway": "river" if w.get("big") else "stream"}}
                  for w in L["water"]],
    }


def draw_panel(s, scene, x, y, w, h, show_risk=True, theme="dark", scale_xy=None):
    """
    지도 패널 하나. show_risk=False 면 위험 레이어를 숨긴다(Before용).
    반환: (proj, clip_id)
    """
    cid = s.clip_rect(x, y, w, h, rx=T.R["card"])
    s.raw(f'<g clip-path="url(#{cid})">')
    s.rect(x, y, w, h, fill=T.pal(theme)["map_land"])

    # 패널 안 좌표계를 쓰기 위해 이동시킨다
    s.raw(f'<g transform="translate({x},{y})">')
    proj = Projection(scene["bbox"], w, h)
    draw_basemap(s, proj, _basemap_layers(scene), theme=theme, detail=1.0)

    if show_risk:
        if scene.get("flood_polys"):
            draw_flood(s, proj, scene["flood_polys"], op=0.5)
        for seg in scene.get("underpass_segs", []):
            s.poly(proj.pts(seg), stroke=T.UNDERPASS, sw=6, op=0.95)

    # 경로 — Before 는 위험경로만, After 는 둘 다
    routes = scene.get("routes", [])
    if show_risk:
        for r in routes:
            if r["role"] == "risk":
                draw_route(s, proj, r["coords"], T.ROUTE_RISK, sw=6, op=0.55)
        for r in routes:
            if r["role"] == "safe":
                draw_route(s, proj, r["coords"], T.ROUTE_SAFE, sw=7)
    else:
        for r in routes:
            if r["role"] == "risk":
                draw_route(s, proj, r["coords"], T.ROUTE_SAFE, sw=7)

    # 마커
    for m in scene.get("markers", []):
        mx, my = proj(m["lat"], m["lon"])
        if not (0 <= mx <= w and 0 <= my <= h):
            continue
        k = m["kind"]
        if k == "here":
            marker_here(s, mx, my)
        elif k == "incident":
            marker_pin(s, mx, my, T.DANGER, "!", r=15)
        elif k == "dest":
            marker_pin(s, mx, my, T.SAFE, "⚑", r=13)
        elif k == "shelter" and show_risk:
            marker_pin(s, mx, my, T.SAFE, None, r=8)
        elif k == "blocked" and show_risk:
            s.line(mx - 6, my - 6, mx + 6, my + 6, T.BLOCKED, 2.6)
            s.line(mx + 6, my - 6, mx - 6, my + 6, T.BLOCKED, 2.6)
        if m.get("label") and k in ("incident", "dest"):
            map_label(s, mx, my + 32, m["label"], theme=theme, size=14)

    sx, sy = scale_xy or (28, h - 28)
    scale_bar(s, proj, sx, sy, theme=theme)
    s.endgroup()
    s.endgroup()
    s.rect(x, y, w, h, fill="none", rx=T.R["card"],
           stroke=T.pal(theme)["hairline"], sw=1)
    return proj


def panel_tag(s, x, y, w, text, sub, tone, theme="dark"):
    """패널 상단 라벨. Before/After 를 글자로도 못 박는다."""
    p = T.pal(theme)
    s.rect(x, y, w, 52, fill=p["surface"], rx=T.R["control"], op=0.92)
    s.rect(x, y, 4, 52, fill=tone, rx=2)
    s.text(x + 20, y + 23, text, size=15, weight=700, fill=tone, tracking=0.4)
    s.text(x + 20, y + 42, sub, size=13, weight=400, fill=p["fg2"])
    return s


# ════════════════════════════════════════════════════════════════════
# 아트보드
# ════════════════════════════════════════════════════════════════════

def render_before_after(scene, out_dir="out", png=True):
    """1920×1080 슬라이드 — 좌 Before / 우 After."""
    theme = scene.get("theme", "dark")
    W, H = T.SLIDE
    s = SVG(W, H, theme=theme)
    p = T.pal(theme)

    s.style(64, 78, scene["title"], "hero", scale="S", fill=p["fg"],
            family=T.FONT_DISPLAY)
    s.style(64, 122, scene["subtitle"], "tile", scale="S", fill=p["fg2"])

    gap, mx = 32, 64
    pw = (W - mx * 2 - gap) / 2
    py = HEADER_H + 62
    ph = H - py - FOOTER_H

    panel_tag(s, mx, HEADER_H, pw, "BEFORE",
              "기존 내비게이션 — 최단거리만 계산한다", T.ROUTE_RISK, theme)
    panel_tag(s, mx + pw + gap, HEADER_H, pw, "AFTER",
              "SAFE-NOTING — 위험을 알고 우회한다", T.ROUTE_SAFE, theme)

    draw_panel(s, scene, mx, py, pw, ph, show_risk=False, theme=theme)
    draw_panel(s, scene, mx + pw + gap, py, pw, ph, show_risk=True, theme=theme)

    # 범례 — After 패널 안 좌하단
    items = []
    if any(r["role"] == "risk" for r in scene.get("routes", [])):
        items.append((T.ROUTE_RISK, "최단경로 · 위험구간 통과", "line"))
        items.append((T.ROUTE_SAFE, "안전경로 · 회피", "line"))
    if scene.get("flood_polys"):
        items.append((T.DANGER, "침수 이력 구역", "area"))
    if scene.get("underpass_segs"):
        items.append((T.UNDERPASS, "지하차도 · 침수 시 차단", "line"))
    if any(m["kind"] == "shelter" for m in scene.get("markers", [])):
        items.append((T.SAFE, "이용 가능 대피소", "dot"))
        items.append((T.BLOCKED, "부적합 대피소", "dot"))
    if items:
        # 범례는 패널 좌상단에 둔다. 좌하단은 축척 막대 자리라 겹친다.
        lx = mx + pw + gap + 26
        ly = py + 34
        s.rect(lx - 16, ly - 26, 306, len(items) * 26 + 30,
               fill=p["surface"], rx=T.R["control"], op=0.92)
        legend(s, lx, ly, items, theme=theme)

    _facts_strip(s, scene, mx, H - FOOTER_H + 4, W - mx * 2, theme)
    s.line(mx, H - 52, W - mx, H - 52, p["hairline"], 1, cap="butt", op=0.7)
    source_note(s, "출처 · " + " | ".join(scene.get("sources", [])),
                x=mx, y=H - 26, theme=theme, size=13)
    mockup_badge(s, x=W - mx - 168, y=H - 42, label="실데이터 기반 시각화", theme=theme)
    return _save(s, scene["id"] + "-before-after", out_dir, png, W, H)


def render_single(scene, out_dir="out", png=True,
                  show_risk=True, suffix="map"):
    """1920×1080 단일 전면 지도. 슬라이드 배경으로 쓰기 좋다."""
    theme = scene.get("theme", "dark")
    W, H = T.SLIDE
    s = SVG(W, H, theme=theme)
    p = T.pal(theme)
    # 축척 막대는 우하단에 둔다. 좌하단은 출처 문구 자리라 겹친다.
    draw_panel(s, scene, 0, 0, W, H, show_risk=show_risk, theme=theme,
               scale_xy=(W - 200, H - 34))

    grad = s.lingrad([(0, "#000000", 0.78), (0.55, "#000000", 0.30),
                      (1, "#000000", 0)], 0, 0, 0, 1)
    s.rect(0, 0, W, 320, fill=f"url(#{grad})")
    s.style(64, 96, scene["title"], "section", scale="S", fill="#ffffff",
            family=T.FONT_DISPLAY)
    s.style(64, 142, scene["subtitle"], "body", scale="S", fill="#d2d2d7")

    g2 = s.lingrad([(0, "#000000", 0), (1, "#000000", 0.82)], 0, 0, 0, 1)
    s.rect(0, H - 210, W, 210, fill=f"url(#{g2})")
    _facts_strip(s, scene, 64, H - 148, W - 128, theme, on_dark=True)

    items = []
    if show_risk and scene.get("flood_polys"):
        items.append((T.DANGER, "침수 이력 구역 (침수흔적도)", "area"))
    if show_risk and scene.get("underpass_segs"):
        items.append((T.UNDERPASS, "지하차도 구간 (OSM)", "line"))
    if items:
        lx, ly = W - 380, 70
        s.rect(lx - 16, ly - 26, 330, len(items) * 26 + 30,
               fill=p["surface"], rx=T.R["control"], op=0.92)
        legend(s, lx, ly, items, theme=theme)
    source_note(s, "출처 · " + " | ".join(scene.get("sources", [])),
                x=64, y=H - 34, theme=theme, size=13)
    return _save(s, f"{scene['id']}-{suffix}", out_dir, png, W, H)


def _facts_strip(s, scene, x, y, w, theme, on_dark=False):
    """핵심 수치 띠. 지도만 있으면 '그래서 얼마나?'에 답하지 못한다."""
    facts = scene.get("facts", [])
    if not facts:
        return
    p = T.pal(theme)
    cw = min(240, w / max(1, len(facts)))
    tone_c = {"risk": T.DANGER, "safe": T.SAFE, "neutral": p["fg"]}
    for i, f in enumerate(facts):
        fx = x + i * cw
        c = tone_c.get(f.get("tone"), p["fg"])
        s.text(fx, y + 18, f["k"], size=13, weight=500,
               fill="#aeaeb2" if on_dark else p["muted"])
        s.text(fx, y + 50, f["v"], size=30, weight=700, fill=c,
               family=T.FONT_DISPLAY, tracking=-0.4)


def _save(s, name, out_dir, png, W, H):
    os.makedirs(out_dir, exist_ok=True)
    svg = os.path.join(out_dir, name + ".svg")
    s.save(svg)
    print(f"  → {svg}  ({os.path.getsize(svg)//1024}KB)")
    if png:
        p = export_png(svg, width=W, height=H, scale=1)
        if p:
            print(f"  → {p}  ({os.path.getsize(p)//1024}KB)")
    return svg


MODES = {"before-after": render_before_after, "single": render_single}


def main():
    a = sys.argv[1:]
    if not a:
        raise SystemExit("사용: render_map.py <scene.json> [--mode before-after|single] "
                         "[--out DIR] [--no-png]")
    scene_path = a[0]
    mode = a[a.index("--mode") + 1] if "--mode" in a else "before-after"
    out = a[a.index("--out") + 1] if "--out" in a else "out"
    png = "--no-png" not in a
    with open(scene_path, encoding="utf-8") as f:
        scene = json.load(f)
    if not scene.get("coverage_ok", True):
        print("  ⚠ 이 장면은 커버리지 경고가 있다. 배경 지도가 빌 수 있다.")
        for pr in scene.get("coverage_problems", []):
            print(f"    {pr}")
    print(f"■ 렌더: {scene['id']}  mode={mode}")
    MODES[mode](scene, out, png)


if __name__ == "__main__":
    main()
