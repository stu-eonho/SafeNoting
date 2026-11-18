# -*- coding: utf-8 -*-
"""
앱 화면 시안 렌더러 — 장면 명세 → iPhone 아트보드 + 슬라이드 배치

■ 절대 규칙: 숫자를 지어내지 않는다
  화면에 뜨는 수위·강수·거리·시간은 전부 scene["readings"] 에서 온다.
  readings 에 없으면 "—" 를 그리고 넘어간다. 렌더러가 그럴듯한 값을
  채워 넣으면 수치의 출처를 확인할 수 없게 된다.
  (청주 장면에 미호강 수위를 적어 넣는 사고가 실제로 났다)

■ 설계 원칙 (Apple HIG)
  · 콘텐츠가 주인공이다. 지도가 화면을 채우고 컨트롤은 그 위에 얇게 뜬다.
  · 한 화면에 채워진 파란 버튼은 하나. 두 개면 무엇이 주 행동인지 사라진다.
  · 색만으로 위험을 전달하지 않는다. 색 + 아이콘 + 문장을 함께 준다
    (색각 이상 사용자에게 빨강과 초록은 같은 색이다).
  · 문장은 다음에 일어날 일을 그대로 말한다. "확인"이 아니라 "안내 시작".

■ 화면
  msg      재난문자 수신 — 90자가 답하지 못하는 질문들
  guide    행동 안내 — 경로를 찾기 전에 행동을 먼저 결정한다
  riskmap  동적 침수위험 지도 — 현재 위치에서 위험을 한눈에
  route    안전경로 안내 — 어디로 어떻게
  reroute  실시간 재탐색 — 재난은 움직인다

사용:
    python scripts/render/render_ui.py scenes/app-osong.json
    python render_ui.py <scene.json> --screens riskmap,route
    python render_ui.py <scene.json> --mode slide --screens msg,riskmap,route
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from svgkit import (SVG, T, Projection, draw_basemap, draw_flood, draw_route,  # noqa: E402
                    marker_pin, marker_here, status_bar, home_indicator,
                    tab_bar, sheet, card, chip, pill, alert_banner,
                    source_note, mockup_badge, export_png)

sys.stdout.reconfigure(encoding="utf-8")

W, H = T.PHONE
TAB_H = 83
TABS = [("◉", "지도"), ("⚑", "대피소"), ("◔", "예보"), ("☰", "더보기")]


# ════════════════════════════════════════════════════════════════════
# readings 접근 — 없으면 지어내지 않고 비운다
# ════════════════════════════════════════════════════════════════════

def R(scene, key, default="—", fmt=None):
    v = (scene.get("readings") or {}).get(key)
    if v is None or v == "":
        return default
    return fmt.format(v) if fmt else v


def has(scene, *keys):
    r = scene.get("readings") or {}
    return all(r.get(k) not in (None, "") for k in keys)


# ════════════════════════════════════════════════════════════════════
# 지도
# ════════════════════════════════════════════════════════════════════

def _layers(scene):
    L = scene["layers"]
    return {"roads": L["roads"], "buildings": L["buildings"], "rail": L["rail"],
            "water": [{"pts": w["pts"],
                       "tags": {"natural": "water" if w.get("area") else None,
                                "waterway": "river" if w.get("big") else "stream"}}
                      for w in L["water"]]}


def fullscreen_map(s, scene, show_risk=True, show_route=True, theme="light",
                   zoom=1.0, h=None):
    """
    화면 전체를 채우는 지도. 앱에서 지도는 배경이 아니라 콘텐츠다.
    반환: proj (마커 위치를 다시 쓰기 위해)
    """
    h = h or s.h
    bbox = list(scene["bbox"])
    if zoom != 1.0:
        la0, la1, lo0, lo1 = bbox
        cy, cx = (la0 + la1) / 2, (lo0 + lo1) / 2
        bbox = [cy - (la1 - la0) / 2 / zoom, cy + (la1 - la0) / 2 / zoom,
                cx - (lo1 - lo0) / 2 / zoom, cx + (lo1 - lo0) / 2 / zoom]
    proj = Projection(bbox, s.w, h)
    s.rect(0, 0, s.w, h, fill=T.pal(theme)["map_land"])
    draw_basemap(s, proj, _layers(scene), theme=theme, detail=0.9)

    if show_risk:
        if scene.get("flood_polys"):
            draw_flood(s, proj, scene["flood_polys"], op=0.40)
        # ⚠ 지하차도 세그먼트는 이름 매칭으로 잡히기 때문에 수십 개가 나온다.
        #   전부 칠하면 노란 도로망처럼 보여 지도가 지도가 아니게 된다.
        #   화면 안에 들어오는 것만, 얇게 그린다.
        shown = 0
        for seg in scene.get("underpass_segs", []):
            q = proj.pts(seg)
            if not any(0 <= a <= s.w and 0 <= b <= h for a, b in q):
                continue
            s.poly(q, stroke=T.UNDERPASS, sw=3, op=0.7)
            shown += 1
            if shown >= 14:
                break

    if show_route:
        for r in scene.get("routes", []):
            if r["role"] == "risk":
                draw_route(s, proj, r["coords"], T.ROUTE_RISK, sw=5, op=0.45,
                           dash="10 7")
        for r in scene.get("routes", []):
            if r["role"] == "safe":
                draw_route(s, proj, r["coords"], T.ROUTE_SAFE, sw=7)

    for m in scene.get("markers", []):
        mx, my = proj(m["lat"], m["lon"])
        if not (-24 <= mx <= s.w + 24 and -24 <= my <= h + 24):
            continue
        if m["kind"] == "shelter" and show_risk:
            marker_pin(s, mx, my, T.SAFE, None, r=9)
        elif m["kind"] == "blocked" and show_risk:
            s.line(mx - 5, my - 5, mx + 5, my + 5, T.BLOCKED, 2.4, op=0.9)
            s.line(mx + 5, my - 5, mx - 5, my + 5, T.BLOCKED, 2.4, op=0.9)

    # 현재 위치는 항상 마지막에, 항상 화면 안에 그린다.
    # 이 점이 없으면 사용자는 화면의 기준점을 잃는다.
    hm = next((m for m in scene.get("markers", []) if m["kind"] == "here"), None)
    if hm:
        hx, hy = proj(hm["lat"], hm["lon"])
        hx = min(max(hx, 30), s.w - 30)
        hy = min(max(hy, 30), h - 30)
        marker_here(s, hx, hy)
    return proj


def top_scrim(s, h=150):
    g = s.lingrad([(0, "#000000", 0.32), (1, "#000000", 0)], 0, 0, 0, 1)
    s.rect(0, 0, s.w, h, fill=f"url(#{g})")


# ════════════════════════════════════════════════════════════════════
# 조각
# ════════════════════════════════════════════════════════════════════

def risk_level_card(s, x, y, w, level, detail, color):
    card(s, x, y, w, 78)
    s.rect(x, y, 4, 78, fill=color, rx=2)
    s.text(x + 20, y + 28, "현재 위치 위험단계", size=12, weight=500,
           fill=s.p["muted"])
    s.text(x + 20, y + 58, level, size=24, weight=700, fill=color,
           family=T.FONT_DISPLAY, tracking=-0.4)
    s.text(x + w - 20, y + 58, detail, size=13, weight=500,
           fill=s.p["fg2"], anchor="end")
    return 78


def stat_row(s, x, y, w, items):
    cw = w / len(items)
    for i, (k, v, unit, tone) in enumerate(items):
        cx = x + i * cw
        s.text(cx, y, k, size=11, weight=500, fill=s.p["muted"])
        s.text(cx, y + 26, str(v), size=21, weight=700, fill=tone or s.p["fg"],
               family=T.FONT_NUM)
        s.text(cx + len(str(v)) * 12 + 5, y + 26, unit, size=11, weight=500,
               fill=s.p["muted"])
        if i:
            s.line(cx - 9, y - 12, cx - 9, y + 12, s.p["hairline"], 1)


def _chars_per_line(width, size):
    return max(6, int(width / (size * 0.98)))


def _wrap_lines(text, width, size):
    n = _chars_per_line(width, size)
    out, line = [], ""
    for ch in text:
        line += ch
        if len(line) >= n and ch in " ,.·])":
            out.append(line.strip())
            line = ""
        elif len(line) >= n + 4:
            out.append(line.strip())
            line = ""
    if line.strip():
        out.append(line.strip())
    return out


def _wrap(s, text, x, y, width, size=15, lh=24, fill=None, weight=400):
    lines = _wrap_lines(text, width, size)
    for i, ln in enumerate(lines):
        s.text(x, y + i * lh, ln, size=size, weight=weight, fill=fill)
    return len(lines) * lh


# ════════════════════════════════════════════════════════════════════
# 화면
# ════════════════════════════════════════════════════════════════════

def screen_msg(scene, theme="light"):
    """재난문자 수신 — 실제 수신 문자를 그대로 인용한다."""
    s = SVG(W, H, theme=theme)
    p = s.p
    m = scene.get("disaster_msg") or {}
    body = m.get("text", "")
    at = (m.get("at") or "")[:16].replace("/", ".")
    region = (m.get("region") or "").split(",")[0].strip()

    status_bar(s, time=at[11:16] or "06:45")
    s.style(20, 98, "긴급재난문자", "title1", fill=p["fg"], family=T.FONT_DISPLAY)
    s.text(20, 124, f"{at}  ·  {region}", size=12, weight=400, fill=p["muted"])

    lines = _wrap_lines(body, 300, 15)
    ch = 38 + len(lines) * 24
    card(s, 20, 144, W - 40, ch)
    s.rect(20, 144, 4, ch, fill=T.WARN, rx=2)
    _wrap(s, body, 40, 176, 300, size=15, lh=24, fill=p["fg"])
    s.text(W - 32, 144 + ch - 13, f"{len(body)}자", size=11, weight=600,
           fill=p["muted"], anchor="end", family=T.FONT_NUM)

    qy = 144 + ch + 26
    s.text(20, qy, "이 문자가 답하지 못하는 것", size=13, weight=600, fill=p["muted"])
    qs = ["'안전한 곳'은 어디인가?",
          "밖으로 나가야 하는가, 위로 올라가야 하는가?",
          "어느 길이 침수되지 않았는가?",
          "나도 그 길로 이동할 수 있는가?"]
    for i, q in enumerate(qs):
        yy = qy + 18 + i * 44
        card(s, 20, yy, W - 40, 36, shadow=False, fill=p["surface"],
             rx=T.R["control"])
        s.text(36, yy + 23, "?", size=15, weight=700, fill=T.WARN)
        s.text(56, yy + 23, q, size=14, weight=400, fill=p["fg2"])

    by = min(qy + 18 + len(qs) * 44 + 18, H - 130)
    pill(s, 20, by, W - 40, 52, "지금 내 대피 경로 보기")
    s.text(W / 2, by + 74, "SAFE-NOTING이 이 질문에 답합니다", size=12,
           weight=500, fill=p["muted"], anchor="middle")
    home_indicator(s)
    return s, "msg"


def _guide_actions(scene):
    """
    행동강령을 readings 에서 조립한다.

    ⚠ 문장은 저자가 쓰되 숫자는 단 하나도 지어내지 않는다.
      readings 에 값이 없으면 그 항목은 통째로 빠진다 — 빈칸이 낫지
      그럴듯한 값이 낫지 않다. 문장 규칙: 한 문장 한 행동 · 하지 말 것 ·
      어디로 · 이동수단.
    """
    acts = []
    # ① 하지 말 것 — 국민행동요령의 최우선 항목
    acts.append(("지하공간·지하차도에 들어가지 마세요",
                 "국민행동요령 · 침수 시 최우선", T.DANGER))
    # ② 어디로 — 실제 경로 탐색 결과
    if has(scene, "dest_name", "walk_min"):
        acts.append((f"{R(scene,'dest_name')}으로 도보 {R(scene,'walk_min')}분 이동하세요",
                     f"실내 시설 · {R(scene,'dest_dist_km')}km", T.ROUTE_SAFE))
    # ③ 왜 그 길인가 — 실제 침수 관통 판정
    if has(scene, "short_flood_m") and R(scene, "short_flood_m", 0):
        acts.append(("최단경로 대신 안내 경로로 가세요",
                     "최단경로는 침수 이력 "
                     f"{int(R(scene,'short_flood_m',0)):,}m를 지납니다",
                     T.WARN))
    # ④ 이동 수단 — 국민행동요령
    acts.append(("물이 발목 위로 오르면 차량을 두고 걸으세요",
                 "국민행동요령 · 수심 30cm 이상 차량 포기", T.CAUTION))
    return acts


def screen_guide(scene, theme="light"):
    """
    행동 안내 — 재난문자를 '지금 무엇을 어떻게'로 바꾼다.
    문장은 규칙 기반 템플릿이고, 들어가는 수치는 전부 readings 에서 온다.

    이 화면이 이 프로젝트의 핵심 주장이다. 그래서 왼쪽에 원본 문자를,
    오른쪽에 그것을 읽은 실측 데이터를, 아래에 생성된 행동강령을 둔다.
    셋이 한 화면에 있어야 '문자 → 행동' 변환이 눈에 보인다.
    """
    s = SVG(W, H, theme=theme)
    p = s.p
    s.rect(0, 0, W, H, fill=p["canvas"])
    status_bar(s, time=R(scene, "time", "06:47"), theme=theme)

    # ── 헤더 ────────────────────────────────────────────────────────
    cw = chip(s, 20, 74, "SAFE-NOTING AI", T.ROUTE_SAFE)
    if has(scene, "level_name"):
        chip(s, 20 + cw + 6, 74, f"{R(scene,'level_name')} 단계", T.DANGER)
    s.style(20, 124, "재난문자를 행동으로", "title1", fill=p["fg"],
            family=T.FONT_DISPLAY)

    # ── 원본 재난문자 (실제 수신 문자) ──────────────────────────────
    m = scene.get("disaster_msg") or {}
    body = m.get("text", "")
    my = 146
    if body:
        lines = _wrap_lines(body, 286, 13)[:3]
        mh = 30 + len(lines) * 20
        card(s, 20, my, W - 40, mh, shadow=False, fill=p["surface"],
             rx=T.R["control"])
        s.rect(20, my, 3, mh, fill=T.WARN, rx=1.5)
        s.text(36, my + 20, "수신한 재난문자", size=11, weight=600,
               fill=p["muted"])
        for i, ln in enumerate(lines):
            s.text(36, my + 40 + i * 20, ln, size=13, weight=400, fill=p["fg2"])
        s.text(W - 32, my + 20, f"{len(body)}자", size=11, weight=600,
               fill=p["muted"], anchor="end", family=T.FONT_NUM)
        my += mh

    # ── AI가 함께 읽은 실측 데이터 ──────────────────────────────────
    dy = my + 20
    s.text(20, dy, "AI가 함께 읽은 실시간 데이터", size=12, weight=600,
           fill=p["muted"])
    facts = []
    if has(scene, "station", "wl"):
        facts.append((f"{R(scene,'station')} {R(scene,'wl')}m", T.DANGER))
    if has(scene, "wl_pred", "ahead_hours"):
        facts.append((f"{R(scene,'ahead_hours')}시간 뒤 예측 {R(scene,'wl_pred')}m",
                      T.DANGER))
    if has(scene, "rain"):
        facts.append((f"시간당 {R(scene,'rain')}mm", T.WARN))
    if has(scene, "n_flood_marks"):
        facts.append((f"침수이력 {R(scene,'n_flood_marks')}곳", T.WARN))
    fx, fy = 20, dy + 12
    for label, color in facts:
        w = len(label) * 11 * 0.62 + 16
        if fx + w > W - 20:
            fx, fy = 20, fy + 28
        chip(s, fx, fy, label, color, size=11, h=24)
        fx += w + 6
    dy = fy + 24

    # ── 생성된 행동강령 ─────────────────────────────────────────────
    s.text(20, dy + 20, "지금 해야 할 행동", size=13, weight=700, fill=p["fg"])
    chip(s, 150, dy + 6, "생성 예시 · 수치는 실측값", T.BLOCKED, size=10, h=20)

    acts = _guide_actions(scene)
    ay = dy + 36
    for i, (title, why, color) in enumerate(acts):
        yy = ay + i * 66
        card(s, 20, yy, W - 40, 56, shadow=False, fill=p["surface"],
             rx=T.R["control"])
        s.circle(46, yy + 28, 13, fill=color, op=0.15)
        s.text(46, yy + 33, str(i + 1), size=14, weight=700, fill=color,
               anchor="middle")
        s.text(70, yy + 24, title, size=14, weight=600, fill=p["fg"])
        s.text(70, yy + 42, why, size=11, weight=400, fill=p["muted"])

    # ── 근거 · CTA · 면책 ───────────────────────────────────────────
    gy = ay + len(acts) * 66 + 4
    s.text(20, gy + 14, "근거 · 행정안전부 자연재난 국민행동요령 (침수·호우)",
           size=11, weight=500, fill=p["muted"])

    by = max(gy + 30, H - 156)
    pill(s, 20, by, W - 40, 52, "안전경로로 안내 시작")
    s.text(W / 2, by + 72, "행정안전부·소방청 공식 안내를 우선하세요", size=11,
           weight=400, fill=p["muted"], anchor="middle")
    home_indicator(s)
    return s, "guide"


def screen_riskmap(scene, theme="light"):
    """동적 침수위험 지도 — 시트 내용이 탭바를 넘지 않게 높이를 계산한다."""
    s = SVG(W, H, theme=theme)
    p = s.p
    fullscreen_map(s, scene, show_risk=True, show_route=False, theme=theme,
                   zoom=1.05)
    top_scrim(s, 160)
    status_bar(s, time=R(scene, "time", "06:52"), theme=theme)

    card(s, 16, 58, W - 32, 44, rx=22)
    s.circle(40, 80, 5, fill=T.ROUTE_SAFE)
    s.text(56, 85, "충북 청주시 흥덕구 오송읍 · 현재 위치", size=13,
           weight=500, fill=p["fg"])

    # 시트 높이를 내용에서 역산한다. 고정값으로 두면 항목이 늘 때 탭바를 뚫는다.
    shel = [m for m in scene.get("markers", []) if m["kind"] == "shelter"][:2]
    need = 26 + 78 + 62 + 46 + len(shel) * 54 + 34
    sy = max(360, H - TAB_H - need)
    sheet(s, sy, theme=theme)

    lvl = R(scene, "level_name", "—")
    lvl_c = {"심각": T.DANGER, "경계": T.WARN, "주의": T.CAUTION}.get(lvl, T.SAFE)
    dist = "미호강 인근" if has(scene, "station") else ""
    risk_level_card(s, 20, sy + 26, W - 40, lvl, dist, lvl_c)

    stat_row(s, 30, sy + 132, W - 60, [
        ("시간당 강수", R(scene, "rain"), "mm", T.WARN),
        (R(scene, "station", "수위")[-6:].strip("()") or "수위",
         R(scene, "wl"), "m", T.DANGER),
        (f"{R(scene,'ahead_hours','3')}h 뒤 예측", R(scene, "wl_pred"), "m",
         T.DANGER)])

    s.line(20, sy + 172, W - 20, sy + 172, p["hairline"], 1)
    s.text(20, sy + 198, "주변 대피시설", size=13, weight=600, fill=p["muted"])
    n_ex = R(scene, "n_outdoor_excluded", 0)
    if n_ex:
        s.text(W - 20, sy + 198, f"야외 {n_ex}곳 제외됨", size=11, weight=500,
               fill=p["muted"], anchor="end")

    for i, m in enumerate(shel):
        yy = sy + 212 + i * 54
        s.circle(34, yy + 22, 9, fill=T.SAFE, op=0.18)
        s.circle(34, yy + 22, 4.5, fill=T.SAFE)
        s.text(54, yy + 19, m.get("label") or "지정 대피소", size=15,
               weight=600, fill=p["fg"])
        s.text(54, yy + 37, (m.get("kind_name") or "지정 대피소")[:22],
               size=11, fill=p["muted"])
        d = m.get("dist_km")
        s.text(W - 24, yy + 27, f"{d:.2f}km" if isinstance(d, (int, float))
               else "—", size=14, weight=600, fill=p["fg2"], anchor="end",
               family=T.FONT_NUM)

    tab_bar(s, TABS, 0, theme=theme)
    return s, "riskmap"


def screen_route(scene, theme="light"):
    """
    안전경로 안내.

    ⚠ 우회 문구는 실제로 우회했을 때만 띄운다.
      readings.routes_identical 이 True 면 두 경로가 같다는 뜻이고,
      그때 '+Nm 우회'를 적으면 거짓말이 된다.
    """
    s = SVG(W, H, theme=theme)
    p = s.p
    fullscreen_map(s, scene, show_risk=True, show_route=True, theme=theme)
    top_scrim(s, 180)
    status_bar(s, time=R(scene, "time", "06:55"), theme=theme)

    card(s, 16, 58, W - 32, 82)
    s.circle(52, 99, 22, fill=T.ROUTE_SAFE, op=0.14)
    s.text(52, 107, "↑", size=24, weight=700, fill=T.ROUTE_SAFE, anchor="middle")
    s.text(86, 92, f"{R(scene,'dest_name','대피소')}로 이동", size=18,
           weight=700, fill=p["fg"], family=T.FONT_DISPLAY)
    s.text(86, 115, f"{R(scene,'dest_kind','지정 대피소')}"[:26], size=12,
           fill=p["muted"])

    # 회피 사유 — 실제로 차단한 것만 말한다
    ups = (scene.get("readings") or {}).get("blocked_underpasses") or []
    nfm = R(scene, "n_flood_marks", 0)
    if ups:
        alert_banner(s, 16, 150, W - 32, f"{ups[0]} 지하차도 차단",
                     "침수 시 진입 금지 구간 · 우회 안내", T.DANGER, "▲", theme)
    elif nfm:
        alert_banner(s, 16, 150, W - 32, f"주변 침수 이력 {nfm}곳 반영",
                     "행정안전부 침수흔적도 기준 위험가중 경로", T.WARN, "▲", theme)

    sy = H - 244
    sheet(s, sy, theme=theme)
    s.text(20, sy + 54, R(scene, "dest_name", "지정 대피소"), size=20,
           weight=700, fill=p["fg"], family=T.FONT_DISPLAY)
    s.text(20, sy + 78, f"실내 시설 · 침수흔적에서 {R(scene, 'dest_flood_m', '—')}m",
           size=12, fill=p["muted"])
    s.text(W - 20, sy + 54, f"{R(scene,'walk_min','—')}분", size=26, weight=700,
           fill=T.ROUTE_SAFE, anchor="end", family=T.FONT_DISPLAY)
    rm = R(scene, "route_m")
    s.text(W - 20, sy + 78, f"{rm:,}m" if isinstance(rm, (int, float)) else "—",
           size=13, fill=p["muted"], anchor="end", family=T.FONT_NUM)

    s.line(20, sy + 100, W - 20, sy + 100, p["hairline"], 1)
    rd = scene.get("readings") or {}
    if rd.get("routes_identical"):
        note = "최단경로에 침수·지하차도 위험이 없어 그대로 안내합니다"
    else:
        note = (f"최단경로보다 +{rd.get('detour_m', 0)}m — "
                f"위험구간을 지나지 않습니다")
    s.text(20, sy + 126, note, size=13, weight=500, fill=p["fg2"])
    pill(s, 20, sy + 148, W - 40, 52, "안내 시작")
    home_indicator(s)
    return s, "route"


def screen_reroute(scene, theme="light"):
    """실시간 재탐색 — 재난은 움직이므로 경로도 바뀌어야 한다."""
    s = SVG(W, H, theme=theme)
    p = s.p
    fullscreen_map(s, scene, show_risk=True, show_route=True, theme=theme,
                   zoom=1.3)
    s.rect(0, 0, W, H, fill=T.DANGER, op=0.05)
    top_scrim(s, 160)
    status_bar(s, time=R(scene, "time", "07:03"), theme=theme)

    alert_banner(s, 16, 58, W - 32, "경로 앞 침수 위험 확인",
                 "CCTV 영상 판정 · 방금 갱신", T.DANGER, "▲", theme)

    cy = 152
    card(s, 16, cy, W - 32, 60, rx=30)
    s.circle(48, cy + 30, 13, fill="none", stroke=T.ROUTE_SAFE, sw=3, op=0.25)
    s.raw(f'<path d="M48,{cy + 17} A13,13 0 0 1 61,{cy + 30}" fill="none" '
          f'stroke="{T.ROUTE_SAFE}" stroke-width="3" stroke-linecap="round"/>')
    s.text(76, cy + 27, "새로운 경로를 계산하고 있습니다", size=14, weight=600,
           fill=p["fg"])
    s.text(76, cy + 45, "ROUTE RECALCULATING", size=10, weight=600,
           fill=p["muted"], tracking=1.2, family=T.FONT_NUM)

    sy = H - 224
    sheet(s, sy, theme=theme)
    s.text(20, sy + 50, "경로가 변경되었습니다", size=20, weight=700,
           fill=p["fg"], family=T.FONT_DISPLAY)
    ups = (scene.get("readings") or {}).get("blocked_underpasses") or []
    prev = f"{ups[0]} 통과" if ups else "최단경로 직진"
    rows = [("이전 경로", prev, T.BLOCKED, "침수 확인", True),
            ("변경 경로", f"{R(scene,'dest_name','대피소')} 방면",
             T.ROUTE_SAFE, f"{R(scene,'walk_min','—')}분", False)]
    for i, (k, v, c, tag, strike) in enumerate(rows):
        yy = sy + 74 + i * 54
        s.text(20, yy + 18, k, size=12, weight=500, fill=p["muted"])
        s.text(20, yy + 38, v[:18], size=15, weight=600,
               fill=p["muted"] if strike else p["fg"],
               extra='text-decoration="line-through"' if strike else "")
        chip(s, W - 122, yy + 20, tag, c)
    home_indicator(s)
    return s, "reroute"


SCREENS = {"msg": screen_msg, "guide": screen_guide, "riskmap": screen_riskmap,
           "route": screen_route, "reroute": screen_reroute}
CAPTIONS = {"msg": "재난문자 수신", "guide": "행동 안내",
            "riskmap": "동적 침수위험 지도", "route": "안전경로 안내",
            "reroute": "실시간 재탐색"}
_NUM = "①②③④⑤⑥⑦⑧⑨"


def caption_for(name, i):
    """
    슬라이드에 실린 순서대로 번호를 매긴다.

    ⚠ 5화면 기준 고정 번호를 쓰면 부분 슬라이드에서 ①②④ 처럼 건너뛴다.
      화면에서 그건 오탈자로 읽힌다.
    """
    return f"{_NUM[i] if i < len(_NUM) else i + 1} {CAPTIONS.get(name, name)}"


# ════════════════════════════════════════════════════════════════════
# 출력
# ════════════════════════════════════════════════════════════════════

def render_screens(scene, names, out_dir="out", png=True,
                   theme="light"):
    out = []
    for n in names:
        if n not in SCREENS:
            print(f"  ! 알 수 없는 화면: {n}")
            continue
        s, tag = SCREENS[n](scene, theme)
        mockup_badge(s, x=W - 92, y=H - 32, label="UI 설계안", theme=theme)
        out.append(_save(s, f"app-{tag}", out_dir, png, W, H))
    return out


def render_slide(scene, names, out_dir="out", png=True,
                 theme="light", title=None, subtitle=None):
    """1920×1080 슬라이드에 폰 화면 여러 개를 배치한다."""
    SW, SH = T.SLIDE
    s = SVG(SW, SH, theme="dark")
    p = T.pal("dark")
    s.style(64, 92, title or scene.get("title", "SAFE-NOTING 앱 화면"),
            "section", scale="S", fill=p["fg"], family=T.FONT_DISPLAY)
    s.style(64, 136, subtitle or scene.get("subtitle", ""), "body", scale="S",
            fill=p["fg2"])

    n = max(1, len(names))
    scale = 0.76 if n <= 3 else 0.60
    pw, ph = W * scale, H * scale
    gap = 58 if n <= 3 else 34
    x0 = (SW - (n * pw + (n - 1) * gap)) / 2
    y0 = 214

    for i, nm in enumerate(names):
        if nm not in SCREENS:
            continue
        sub, _ = SCREENS[nm](scene, theme)
        x = x0 + i * (pw + gap)
        sh = s.shadow(18, 56, 0.55)
        s.rect(x - 7, y0 - 7, pw + 14, ph + 14, fill="#2c2c2e",
               rx=(T.R["sheet"] + 7) * scale, extra=f'filter="url(#{sh})"')
        cid = s.clip_rect(x, y0, pw, ph, rx=T.R["sheet"] * scale)
        s.defs.extend(sub.defs)
        s.raw(f'<g clip-path="url(#{cid})">'
              f'<g transform="translate({x},{y0}) scale({scale})">')
        s.raw("".join(sub.o))
        s.raw("</g></g>")
        s.text(x + pw / 2, y0 + ph + 44, caption_for(nm, i), size=17,
               weight=600, fill=p["fg"], anchor="middle")

    s.line(64, SH - 78, SW - 64, SH - 78, p["hairline"], 1, cap="butt", op=0.7)
    source_note(s, "출처 · " + " | ".join(scene.get("sources", [])),
                x=64, y=SH - 46, theme="dark", size=13)
    mockup_badge(s, x=SW - 296, y=SH - 62, label="UI 설계안 · 수치는 실측값",
                 theme="dark")
    return _save(s, f"slide-app-{'-'.join(names)}", out_dir, png, SW, SH)


def _save(s, name, out_dir, png, w, h):
    os.makedirs(out_dir, exist_ok=True)
    svg = os.path.join(out_dir, name + ".svg")
    s.save(svg)
    print(f"  → {svg}  ({os.path.getsize(svg) // 1024}KB)")
    if png:
        q = export_png(svg, width=w, height=h, scale=2 if w <= 500 else 1)
        if q:
            print(f"  → {q}  ({os.path.getsize(q) // 1024}KB)")
    return svg


def main():
    a = sys.argv[1:]
    if not a:
        raise SystemExit("사용: render_ui.py <scene.json> [--screens a,b] "
                         "[--mode screens|slide] [--out DIR] [--no-png]")
    with open(a[0], encoding="utf-8") as f:
        scene = json.load(f)
    if not scene.get("readings"):
        print("  ⚠ 이 장면에는 readings 가 없다. 수치 자리는 '—' 로 그려진다.")
    names = (a[a.index("--screens") + 1].split(",") if "--screens" in a
             else ["msg", "guide", "riskmap", "route", "reroute"])
    mode = a[a.index("--mode") + 1] if "--mode" in a else "screens"
    out = a[a.index("--out") + 1] if "--out" in a else "out"
    png = "--no-png" not in a
    print(f"■ 앱 화면 렌더: {scene['id']}  mode={mode}  screens={','.join(names)}")
    if mode == "slide":
        render_slide(scene, names, out, png)
    else:
        render_screens(scene, names, out, png)


if __name__ == "__main__":
    main()
