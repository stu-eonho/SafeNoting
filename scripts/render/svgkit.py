# -*- coding: utf-8 -*-
"""
SAFE-NOTING 렌더링 툴킷 — 디자인 토큰 + SVG 빌더 + 지도 투영

■ 왜 SVG인가
  이 환경에는 playwright도 cairosvg도 없다. 그러나 이미지는 확대해도
  깨지지 않아야 하고, PowerPoint 2016+는 SVG를 그대로 삽입할 수 있다.
  SVG를 직접 문자열로 만들면 의존성이 0이고, 텍스트가 벡터로 남아
  텍스트를 바로 고칠 수도 있다.

■ 왜 토큰을 코드에 두는가
  색·글꼴·간격을 렌더러마다 하드코딩하면 지도 시안과 앱 시안이 서로
  다른 디자인이 된다. 한 곳에서만 정의하고 전부 여기서 가져다 쓴다.

사용:
    import sys; sys.path.insert(0, "scripts/render")
    from svgkit import SVG, T, Projection, phone_frame
"""
import math
import os

# ════════════════════════════════════════════════════════════════════
# 디자인 토큰 — Apple HIG 기반 + 재난 도메인 확장
# ════════════════════════════════════════════════════════════════════

class T:
    """디자인 토큰. 모든 렌더러는 여기서만 색·크기를 가져온다."""

    # ── 글꼴 ────────────────────────────────────────────────────────
    # SF Pro는 이 환경에 없다. Pretendard가 설치돼 있으면 가장 가깝고,
    # 없으면 Noto Sans KR로 떨어진다(둘 다 없을 때를 대비해 맑은 고딕까지).
    FONT = ('Pretendard Variable, Pretendard, -apple-system, "SF Pro Text", '
            '"Noto Sans KR", "Malgun Gothic", sans-serif')
    FONT_DISPLAY = ('Pretendard Variable, Pretendard, -apple-system, "SF Pro Display", '
                    '"Noto Sans KR", "Malgun Gothic", sans-serif')
    # 숫자가 세로로 정렬돼야 하는 곳(수위, 시각, 거리)
    FONT_NUM = ('"SF Mono", ui-monospace, "Menlo", '
                '"Noto Sans KR", "Malgun Gothic", monospace')

    # ── 라이트 팔레트 (앱 UI 기본) ──────────────────────────────────
    L = dict(
        primary="#0071e3",       # Apple 확인값 — 채워진 주요 액션
        link="#0066cc",          # 라이트 표면 링크 / 아웃라인 버튼
        canvas="#f5f5f7",        # 페이지 바탕
        surface="#ffffff",       # 카드·시트 표면
        surface2="#fbfbfd",      # 한 단계 낮은 표면
        fg="#1d1d1f",            # 본문
        fg2="#515154",           # 보조 텍스트
        muted="#6e6e73",         # 캡션·라벨
        hairline="#d2d2d7",      # 구분선
        map_land="#f2f1ec",      # 지도 바탕(Apple Maps 라이트 계열)
        map_road="#ffffff",
        map_road_edge="#e3e2dc",
        map_bldg="#e8e7e1",
        map_water="#a9d3ec",
        map_rail="#cfcec8",
        map_label="#6e6e73",
    )

    # ── 다크 팔레트 (발표 슬라이드 지도 기본) ────────────────────────
    D = dict(
        primary="#2997ff",       # Apple 확인값 — 어두운 표면 위 블루
        link="#2997ff",
        canvas="#000000",
        surface="#1c1c1e",
        surface2="#2c2c2e",
        fg="#f5f5f7",
        fg2="#aeaeb2",
        muted="#8e8e93",
        hairline="#38383a",
        map_land="#0d1117",
        map_road="#3a3a3c",
        map_road_edge="#2c2c2e",
        map_bldg="#1c2128",
        map_water="#0a3d62",
        map_rail="#2c2c2e",
        map_label="#8e8e93",
    )

    # ── 재난 시맨틱 (테마 무관 고정 — 의미가 색보다 우선한다) ─────────
    # iOS 시스템 컬러를 쓴다. 사용자가 이미 아는 색이라 학습이 필요 없다.
    DANGER = "#ff3b30"    # 침수 / 진입 금지
    WARN = "#ff9f0a"      # 주의 / 위험 상승
    CAUTION = "#ffd60a"   # 경보 단계
    SAFE = "#30d158"      # 이용 가능 대피소
    BLOCKED = "#8e8e93"   # 통제 / 접근 불가
    ROUTE_SAFE = "#0a84ff"  # 안전경로 — 어느 테마에서도 같은 파랑
    ROUTE_RISK = "#ff453a"  # 최단경로(위험)
    UNDERPASS = "#ffd60a"   # 지하차도 구간

    # ── 타이포 스케일 ───────────────────────────────────────────────
    # 슬라이드용(Apple 마케팅 스케일)
    S = dict(
        hero=(56, 600, -0.28),
        section=(40, 600, 0),
        tile=(28, 400, 0.2),
        body=(17, 400, -0.37),
        small=(14, 400, -0.22),
        caption=(12, 400, -0.12),
    )
    # 앱 UI용(iOS 타입 램프)
    I = dict(
        largeTitle=(34, 700, 0.37),
        title1=(28, 700, 0.36),
        title2=(22, 700, 0.35),
        title3=(20, 600, 0.38),
        headline=(17, 600, -0.41),
        body=(17, 400, -0.41),
        callout=(16, 400, -0.32),
        subhead=(15, 400, -0.24),
        footnote=(13, 400, -0.08),
        caption1=(12, 400, 0),
        caption2=(11, 400, 0.07),
    )

    # ── 간격 / 반경 ─────────────────────────────────────────────────
    SP = dict(xs=4, sm=8, md=16, lg=20, xl=32, xxl=48)
    R = dict(control=8, card=14, sheet=20, chip=999, pill=980)

    # ── 아트보드 ────────────────────────────────────────────────────
    PHONE = (393, 852)        # iPhone 15/16 논리 해상도
    SLIDE = (1920, 1080)      # 16:9 발표 슬라이드
    SLIDE_HALF = (940, 1080)  # Before/After 좌우 분할용

    @staticmethod
    def pal(theme="light"):
        return T.D if theme == "dark" else T.L


# ════════════════════════════════════════════════════════════════════
# 지도 투영 — WGS84 → 아트보드 픽셀
# ════════════════════════════════════════════════════════════════════

class Projection:
    """
    Web Mercator 투영 후 bbox를 뷰포트에 맞춘다.

    ⚠ 가로세로 비율을 반드시 보존한다. 늘리면 도로 각도가 틀어져
      지도가 미묘하게 어색해 보인다.
    """

    def __init__(self, bbox, width, height, pad=0):
        """bbox = (lat_min, lat_max, lon_min, lon_max)"""
        self.lat0, self.lat1, self.lon0, self.lon1 = bbox
        self.w, self.h = width, height
        self.pad = pad
        y0, y1 = self._my(self.lat0), self._my(self.lat1)
        self.my0, self.my1 = min(y0, y1), max(y0, y1)
        iw, ih = width - 2 * pad, height - 2 * pad
        sx = iw / (self.lon1 - self.lon0)
        sy = ih / (self.my1 - self.my0)
        self.s = min(sx, sy)                       # 비율 보존
        self.ox = pad + (iw - self.s * (self.lon1 - self.lon0)) / 2
        self.oy = pad + (ih - self.s * (self.my1 - self.my0)) / 2

    @staticmethod
    def _my(lat):
        lat = max(-85.0, min(85.0, lat))
        return math.degrees(math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))

    def __call__(self, lat, lon):
        x = self.ox + (lon - self.lon0) * self.s
        y = self.oy + (self.my1 - self._my(lat)) * self.s
        return x, y

    def pts(self, coords, latlon=True):
        """[[lat,lon],...] → [(x,y),...]"""
        if latlon:
            return [self(c[0], c[1]) for c in coords]
        return [self(c[1], c[0]) for c in coords]

    @staticmethod
    def bbox_of(coords, margin=0.12, latlon=True):
        """좌표 목록을 감싸는 bbox. margin은 범위 대비 비율."""
        la = [c[0] if latlon else c[1] for c in coords]
        lo = [c[1] if latlon else c[0] for c in coords]
        dla = (max(la) - min(la)) or 0.004
        dlo = (max(lo) - min(lo)) or 0.004
        return (min(la) - dla * margin, max(la) + dla * margin,
                min(lo) - dlo * margin, max(lo) + dlo * margin)


def haversine_m(a, b):
    """두 [lat,lon] 사이 거리(m). 축척 막대와 거리 라벨에 쓴다."""
    R = 6371000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp = p2 - p1
    dl = math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def path_len_m(coords):
    return sum(haversine_m(coords[i], coords[i + 1]) for i in range(len(coords) - 1))


# ════════════════════════════════════════════════════════════════════
# SVG 빌더
# ════════════════════════════════════════════════════════════════════

def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


class SVG:
    """
    문자열을 쌓아 SVG 문서를 만든다.

    좌표는 전부 아트보드 픽셀 기준이다. 뷰포트를 논리 해상도로 두고
    PPT에서 확대하는 방식이라, 여기서 2배 크기로 그릴 필요가 없다.
    """

    def __init__(self, w, h, bg=None, theme="light"):
        self.w, self.h = w, h
        self.theme = theme
        self.p = T.pal(theme)
        self.o = []
        self.defs = []
        self._uid = 0
        if bg is None:
            bg = self.p["canvas"]
        if bg != "none":
            self.rect(0, 0, w, h, fill=bg)

    def uid(self, k="id"):
        self._uid += 1
        return f"{k}{self._uid}"

    # ── 원시 도형 ───────────────────────────────────────────────────
    def raw(self, s):
        self.o.append(s)
        return self

    def rect(self, x, y, w, h, fill="none", rx=0, stroke=None, sw=1, op=None, extra=""):
        a = f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}"'
        if rx:
            # ⚠ SVG 의 rx 는 폭의 절반까지 커진다. Apple 의 pill(980) 을 그대로
            #   넘기면 넓은 버튼이 캡슐이 아니라 '타원'이 된다 — 실제로 그랬다.
            #   높이의 절반으로 잘라야 캡슐 모양이 나온다.
            rx = min(rx, w / 2.0, h / 2.0)
            a += f' rx="{rx:.2f}"'
        if stroke:
            a += f' stroke="{stroke}" stroke-width="{sw}"'
        if op is not None:
            a += f' opacity="{op}"'
        return self.raw(a + f' {extra}/>')

    def circle(self, cx, cy, r, fill="none", stroke=None, sw=1, op=None):
        a = f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}"'
        if stroke:
            a += f' stroke="{stroke}" stroke-width="{sw}"'
        if op is not None:
            a += f' opacity="{op}"'
        return self.raw(a + '/>')

    def line(self, x1, y1, x2, y2, stroke, sw=1, cap="round", op=None, dash=None):
        a = (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
             f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="{cap}"')
        if dash:
            a += f' stroke-dasharray="{dash}"'
        if op is not None:
            a += f' opacity="{op}"'
        return self.raw(a + '/>')

    def poly(self, pts, fill="none", stroke=None, sw=1, close=False, op=None,
             cap="round", join="round", dash=None):
        if len(pts) < 2:
            return self
        d = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in pts) + (" Z" if close else "")
        a = f'<path d="{d}" fill="{fill}"'
        if stroke:
            a += (f' stroke="{stroke}" stroke-width="{sw}" '
                  f'stroke-linecap="{cap}" stroke-linejoin="{join}"')
        if dash:
            a += f' stroke-dasharray="{dash}"'
        if op is not None:
            a += f' opacity="{op}"'
        return self.raw(a + '/>')

    def text(self, x, y, s, size=17, weight=400, fill=None, anchor="start",
             tracking=0, family=None, op=None, extra=""):
        fill = fill or self.p["fg"]
        family = family or T.FONT
        a = (f'<text x="{x:.2f}" y="{y:.2f}" font-family=\'{family}\' '
             f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
             f'text-anchor="{anchor}"')
        if tracking:
            a += f' letter-spacing="{tracking}"'
        if op is not None:
            a += f' opacity="{op}"'
        return self.raw(a + f' {extra}>{esc(s)}</text>')

    def style(self, x, y, s, key, scale="I", **kw):
        """토큰 스케일로 텍스트. key는 T.I / T.S 의 키."""
        size, weight, tr = (T.I if scale == "I" else T.S)[key]
        kw.setdefault("tracking", tr)
        return self.text(x, y, s, size=size, weight=weight, **kw)

    def group(self, extra=""):
        self.raw(f'<g {extra}>')
        return self

    def endgroup(self):
        return self.raw('</g>')

    # ── 효과 ────────────────────────────────────────────────────────
    def shadow(self, dy=8, blur=24, op=0.14, color="#000000"):
        """iOS풍 부드러운 그림자 필터를 정의하고 id를 반환한다."""
        i = self.uid("sh")
        self.defs.append(
            f'<filter id="{i}" x="-50%" y="-50%" width="200%" height="200%">'
            f'<feDropShadow dx="0" dy="{dy}" stdDeviation="{blur/2:.1f}" '
            f'flood-color="{color}" flood-opacity="{op}"/></filter>')
        return i

    def clip_rect(self, x, y, w, h, rx=0):
        i = self.uid("cp")
        self.defs.append(
            f'<clipPath id="{i}"><rect x="{x}" y="{y}" width="{w}" '
            f'height="{h}" rx="{rx}"/></clipPath>')
        return i

    def lingrad(self, stops, x1=0, y1=0, x2=0, y2=1):
        """stops = [(offset, color, opacity), ...]"""
        i = self.uid("lg")
        s = "".join(f'<stop offset="{o}" stop-color="{c}" stop-opacity="{a}"/>'
                    for o, c, a in stops)
        self.defs.append(
            f'<linearGradient id="{i}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}">{s}</linearGradient>')
        return i

    # ── 출력 ────────────────────────────────────────────────────────
    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        d = f'<defs>{"".join(self.defs)}</defs>' if self.defs else ""
        doc = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
               f'height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
               f'{d}{"".join(self.o)}</svg>')
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)
        return path


# ════════════════════════════════════════════════════════════════════
# 컴포넌트 — 앱 UI 크롬
# ════════════════════════════════════════════════════════════════════

def status_bar(s, y=0, time="08:12", theme=None):
    """iOS 상태바. 시각은 시나리오 시각으로 바꿔 넣는다."""
    p = T.pal(theme or s.theme)
    fg = p["fg"]
    s.text(28, y + 33, time, size=17, weight=600, fill=fg, family=T.FONT)
    # 신호·와이파이·배터리 — 형태만 암시한다. 과하게 그리면 시선을 뺏는다.
    bx = s.w - 30
    s.rect(bx - 26, y + 20, 24, 12, fill="none", stroke=fg, sw=1, rx=3, op=0.5)
    s.rect(bx - 24.5, y + 21.5, 18, 9, fill=fg, rx=1.5, op=0.9)
    for i, hgt in enumerate((4, 6, 8, 10)):
        s.rect(bx - 74 + i * 6, y + 30 - hgt, 4, hgt, fill=fg, rx=1, op=0.9)
    return s


def home_indicator(s, theme=None):
    p = T.pal(theme or s.theme)
    s.rect(s.w / 2 - 67, s.h - 9, 134, 5, fill=p["fg"], rx=2.5, op=0.35)
    return s


def phone_frame(w=None, h=None, theme="light", bg=None):
    """앱 화면 아트보드 하나를 만든다. 상태바·홈인디케이터는 호출자가 붙인다."""
    w = w or T.PHONE[0]
    h = h or T.PHONE[1]
    return SVG(w, h, bg=bg, theme=theme)


def pill(s, x, y, w, h, label, fill=None, fg="#ffffff", size=17, weight=600, icon=None):
    """Apple 마케팅 pill 버튼 (radius 980)."""
    fill = fill or s.p["primary"]
    s.rect(x, y, w, h, fill=fill, rx=T.R["pill"])
    tx = x + w / 2 + (9 if icon else 0)
    if icon:
        s.text(x + w / 2 - len(label) * size * 0.32 - 10, y + h / 2 + size * 0.36,
               icon, size=size, fill=fg)
    s.text(tx, y + h / 2 + size * 0.36, label, size=size, weight=weight,
           fill=fg, anchor="middle", tracking=-0.4)
    return s


def card(s, x, y, w, h, fill=None, rx=None, shadow=True, stroke=None):
    fill = fill or s.p["surface"]
    rx = T.R["card"] if rx is None else rx
    ex = f'filter="url(#{s.shadow(6, 20, 0.10)})"' if shadow else ""
    s.rect(x, y, w, h, fill=fill, rx=rx, stroke=stroke, sw=1, extra=ex)
    return s


def chip(s, x, y, label, color, fg=None, size=12, ph=8, h=22, weight=600):
    """상태 칩. 색으로 심각도를, 글자로 의미를 동시에 전달한다."""
    w = len(label) * size * 0.62 + ph * 2
    s.rect(x, y, w, h, fill=color, rx=T.R["chip"], op=0.16)
    s.text(x + w / 2, y + h / 2 + size * 0.36, label, size=size, weight=weight,
           fill=fg or color, anchor="middle")
    return w


def sheet(s, y, title=None, theme=None):
    """하단 바텀시트. iOS는 상단 모서리만 둥글고 그랩바가 있다."""
    p = T.pal(theme or s.theme)
    i = s.shadow(-6, 30, 0.16)
    s.rect(0, y, s.w, s.h - y + T.R["sheet"], fill=p["surface"],
           rx=T.R["sheet"], extra=f'filter="url(#{i})"')
    s.rect(s.w / 2 - 18, y + 8, 36, 5, fill=p["muted"], rx=2.5, op=0.4)
    if title:
        s.style(20, y + 44, title, "title3", fill=p["fg"])
    return s


def tab_bar(s, items, active=0, theme=None):
    """하단 탭바. items = [(아이콘문자, 라벨), ...]"""
    p = T.pal(theme or s.theme)
    ty = s.h - 83
    s.rect(0, ty, s.w, 83, fill=p["surface"], op=0.96)
    s.line(0, ty, s.w, ty, p["hairline"], 0.5, cap="butt")
    n = len(items)
    for i, (ic, lb) in enumerate(items):
        cx = s.w * (i + 0.5) / n
        c = p["primary"] if i == active else p["muted"]
        s.text(cx, ty + 30, ic, size=22, fill=c, anchor="middle")
        s.text(cx, ty + 46, lb, size=10, weight=500, fill=c, anchor="middle")
    return s


def alert_banner(s, x, y, w, title, body, color, icon="!", theme=None):
    """상단 경보 배너. 재난 앱의 1차 전달 수단이라 가장 강한 대비를 준다."""
    p = T.pal(theme or s.theme)
    h = 74
    card(s, x, y, w, h, fill=p["surface"])
    s.rect(x, y, 4, h, fill=color, rx=2)
    s.circle(x + 34, y + h / 2, 15, fill=color, op=0.15)
    s.text(x + 34, y + h / 2 + 7, icon, size=18, weight=700, fill=color, anchor="middle")
    s.style(x + 60, y + 30, title, "headline", fill=p["fg"])
    s.style(x + 60, y + 51, body, "footnote", fill=p["muted"])
    return h


# ════════════════════════════════════════════════════════════════════
# 지도 레이어 — 실데이터를 받아 그린다
# ════════════════════════════════════════════════════════════════════

def draw_basemap(s, proj, ctx, theme=None, detail=1.0):
    """
    OSM 맥락 데이터(context_*.json)를 배경 지도로 그린다.

    순서가 중요하다: 건물 → 철도 → 도로 → 물.
    물을 마지막에 그려야 하천이 도로 위로 올라오지 않는다.
    """
    p = T.pal(theme or s.theme)
    for b in ctx.get("buildings", []):
        pts = proj.pts(b["pts"])
        if len(pts) > 2:
            s.poly(pts, fill=p["map_bldg"], close=True)
    for r in ctx.get("rail", []):
        pts = proj.pts(r["pts"])
        s.poly(pts, stroke=p["map_rail"], sw=1.6, dash="6 5")
    for r in ctx.get("roads", []):
        hw = (r.get("tags") or {}).get("highway", "")
        sw = 4.2 if hw in ("motorway", "trunk", "primary") else \
             3.0 if hw in ("secondary", "tertiary") else 1.8
        s.poly(proj.pts(r["pts"]), stroke=p["map_road"], sw=sw * detail)
    for w in ctx.get("water", []):
        raw = w["pts"]
        tags = w.get("tags") or {}
        pts = proj.pts(raw)
        # ⚠ 점 개수로 면/선을 가르면 안 된다.
        #   미호강 중심선은 450점짜리 '선'인데 개수로 판정하면 면으로 칠해져
        #   지도 절반이 파란 쐐기로 뭉갠다(실제로 그렇게 났다).
        #   OSM은 면을 natural=water로 태깅하고 폴리곤을 닫는다. 그것만 본다.
        is_area = tags.get("natural") == "water" or (len(raw) > 3 and raw[0] == raw[-1])
        if is_area and len(pts) > 2:
            s.poly(pts, fill=p["map_water"], close=True, op=0.92)
        else:
            sw = 5.5 if tags.get("waterway") == "river" else 2.4
            s.poly(pts, stroke=p["map_water"], sw=sw * detail, op=0.92)
    return s


def draw_flood(s, proj, polys, color=None, op=0.55, stroke_op=0.9):
    """침수 구역 폴리곤. 면으로 그려야 '구역'으로 읽힌다."""
    color = color or T.DANGER
    for poly in polys:
        pts = proj.pts(poly)
        if len(pts) > 2:
            s.poly(pts, fill=color, close=True, op=op)
            s.poly(pts, stroke=color, sw=1, close=True, op=stroke_op)
    return s


def draw_route(s, proj, coords, color, sw=6, casing=True, dash=None, op=1.0):
    """
    경로선. 케이싱(어두운 테두리)을 깔아야 복잡한 지도 위에서도 읽힌다.
    Apple/Google 지도가 모두 쓰는 기법이다.
    """
    pts = proj.pts(coords)
    if casing:
        s.poly(pts, stroke="#000000", sw=sw + 3, op=0.25)
    s.poly(pts, stroke=color, sw=sw, dash=dash, op=op)
    return pts


def marker_pin(s, x, y, color, label=None, r=13, fg="#ffffff", size=13):
    """지도 핀. 그림자를 줘야 지도 표면 위로 떠 보인다."""
    s.circle(x, y + 2, r + 1, fill="#000000", op=0.22)
    s.circle(x, y, r, fill=color, stroke="#ffffff", sw=2.5)
    if label:
        s.text(x, y + size * 0.36, label, size=size, weight=700,
               fill=fg, anchor="middle")
    return s


def marker_here(s, x, y, color=None):
    """현재 위치. iOS의 파란 점 + 정확도 원."""
    color = color or T.ROUTE_SAFE
    s.circle(x, y, 26, fill=color, op=0.16)
    s.circle(x, y, 11, fill="#ffffff")
    s.circle(x, y, 8, fill=color)
    return s


def map_label(s, x, y, text, theme=None, size=12, weight=600, halo=True):
    """지도 위 지명. 후광을 깔지 않으면 도로 위에서 읽히지 않는다."""
    p = T.pal(theme or s.theme)
    if halo:
        s.text(x, y, text, size=size, weight=weight, fill=p["map_land"],
               anchor="middle", extra=f'stroke="{p["map_land"]}" stroke-width="3.5" '
                                      f'stroke-linejoin="round"')
    s.text(x, y, text, size=size, weight=weight, fill=p["map_label"], anchor="middle")
    return s


def scale_bar(s, proj, x, y, theme=None, target_px=110):
    """축척 막대."""
    p = T.pal(theme or s.theme)
    lat = (proj.lat0 + proj.lat1) / 2
    x0, _ = proj(lat, proj.lon0)
    x1, _ = proj(lat, proj.lon1)
    span_m = haversine_m([lat, proj.lon0], [lat, proj.lon1])
    mpp = span_m / max(1e-6, (x1 - x0))
    for cand in (50, 100, 200, 300, 500, 1000, 2000):
        px = cand / mpp
        if px >= target_px * 0.6:
            break
    s.line(x, y, x + px, y, p["map_label"], 2, cap="butt")
    s.line(x, y - 4, x, y + 4, p["map_label"], 2, cap="butt")
    s.line(x + px, y - 4, x + px, y + 4, p["map_label"], 2, cap="butt")
    lbl = f"{cand}m" if cand < 1000 else f"{cand//1000}km"
    s.text(x + px / 2, y - 9, lbl, size=11, weight=600,
           fill=p["map_label"], anchor="middle")
    return s


def legend(s, x, y, items, theme=None, size=13, gap=26, swatch=16):
    """
    범례. items = [(색, 라벨, 종류), ...]  종류: line|area|dot
    범례가 없으면 읽는 사람이 색의 의미를 추측해야 한다.
    """
    p = T.pal(theme or s.theme)
    for i, (c, lb, kind) in enumerate(items):
        yy = y + i * gap
        if kind == "line":
            s.line(x, yy, x + swatch + 8, yy, c, 5)
        elif kind == "area":
            s.rect(x, yy - 7, swatch + 8, 14, fill=c, rx=3, op=0.6)
            s.rect(x, yy - 7, swatch + 8, 14, fill="none", stroke=c, sw=1, rx=3)
        else:
            s.circle(x + (swatch + 8) / 2, yy, 7, fill=c, stroke="#ffffff", sw=1.5)
        s.text(x + swatch + 20, yy + size * 0.36, lb, size=size,
               weight=500, fill=p["fg2"])
    return s


# ════════════════════════════════════════════════════════════════════
# 출처 표기 — 이 툴킷의 존재 이유 중 하나
# ════════════════════════════════════════════════════════════════════

def source_note(s, text, y=None, theme=None, size=12, x=None, anchor="start"):
    """
    아트보드 하단 출처. 모든 아트보드에 데이터 출처를 붙인다.
    """
    p = T.pal(theme or s.theme)
    y = y if y is not None else s.h - 18
    x = x if x is not None else 24
    s.text(x, y, text, size=size, weight=400, fill=p["muted"], anchor=anchor)
    return s


def export_png(svg_path, png_path=None, width=None, height=None, scale=2):
    """
    SVG → PNG. cairosvg가 없는 환경이라 설치된 브라우저를 헤드리스로 쓴다.

    scale=2로 두는 이유: PPT에서 슬라이드 폭에 맞춰 확대하면 1배 래스터는
    가장자리가 뭉갠다. 2배로 뽑아두면 인쇄·빔프로젝터 모두에서 선명하다.
    (SVG를 PowerPoint에 직접 넣을 수 있다면 그쪽이 항상 더 낫다)

    반환: 생성된 png 경로 / 실패 시 None
    """
    import subprocess
    import pathlib

    svg_path = os.path.abspath(svg_path)
    png_path = os.path.abspath(png_path or os.path.splitext(svg_path)[0] + ".png")

    if width is None or height is None:
        head = open(svg_path, encoding="utf-8").read(400)
        import re
        mw = re.search(r'width="(\d+)"', head)
        mh = re.search(r'height="(\d+)"', head)
        width = width or (int(mw.group(1)) if mw else 1920)
        height = height or (int(mh.group(1)) if mh else 1080)

    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    exe = next((c for c in candidates if os.path.exists(c)), None)
    if not exe:
        print(f"  ! 브라우저를 찾지 못해 PNG 변환을 건너뜀 — SVG를 직접 사용: {svg_path}")
        return None

    cmd = [exe, "--headless=new", "--disable-gpu", "--hide-scrollbars",
           f"--screenshot={png_path}", f"--window-size={width},{height}",
           f"--force-device-scale-factor={scale}",
           pathlib.Path(svg_path).as_uri()]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
    except Exception as e:
        print(f"  ! PNG 변환 실패({e}) — SVG를 직접 사용")
        return None
    if r.returncode != 0 or not os.path.exists(png_path):
        print(f"  ! PNG 변환 실패(exit {r.returncode}) — SVG를 직접 사용")
        return None
    return png_path


def mockup_badge(s, x=None, y=24, label="UI 설계안", theme=None):
    """
    '이것은 화면 시안이다'를 명시하는 배지.

    ⚠ 실제로 구동된 화면이 아닌 아트보드에는 반드시 붙인다.
      실제 구동 화면이 아니라는 사실을 숨기지 않는다.
    """
    p = T.pal(theme or s.theme)
    size = 11
    w = len(label) * size * 0.72 + 20
    x = x if x is not None else s.w - w - 24
    s.rect(x, y, w, 22, fill=p["fg"], rx=T.R["chip"], op=0.08)
    s.text(x + w / 2, y + 15, label, size=size, weight=600,
           fill=p["muted"], anchor="middle", tracking=0.3)
    return s
