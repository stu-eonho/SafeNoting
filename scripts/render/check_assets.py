# -*- coding: utf-8 -*-
"""
산출물 검증기 — 화면의 수치가 장면 명세의 실데이터에서 왔는지 확인한다

■ 이 검증기가 존재하는 이유
  이미지는 한 번 만들고 끝이 아니라 여러 번 고쳐 그린다.
  그 과정에서 가장 흔하게 벌어지는 사고가 두 가지다.

    ① 화면에 그럴듯한 숫자가 슬쩍 들어간다
       (청주 장면에 미호강 수위를 적어 넣는 사고가 실제로 났다)
    ② 아트보드 밖으로 글자가 넘어가거나 탭바 아래로 잘린다
       (바텀시트 항목이 늘면서 탭바를 뚫는 사고가 실제로 났다)

  둘 다 눈으로는 놓치기 쉽다.

■ 검사 항목
  A 추적성   화면의 모든 숫자가 scene 의 실측값에서 유래하는가
  B 경계     텍스트·도형이 아트보드를 벗어나지 않는가
  C 커버리지  배경 지도가 비어 있지 않은가
  D 출처     출처 표기와 provenance 파일이 실제로 존재하는가
  E 형태     캡슐 반경이 타원으로 퍼지지 않았는가

사용:
    python check_assets.py scenes out
    python check_assets.py --scene <scene.json> --asset <a.svg> [<b.svg> ...]
"""
import glob
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

# 숫자로 보이지만 데이터가 아닌 것들 — 추적성 검사에서 제외한다
ALLOW_LITERAL = {
    "1", "2", "3", "4", "5",           # 순번 · 단계 표시
    "0", "24", "12", "10", "100",      # 흔한 UI 상수
    "2023", "2026", "2017", "2020",    # 연도 (사건 연도는 문서로 뒷받침됨)
    "90", "157",                       # 재난문자 글자수 제한 (공고·보도자료 근거)
    "300", "500",                      # 반경 안내 문구
}
NUM_RE = re.compile(r"(?<![A-Za-z])(\d+(?:[.,]\d+)?)")
# 날짜·시각은 숫자 추적 대상이 아니다. 먼저 떼어내지 않으면
# "2026.07.22" 가 소수 2026.07 로 읽혀 오탐이 난다 — 실제로 그랬다.
DATETIME_RE = re.compile(r"\d{2,4}[./\-]\d{1,2}[./\-]\d{1,2}|\d{1,2}:\d{2}(?::\d{2})?")
TEXT_RE = re.compile(r"<text[^>]*>(.*?)</text>", re.S)
RECT_RE = re.compile(r'<rect[^>]*?width="([\d.]+)"[^>]*?height="([\d.]+)"'
                     r'[^>]*?rx="([\d.]+)"')
SVG_SIZE_RE = re.compile(r'<svg[^>]*width="(\d+)"[^>]*height="(\d+)"')
XY_RE = re.compile(r'<text x="(-?[\d.]+)" y="(-?[\d.]+)"')


class Report:
    def __init__(self):
        self.rows = []

    def add(self, level, code, target, msg):
        self.rows.append((level, code, target, msg))

    @property
    def fails(self):
        return [r for r in self.rows if r[0] == "FAIL"]

    @property
    def warns(self):
        return [r for r in self.rows if r[0] == "WARN"]

    def print(self):
        if not self.rows:
            print("  이상 없음")
            return
        for lv, code, tgt, msg in sorted(self.rows, key=lambda r: r[0]):
            mark = "✗" if lv == "FAIL" else "!"
            print(f"  {mark} [{code}] {tgt}")
            print(f"      {msg}")


# ════════════════════════════════════════════════════════════════════
# 장면 검사
# ════════════════════════════════════════════════════════════════════

def known_numbers(scene):
    """이 장면이 정당하게 화면에 띄울 수 있는 숫자 집합."""
    ok = set(ALLOW_LITERAL)

    def push(v):
        if isinstance(v, (int, float)):
            ok.add(_norm(v))
            ok.add(_norm(round(float(v))))
            ok.add(_norm(round(float(v), 1)))
            ok.add(_norm(round(float(v), 2)))
        elif isinstance(v, str):
            for m in NUM_RE.findall(DATETIME_RE.sub(" ", v)):
                ok.add(_norm(m))

    for v in (scene.get("readings") or {}).values():
        if isinstance(v, list):
            for x in v:
                push(x)
        else:
            push(v)
    for f in scene.get("facts", []):
        push(f.get("v"))
        push(f.get("k"))
    for r in scene.get("routes", []):
        push(r.get("length_m"))
        push(r.get("label"))
    for key in ("title", "subtitle"):
        push(scene.get(key, ""))
    for s in scene.get("sources", []):
        push(s)
    for m in scene.get("markers", []):
        push(m.get("dist_km"))
        push(m.get("label"))
    for s in scene.get("shelters", []):
        for v in s.values():
            push(v)
    if scene.get("disaster_msg"):
        push(scene["disaster_msg"].get("text", ""))
        push(scene["disaster_msg"].get("at", ""))
        # 문자 글자수 배지
        ok.add(_norm(len(scene["disaster_msg"].get("text", ""))))
    for t in scene.get("timeline", []):
        for v in t.values():
            push(v)
    # 침수 폴리곤 개수 · 대피소 개수처럼 세어서 나오는 값
    push(len(scene.get("flood_polys", [])))
    push(len(scene.get("markers", [])))
    push(len([m for m in scene.get("markers", []) if m.get("kind") == "shelter"]))
    push(len([m for m in scene.get("markers", []) if m.get("kind") == "blocked"]))
    return ok


def _norm(v):
    s = str(v).replace(",", "")
    if s.endswith(".0"):
        s = s[:-2]
    return s


def check_scene(path, rep):
    name = os.path.basename(path)
    with open(path, encoding="utf-8") as f:
        sc = json.load(f)

    # C 커버리지
    if not sc.get("coverage_ok", True):
        for p in sc.get("coverage_problems", []):
            rep.add("FAIL", "C-coverage", name, p)
    L = sc.get("layers", {})
    if len(L.get("roads") or []) < 20:
        rep.add("FAIL", "C-coverage", name,
                f"도로 {len(L.get('roads') or [])}개 — 배경 지도가 비어 보인다. "
                f"장면 bbox 와 원본 데이터 범위가 어긋났을 가능성이 높다.")

    # D 출처
    prov = (sc.get("provenance") or {}).get("files") or []
    if not prov:
        rep.add("FAIL", "D-source", name, "provenance.files 가 비어 있다. "
                                          "출처를 추적할 수 없는 장면은 쓰지 않는다.")
    root = _project_root()
    for f in prov:
        if not os.path.exists(os.path.join(root, f)):
            rep.add("WARN", "D-source", name, f"원본 파일이 없다: {f}")
    if not sc.get("sources"):
        rep.add("FAIL", "D-source", name, "sources 가 비어 있어 아트보드에 "
                                          "출처를 찍을 수 없다.")

    # 앱 장면은 readings 없이는 숫자를 그릴 수 없다
    if sc.get("kind") == "app" and not sc.get("readings"):
        rep.add("FAIL", "A-trace", name,
                "app 장면인데 readings 가 없다. 렌더러가 숫자를 지어내게 된다.")
    return sc


# ════════════════════════════════════════════════════════════════════
# 아트보드 검사
# ════════════════════════════════════════════════════════════════════

def check_asset(path, scene, rep, allow=None):
    name = os.path.basename(path)
    src = open(path, encoding="utf-8").read()
    m = SVG_SIZE_RE.search(src)
    if not m:
        rep.add("FAIL", "B-bounds", name, "SVG 크기를 읽을 수 없다.")
        return
    W, Hh = int(m.group(1)), int(m.group(2))

    # B 경계 — 텍스트 좌표가 아트보드 밖
    out = [(x, y) for x, y in
           ((float(a), float(b)) for a, b in XY_RE.findall(src))
           if x < -40 or x > W + 40 or y < -20 or y > Hh + 12]
    if out:
        rep.add("FAIL", "B-bounds", name,
                f"아트보드({W}×{Hh}) 밖의 텍스트 {len(out)}개. "
                f"예: {out[:3]} — 시트 높이나 목록 개수를 다시 계산하라.")

    # E 형태 — 캡슐이 타원으로 퍼졌는지
    for w, h, rx in RECT_RE.findall(src):
        w, h, rx = float(w), float(h), float(rx)
        if rx > min(w, h) / 2 + 0.6:
            rep.add("FAIL", "E-shape", name,
                    f"rx={rx} 가 {w:.0f}×{h:.0f} 사각형의 절반을 넘는다 — "
                    f"캡슐이 아니라 타원으로 렌더된다. svgkit.rect 의 클램프 확인.")
            break

    # A 추적성 — 화면에 뜬 숫자가 장면 데이터에서 유래하는가
    if scene is not None:
        ok = known_numbers(scene) | set(allow or [])
        strays = []
        for t in TEXT_RE.findall(src):
            txt = DATETIME_RE.sub(" ", re.sub(r"<[^>]+>", "", t))
            for num in NUM_RE.findall(txt):
                if _norm(num) not in ok:
                    strays.append((num, txt.strip()[:34]))
        # 같은 값이 반복되면 한 번만 보고한다
        seen, uniq = set(), []
        for n, ctx in strays:
            if n in seen:
                continue
            seen.add(n)
            uniq.append((n, ctx))
        if uniq:
            rep.add("FAIL", "A-trace", name,
                    f"장면 데이터에 없는 숫자 {len(uniq)}종이 화면에 있다: "
                    + ", ".join(f"'{n}'(…{c}…)" for n, c in uniq[:6])
                    + " — 하드코딩했거나 readings 에 넣지 않은 값이다.")

    # 출처 표기가 아트보드에 실제로 찍혔는가
    if "출처" not in src and scene is not None and scene.get("kind") == "map":
        rep.add("WARN", "D-source", name,
                "아트보드에 '출처' 문구가 없다. source_note() 를 붙여라.")


def _project_root(start=None):
    p = os.path.abspath(start or os.getcwd())
    for _ in range(8):
        if os.path.isdir(os.path.join(p, "data")):
            return p
        nxt = os.path.dirname(p)
        if nxt == p:
            break
        p = nxt
    return os.getcwd()


# ════════════════════════════════════════════════════════════════════

def _match_scene(asset_path, scenes):
    """아트보드 파일명에서 장면을 추정한다."""
    b = os.path.basename(asset_path)
    for sid, sc in scenes.items():
        if sid in b:
            return sc
    if b.startswith("app-") or "slide-app" in b:
        return next((sc for sid, sc in scenes.items()
                     if sc.get("kind") == "app"), None)
    return None


def main():
    a = sys.argv[1:]
    scene_dir = a[0] if a and not a[0].startswith("--") else "scenes"
    asset_dir = a[1] if len(a) > 1 and not a[1].startswith("--") else "out"

    rep = Report()
    print("■ 장면 검사")
    scenes = {}
    for p in sorted(glob.glob(os.path.join(scene_dir, "*.json"))):
        sc = check_scene(p, rep)
        scenes[sc["id"]] = sc
    print(f"  장면 {len(scenes)}개")

    print("■ 아트보드 검사")
    assets = sorted(glob.glob(os.path.join(asset_dir, "*.svg")))
    for p in assets:
        check_asset(p, _match_scene(p, scenes), rep)
    print(f"  아트보드 {len(assets)}개")

    print("\n■ 결과")
    rep.print()
    print(f"\n  FAIL {len(rep.fails)}  ·  WARN {len(rep.warns)}")
    return 1 if rep.fails else 0


if __name__ == "__main__":
    sys.exit(main())
