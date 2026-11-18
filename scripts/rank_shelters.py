# -*- coding: utf-8 -*-
"""
지정 대피소의 침수 안전성 재평가

행정안전부 통합대피소를 받아보면 이렇다.
    한파쉼터·무더위쉼터  230곳  (은행 지점, 새마을금고, 신협…)
    지진옥외대피장소      44곳  (야외 공터)
    침수·홍수 대피소       0곳  ← 분류 자체가 없다

즉 **침수 대피소는 제도적으로 존재하지 않는다.**
그 결과 침수 상황에서 지정 대피소를 그대로 안내하면
  · 은행 1층 지점 → 물이 차오르면 오히려 위험
  · 지진옥외대피장소 → 야외라 폭우 중에는 무용지물
이 된다.

그래서 다음 세 축으로 재평가한다.
    ① 표고        — 주변보다 높은가 (SRTM DEM)
    ② 침수 이력    — 과거 침수 구역에서 얼마나 떨어져 있는가
    ③ 시설 유형    — 실내인가 야외인가

청주 도심 274곳 기준 결과: 야외 시설 44곳, 침수흔적 200m 이내 31곳, 두 조건 중복 3곳
→ 침수 대피에 부적합 72곳 (26%).

사용:
    python scripts/rank_shelters.py
"""
import sys, os, json, math, re, time, urllib.parse, urllib.request, ssl

sys.stdout.reconfigure(encoding='utf-8')
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(DATA, "routing")

OUTDOOR = ("지진옥외대피장소", "지진해일긴급대피장소")


def m3857(x, y):
    lon = x / 20037508.34 * 180.0
    lat = y / 20037508.34 * 180.0
    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
    return lat, lon


def load_flood_centers():
    p = os.path.join(DATA, "osong_2023", "fludmarks_chungbuk.json")
    out = []
    for f in json.load(open(p, encoding="utf-8")):
        nums = re.findall(r'(\d{6,9}\.?\d*)\s+(\d{6,9}\.?\d*)', f.get("GEOM") or "")
        if len(nums) < 3:
            continue
        ring = [m3857(float(a), float(b)) for a, b in nums]
        cy = sum(q[0] for q in ring) / len(ring)
        cx = sum(q[1] for q in ring) / len(ring)
        try:
            dep = float(f.get("FLDN_DOWA") or 0)
        except ValueError:
            dep = 0.0
        out.append((cy, cx, dep if dep <= 5.0 else 0.0))
    return out


def fetch_elev(points, chunk=100):
    """OpenTopoData SRTM 30m — 인증 불필요, 1회 100지점"""
    elevs = []
    for i in range(0, len(points), chunk):
        part = points[i:i + chunk]
        loc = "|".join(f"{a:.6f},{b:.6f}" for a, b in part)
        url = ("https://api.opentopodata.org/v1/srtm30m?locations="
               + urllib.parse.quote(loc, safe="|,"))
        got = None
        for t in range(3):
            try:
                with urllib.request.urlopen(url, timeout=90, context=CTX) as r:
                    d = json.loads(r.read().decode())
                if d.get("status") == "OK":
                    got = [x["elevation"] for x in d["results"]]
                    break
            except Exception:
                time.sleep(3 + t * 3)
        elevs += got if got else [None] * len(part)
        time.sleep(1.1)
        print(f"    표고 {min(i+chunk, len(points))}/{len(points)}", flush=True)
    return elevs


def main():
    sh = json.load(open(os.path.join(OUT, "shelters_area.json"), encoding="utf-8"))
    print(f"■ 지정 대피소 {len(sh)}곳 재평가")

    fl = load_flood_centers()
    print(f"  침수흔적 {len(fl):,}건 대조")

    print("  표고 조회 (SRTM 30m)")
    elevs = fetch_elev([(x["lat"], x["lon"]) for x in sh])

    import numpy as np
    fy = np.array([p[0] for p in fl]); fx = np.array([p[1] for p in fl])
    fd = np.array([p[2] for p in fl])
    ok = [e for e in elevs if e is not None]
    base = float(np.median(ok)) if ok else 0.0
    print(f"  지역 표고 중앙값 {base:.0f}m")

    for x, e in zip(sh, elevs):
        x["elev"] = e
        dm = np.hypot((fy - x["lat"]) * 111000, (fx - x["lon"]) * 88800)
        i = int(np.argmin(dm))
        x["flood_dist"] = float(dm[i])
        x["flood_depth"] = float(fd[i])
        x["outdoor"] = any(k in str(x.get("kind", "")) for k in OUTDOOR)

        # ── 점수 (높을수록 안전) ─────────────────────────────
        # 야외 시설은 폭우 중 대피처로 쓸 수 없다 → 큰 감점
        s_out = -40 if x["outdoor"] else 0
        # 표고: 지역 중앙값 대비
        s_ele = 0 if e is None else max(-25, min(35, (e - base) * 3.0))
        # 침수 이력 거리: 300m를 안전 기준선으로
        d = x["flood_dist"]
        s_fl = -30 if d < 100 else (-15 if d < 200 else (0 if d < 300 else 25))
        x["score"] = round(s_out + s_ele + s_fl, 1)

    ranked = sorted([x for x in sh if x["elev"] is not None],
                    key=lambda z: -z["score"])
    print(f"\n■ 침수 안전성 상위 8곳")
    print(f"  {'점수':>6} {'표고':>6} {'침수흔적':>9}  {'유형':<14}{'명칭'}")
    for x in ranked[:8]:
        print(f"  {x['score']:>6.1f} {x['elev']:>5.0f}m {x['flood_dist']:>8.0f}m  "
              f"{str(x['kind'])[:12]:<14}{str(x['name'])[:24]}")
    print(f"\n■ 침수 시 부적합 하위 6곳")
    for x in ranked[-6:]:
        why = []
        if x["outdoor"]:
            why.append("야외")
        if x["flood_dist"] < 200:
            why.append(f"침수흔적 {x['flood_dist']:.0f}m")
        if x["elev"] is not None and x["elev"] < base:
            why.append(f"저지대 {x['elev']:.0f}m")
        print(f"  {x['score']:>6.1f} {str(x['name'])[:24]:<26} {' · '.join(why)}")

    n_out = sum(1 for x in sh if x["outdoor"])
    n_near = sum(1 for x in sh if x["flood_dist"] < 200)
    print(f"\n■ 요약")
    print(f"  야외 시설(폭우 중 사용 불가)      {n_out}곳 ({n_out/len(sh)*100:.0f}%)")
    print(f"  침수흔적 200m 이내               {n_near}곳 ({n_near/len(sh)*100:.0f}%)")
    print(f"  둘 중 하나라도 해당              "
          f"{sum(1 for x in sh if x['outdoor'] or x['flood_dist'] < 200)}곳")

    json.dump(ranked, open(os.path.join(OUT, "shelters_ranked.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  → data/routing/shelters_ranked.json")


if __name__ == "__main__":
    main()
