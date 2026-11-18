# -*- coding: utf-8 -*-
"""
지도 맥락 데이터 수집 — 건물·랜드마크·수계 (Overpass 직접 호출)

도로만 그리면 선의 집합일 뿐이라 "여기가 어디인지" 읽히지 않는다.
건물 배치와 랜드마크가 있어야 대피 경로가 지나는 곳이 어떤 동네인지 보인다.

■ osmnx를 쓰지 않는 이유
  같은 Overpass 서버에 curl은 1초 만에 응답하는데 osmnx는 ConnectTimeout이 났다.
  (rate limit 여유 2슬롯, status 엔드포인트 정상 응답 확인)
  원인 규명보다 우회가 빠르므로 Overpass QL을 직접 POST한다.

사용:
    python scripts/fetch_context.py busan_choryang
"""
import sys, os, json, time, subprocess, tempfile

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data", "routing")
EP = "https://overpass-api.de/api/interpreter"

BBOX = {   # south, north, west, east
    "busan_choryang": (35.095, 35.135, 129.020, 129.060),
    "osong":          (36.58, 36.68, 127.28, 127.40),
    "seoul_gangnam":  (37.480, 37.515, 127.010, 127.050),
    "daejeon":        (36.315, 36.355, 127.375, 127.425),
    "cheongju":       (36.620, 36.690, 127.440, 127.530),
}

# 대피와 관련된 시설만. 상점·카페까지 넣으면 라벨이 뭉개진다.
AMENITY = ("school|university|college|hospital|clinic|townhall|police|"
           "fire_station|community_centre|kindergarten|public_building")


def overpass(query, tries=3):
    """curl로 POST. requests/urllib이 이 환경에서 불안정해 검증된 경로를 쓴다."""
    for i in range(tries):
        fd, path = tempfile.mkstemp(suffix=".ql", text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(query)
            r = subprocess.run(
                ["curl", "-s", "-m", "180", "-X", "POST",
                 "--data-binary", f"@{path}", EP],
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode == 0 and r.stdout.strip().startswith("{"):
                return json.loads(r.stdout)
            err = (r.stderr or r.stdout)[:80]
        except Exception as e:
            err = str(e)[:80]
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        if i < tries - 1:
            print(f"    재시도 {i+1} ({err})", flush=True)
            time.sleep(6 + i * 6)
    return None


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "busan_choryang"
    if key not in BBOX:
        raise SystemExit(f"알 수 없는 지역: {key}")
    s, n, w, e = BBOX[key]
    B = f"{s},{w},{n},{e}"
    print(f"■ {key} 맥락 데이터 수집 (Overpass 직접)")
    out = {}

    # ── 건물 ────────────────────────────────────────────────
    print("  건물...", flush=True)
    t0 = time.time()
    d = overpass(f"[out:json][timeout:180];way({B})[\"building\"];out geom;")
    polys = []
    if d:
        for el in d.get("elements", []):
            g = el.get("geometry")
            if not g or len(g) < 3:
                continue
            polys.append([[p["lat"], p["lon"]] for p in g])
    out["buildings"] = polys
    print(f"    {len(polys):,}동  ({time.time()-t0:.0f}s)")

    # ── 랜드마크 ────────────────────────────────────────────
    print("  랜드마크...", flush=True)
    t0 = time.time()
    q = (f'[out:json][timeout:180];('
         f'node({B})["amenity"~"{AMENITY}"];'
         f'way({B})["amenity"~"{AMENITY}"];'
         f'node({B})["railway"="station"];'
         f'way({B})["leisure"~"park|sports_centre|stadium"];'
         f');out center tags;')
    d = overpass(q)
    marks, seen = [], set()
    if d:
        for el in d.get("elements", []):
            t = el.get("tags") or {}
            nm = t.get("name")
            if not nm or nm in seen:
                continue
            lat = el.get("lat") or (el.get("center") or {}).get("lat")
            lon = el.get("lon") or (el.get("center") or {}).get("lon")
            if lat is None or lon is None:
                continue
            seen.add(nm)
            marks.append(dict(name=nm, lat=float(lat), lon=float(lon),
                              kind=t.get("amenity") or t.get("railway") or t.get("leisure")))
    out["landmarks"] = marks
    print(f"    {len(marks)}곳  ({time.time()-t0:.0f}s)")

    # ── 수계 ────────────────────────────────────────────────
    print("  하천·수역...", flush=True)
    t0 = time.time()
    d = overpass(f'[out:json][timeout:180];('
                 f'way({B})["natural"="water"];'
                 f'way({B})["waterway"~"river|stream"];'
                 f');out geom;')
    lines, areas = [], []
    if d:
        for el in d.get("elements", []):
            g = el.get("geometry")
            if not g or len(g) < 2:
                continue
            pts = [[p["lat"], p["lon"]] for p in g]
            t = el.get("tags") or {}
            if t.get("natural") == "water":
                areas.append(pts)
            else:
                lines.append(pts)
    out["water_lines"] = lines
    out["water_areas"] = areas
    print(f"    하천 {len(lines)}개 · 수역 {len(areas)}개  ({time.time()-t0:.0f}s)")

    p = os.path.join(OUT, f"context_{key}.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\n  → {p}  ({os.path.getsize(p)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
