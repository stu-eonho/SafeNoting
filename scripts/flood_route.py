# -*- coding: utf-8 -*-
"""
침수 위험구역 회피 경로 — 침수흔적도의 본래 용도

실제 상황은 이렇다.
    ① 지금 내가 있는 곳이 침수 위험 지역이다
    ② 대피소로 가려고 지도를 켠다
    ③ 그런데 대피 경로 한가운데 또 침수 구역이 있다
    ④ 그걸 피해 돌아가야 한다

카카오맵·네이버지도는 ③을 모른다. 침수 이력이 지도에 없기 때문이다.
행정안전부 침수흔적도는 바로 이걸 위해 존재하는 데이터다.

■ 지하차도 차단과의 차이
  지하차도는 '점'이지만 침수구역은 '면'이다.
  면을 통과하는 모든 도로에 비용을 매겨야 우회가 발생한다.

osmnx는 이 환경에서 Overpass 연결이 불안정해(curl은 1초, osmnx는 타임아웃)
Overpass QL을 직접 호출한다.

사용:
    python scripts/flood_route.py
"""
import sys, os, json, math, re, time, pickle, subprocess, tempfile

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(DATA, "routing")
EP = "https://overpass-api.de/api/interpreter"

# 침수흔적 밀집 지역 (1km 격자당 50건 이상, 평균 수심 1~2m)
BBOX = (36.620, 36.690, 127.440, 127.530)      # south, north, west, east — 청주 도심(무심천)
ROADS = ("motorway|trunk|primary|secondary|tertiary|unclassified|"
         "residential|living_street|motorway_link|trunk_link|primary_link|"
         "secondary_link|tertiary_link")
FLOOD_BUFFER_M = 60          # 침수 폴리곤에서 이 거리 안의 도로는 침수 위험으로 본다


def overpass(query, tries=3):
    for i in range(tries):
        fd, path = tempfile.mkstemp(suffix=".ql", text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(query)
            r = subprocess.run(["curl", "-s", "-m", "180", "-X", "POST",
                                "--data-binary", f"@{path}", EP],
                               capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            if r.returncode == 0 and r.stdout.strip().startswith("{"):
                return json.loads(r.stdout)
        except Exception:
            pass
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        if i < tries - 1:
            time.sleep(6 + i * 6)
    return None


def m3857(x, y):
    lon = x / 20037508.34 * 180.0
    lat = y / 20037508.34 * 180.0
    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
    return lat, lon


def load_flood_polys():
    """침수흔적 폴리곤을 '면'으로 유지한다. 중심점으로 뭉개면 구역 회피가 안 된다."""
    p = os.path.join(DATA, "osong_2023", "fludmarks_chungbuk.json")
    s, n, w, e = BBOX
    out = []
    for f in json.load(open(p, encoding="utf-8")):
        nums = re.findall(r'(\d{6,9}\.?\d*)\s+(\d{6,9}\.?\d*)', f.get("GEOM") or "")
        if len(nums) < 3:
            continue
        ring = [m3857(float(a), float(b)) for a, b in nums]
        cy = sum(q[0] for q in ring) / len(ring)
        cx = sum(q[1] for q in ring) / len(ring)
        if not (s <= cy <= n and w <= cx <= e):
            continue
        try:
            dep = float(f.get("FLDN_DOWA") or 0)
        except ValueError:
            dep = 0.0
        # ⚠ 침수심에 47m 같은 값이 섞여 있다(중앙값 0.49m).
        #   물리적으로 불가능하므로 데이터 오류로 보고 상한을 둔다.
        #   그대로 쓰면 "수심 46m" 같은 문구가 그림에 박혀 신뢰를 잃는다.
        if dep > 5.0:
            dep = 0.0
        out.append(dict(ring=ring, cy=cy, cx=cx, depth=dep,
                        year=f.get("FLDN_YR"), cause=f.get("FLDN_CS_DTL_NM")))
    return out


def build_graph():
    import networkx as nx
    s, n, w, e = BBOX
    print("■ 도로망 수집 (Overpass 직접)")
    t0 = time.time()
    d = overpass(f'[out:json][timeout:180];'
                 f'way({s},{w},{n},{e})["highway"~"{ROADS}"];out geom tags;')
    if not d:
        raise SystemExit("도로망 수집 실패")
    G = nx.Graph()
    nun = 0
    for el in d.get("elements", []):
        g = el.get("geometry")
        if not g or len(g) < 2:
            continue
        t = el.get("tags") or {}
        tun = t.get("tunnel")
        lay = t.get("layer")
        is_und = tun in ("yes", "building_passage")
        try:
            is_und = is_und or float(lay) < 0
        except (TypeError, ValueError):
            pass
        if is_und:
            nun += 1
        nm = t.get("name")
        for a, b in zip(g[:-1], g[1:]):
            u = (round(a["lat"], 7), round(a["lon"], 7))
            v = (round(b["lat"], 7), round(b["lon"], 7))
            if u == v:
                continue
            L = math.hypot((u[0] - v[0]) * 111000, (u[1] - v[1]) * 88800)
            if G.has_edge(u, v):
                continue
            G.add_edge(u, v, length=L, name=nm, underpass=is_und)
    print(f"  노드 {G.number_of_nodes():,} · 엣지 {G.number_of_edges():,} "
          f"· 지하차도 way {nun}개  ({time.time()-t0:.0f}s)")
    return G


def mark_flood(G, polys):
    """각 엣지에 '침수구역 포함 여부'와 예상 수심을 붙인다."""
    import numpy as np
    if not polys:
        return 0
    cy = np.array([p["cy"] for p in polys])
    cx = np.array([p["cx"] for p in polys])
    dep = np.array([p["depth"] for p in polys])
    hit = 0
    for u, v, d in G.edges(data=True):
        my = (u[0] + v[0]) / 2
        mx = (u[1] + v[1]) / 2
        dm = np.hypot((cy - my) * 111000, (cx - mx) * 88800)
        i = int(np.argmin(dm))
        dist = float(dm[i])
        d["flood_dist"] = dist
        if dist <= FLOOD_BUFFER_M:
            d["flooded"] = True
            d["flood_depth"] = float(dep[i])
            hit += 1
        else:
            d["flooded"] = False
            d["flood_depth"] = 0.0
    return hit


def main():
    import networkx as nx
    os.makedirs(OUT, exist_ok=True)

    polys = load_flood_polys()
    print(f"■ 대상 범위 침수흔적 {len(polys)}건")
    dep = [p["depth"] for p in polys if p["depth"] > 0]
    if dep:
        print(f"  수심 중앙값 {sorted(dep)[len(dep)//2]:.2f}m · 최대 {max(dep):.2f}m")

    G = build_graph()
    hit = mark_flood(G, polys)
    print(f"■ 침수구역({FLOOD_BUFFER_M}m 이내) 도로 {hit:,}개 / 전체 {G.number_of_edges():,}"
          f"  ({hit/G.number_of_edges()*100:.1f}%)")

    # 비용: 침수구역은 통행 불가로 본다. 물이 찬 길은 '느린 길'이 아니라 '못 가는 길'이다.
    INF = 1e9
    for u, v, d in G.edges(data=True):
        d["w_short"] = d["length"]
        d["w_safe"] = INF if (d["flooded"] or d["underpass"]) else d["length"]

    # ── 대피소 후보 ─────────────────────────────────────────
    # 행정안전부 지정 대피소를 침수 안전성으로 재평가한 결과를 쓴다.
    # 지정 대피소에는 '침수' 분류가 없어, 야외 운동장이나 저지대 은행이 섞여 있다.
    # (274곳 중 26%가 침수 시 부적합)
    rp = os.path.join(OUT, "shelters_ranked.json")
    if not os.path.exists(rp):
        raise SystemExit("shelters_ranked.json 없음 — rank_shelters.py 먼저 실행")
    allsh = json.load(open(rp, encoding="utf-8"))
    shelters = [x for x in allsh if x["score"] >= 40 and not x["outdoor"]]
    unsafe = [x for x in allsh if x["score"] < 0 or x["outdoor"]]
    print(f"■ 대피소 {len(allsh)}곳 → 침수 안전 {len(shelters)}곳 / 부적합 {len(unsafe)}곳")

    def nearest_node(lat, lon):
        best, bd = None, 9e18
        for n in G.nodes():
            dd = (n[0] - lat) ** 2 * 12321 + (n[1] - lon) ** 2 * 7885
            if dd < bd:
                bd, best = dd, n
        return best, math.sqrt(bd) * 1000

    shelter_nodes = []
    for sh in shelters:
        nd, dist = nearest_node(sh["lat"], sh["lon"])
        if nd is not None and dist < 400:
            shelter_nodes.append((nd, sh))
    print(f"  도로망에 연결된 대피소 {len(shelter_nodes)}곳")

    # 침수구역을 통과해야만 하는 출발-도착 쌍을 찾는다
    flooded_edges = [(u, v) for u, v, d in G.edges(data=True) if d["flooded"]]
    print(f"■ 우회 사례 탐색 (침수 도로 {len(flooded_edges):,}개 기준)")
    cases = []
    seen_start = set()
    for u, v in flooded_edges[:400]:
        if u in seen_start:
            continue
        try:
            ring_u = nx.single_source_dijkstra_path_length(G, u, cutoff=700, weight="length")
            ring_v = nx.single_source_dijkstra_path_length(G, v, cutoff=1800, weight="length")
        except Exception:
            continue
        cs = [x for x, dd in ring_u.items() if 300 <= dd <= 700]
        # 목적지는 임의 노드가 아니라 실제 대피소로 한정한다.
        # "어디로 가는지"가 그림에 나와야 대피 안내로 읽힌다.
        cd = [nd for nd, _ in shelter_nodes if nd in ring_v] or \
             [x for x, dd in ring_v.items() if 300 <= dd <= 700]
        shmap = {nd: sh for nd, sh in shelter_nodes}
        for s in cs[:6]:
            for t in cd[:6]:
                if s == t:
                    continue
                try:
                    p1 = nx.shortest_path(G, s, t, weight="w_short")
                    L1 = nx.shortest_path_length(G, s, t, weight="w_short")
                except nx.NetworkXNoPath:
                    continue
                nfl = sum(1 for a, b in zip(p1[:-1], p1[1:]) if G[a][b]["flooded"])
                if nfl < 2:
                    continue
                try:
                    p2 = nx.shortest_path(G, s, t, weight="w_safe")
                    L2 = nx.shortest_path_length(G, s, t, weight="w_safe")
                except nx.NetworkXNoPath:
                    continue
                if L2 >= INF / 10:
                    continue
                det = (L2 - L1) / max(L1, 1)
                if not (0.05 <= det <= 1.2) or L1 < 400:
                    continue
                straight = math.hypot((s[0] - t[0]) * 111000, (s[1] - t[1]) * 88800)
                if straight < 250 or L1 / max(straight, 1) > 2.2:
                    continue
                depths = [G[a][b]["flood_depth"] for a, b in zip(p1[:-1], p1[1:])
                          if G[a][b]["flooded"]]
                sh = shmap.get(t)
                cases.append(dict(
                    L1=L1, L2=L2, detour=det, n_flood=nfl,
                    max_depth=max(depths) if depths else 0,
                    shelter=sh["name"] if sh else None,
                    shelter_kind=sh.get("kind") if sh else None,
                    short=[[a[0], a[1]] for a in p1],
                    safe=[[a[0], a[1]] for a in p2]))
        seen_start.add(u)
        if len(cases) >= 60:
            break

    if not cases:
        print("  조건을 만족하는 사례 없음")
        return
    # 침수 도로를 많이 지나면서 우회가 과하지 않은 사례
    # 경로 주변 대피소 밀도를 센다. 외곽 지역이 뽑히면 배경이 비어
    # "지도"가 아니라 "선 몇 개"로 보인다.
    import numpy as _np
    _sy = _np.array([x["lat"] for x in allsh]); _sx = _np.array([x["lon"] for x in allsh])
    for _c in cases:
        _my = sum(p[0] for p in _c["short"]) / len(_c["short"])
        _mx = sum(p[1] for p in _c["short"]) / len(_c["short"])
        _d = _np.hypot((_sy - _my) * 111000, (_sx - _mx) * 88800)
        _c["density"] = int((_d < 1200).sum())

    # ① 대피소가 명시되고 ② 주변이 시가지이며 ③ 침수도로를 많이 지나는 사례
    cases.sort(key=lambda z: (z["shelter"] is None, -z["density"],
                              -z["n_flood"], abs(z["detour"] - 0.3)))
    print(f"  사례 {len(cases)}건 · 상위 5건")
    for c in cases[:5]:
        print(f"    최단 {c['L1']:>6.0f}m (침수도로 {c['n_flood']:>2}개, 수심 "
              f"{c['max_depth']:.2f}m) → 안전 {c['L2']:>6.0f}m (+{c['detour']*100:>5.1f}%)"
              f"  주변대피소 {c.get('density',0):>2}곳  {c['shelter'] or '(미지정)'}")

    json.dump(dict(bbox=BBOX, buffer_m=FLOOD_BUFFER_M,
                   shelters_safe=shelters, shelters_unsafe=unsafe,
                   polys=[dict(ring=p["ring"], depth=p["depth"], year=p["year"])
                          for p in polys],
                   cases=cases[:10]),
              open(os.path.join(OUT, "flood_route.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    pickle.dump(G, open(os.path.join(OUT, "graph_cheongju.pkl"), "wb"))
    print(f"\n  → data/routing/flood_route.json")


if __name__ == "__main__":
    main()
