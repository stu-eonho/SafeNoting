# -*- coding: utf-8 -*-
"""
오송 일대 도로망 그래프 구축 + 침수위험 가중치 부여

이 모듈이 답하려는 질문은 하나다.
    "가장 가까운 대피소가 가장 안전한 대피소인가?"

가까운 대피소 두 곳만 봐도 침수흔적과의 거리가 크게 다르다.
    탑연4경로당  0.83km   침수흔적까지 667m
    궁평1경로당  0.91km   침수흔적까지 152m  ← 비슷한 거리지만 침수흔적에 가깝다
최단거리만 계산하는 길찾기는 이 차이를 반영하지 않는다.

■ 비용 함수
    cost = 길이(m)
         × (지하차도면 ∞)              ← tunnel=yes / layer<0 하드 차단
         × (1 + α · 침수흔적 근접도)
         × (1 + β · 저지대 정도)        ← DEM 기반 (미구현: DEM 미확보)

  지하차도를 '가중치가 높은 길'이 아니라 '갈 수 없는 길'로 두는 것이 핵심이다.
  2023 궁평2지하차도에서 실제로 벌어진 일이 그것이었다.

사용:
    python scripts/build_road_graph.py
"""
import sys, os, json, math, re, pickle, time
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(DATA, "routing")

# 궁평2지하차도 (2023-07-15 참사 지점)
OSONG = (36.62250963, 127.34523676)
BBOX = (36.58, 36.68, 127.28, 127.40)      # south, north, west, east
FLOOD_INFLUENCE_M = 300                     # 침수흔적 영향 반경


def km(a, b):
    return math.hypot((a[0] - b[0]) * 111.0, (a[1] - b[1]) * 88.8)


def m3857_to_wgs(x, y):
    lon = x / 20037508.34 * 180.0
    lat = y / 20037508.34 * 180.0
    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
    return lat, lon


def load_flood_points():
    """침수흔적 폴리곤 → 대표점(중심) + 수심. 폴리곤이 작아 중심점으로 충분하다."""
    p = os.path.join(DATA, "osong_2023", "fludmarks_chungbuk.json")
    pts = []
    for f in json.load(open(p, encoding="utf-8")):
        nums = re.findall(r'(\d{6,9}\.?\d*)\s+(\d{6,9}\.?\d*)', f.get("GEOM") or "")
        if len(nums) < 3:
            continue
        ring = [m3857_to_wgs(float(a), float(b)) for a, b in nums]
        clat = sum(q[0] for q in ring) / len(ring)
        clon = sum(q[1] for q in ring) / len(ring)
        if not (BBOX[0] <= clat <= BBOX[1] and BBOX[2] <= clon <= BBOX[3]):
            continue
        try:
            depth = float(f.get("FLDN_DOWA") or 0)
        except ValueError:
            depth = 0.0
        pts.append((clat, clon, depth))
    return pts


def main():
    import osmnx as ox
    import networkx as nx
    os.makedirs(OUT, exist_ok=True)

    print("■ 1) OSM 도로망 다운로드")
    t0 = time.time()
    # 차량 도로망(drive)을 쓴다. 지하차도 차단이 핵심이고, 지하차도는 차량 도로에 있다.
    G = ox.graph_from_bbox(bbox=(BBOX[2], BBOX[0], BBOX[3], BBOX[1]),
                           network_type="drive", simplify=True)
    print(f"  노드 {G.number_of_nodes():,} · 엣지 {G.number_of_edges():,}  ({time.time()-t0:.0f}s)")

    print("\n■ 2) 지하차도 식별 (tunnel=yes / layer<0)")
    under = []
    for u, v, k, d in G.edges(keys=True, data=True):
        tun = d.get("tunnel")
        lay = d.get("layer")
        is_tunnel = (tun in ("yes", "building_passage") or
                     (isinstance(tun, list) and "yes" in tun))
        low = False
        try:
            low = float(lay if not isinstance(lay, list) else lay[0]) < 0
        except (TypeError, ValueError):
            pass
        if is_tunnel or low:
            under.append((u, v, k))
            d["_underpass"] = True
    print(f"  지하차도/터널 엣지 {len(under)}개")
    names = set()
    for u, v, k in under:
        n = G.edges[u, v, k].get("name")
        if n:
            names.add(n if isinstance(n, str) else n[0])
    print(f"  명칭: {sorted(names)[:8]}")

    print("\n■ 3) 침수흔적 근접도 계산")
    fp = load_flood_points()
    print(f"  대상 범위 내 침수흔적 {len(fp)}건")
    fl = np.array([[p[0], p[1]] for p in fp]) if fp else np.zeros((0, 2))
    fdep = np.array([p[2] for p in fp]) if fp else np.zeros(0)

    def flood_risk(lat, lon):
        """가장 가까운 침수흔적까지 거리와 그 수심으로 0~1 위험도"""
        if len(fl) == 0:
            return 0.0, None
        dm = np.hypot((fl[:, 0] - lat) * 111000.0, (fl[:, 1] - lon) * 88800.0)
        i = int(np.argmin(dm))
        dist = float(dm[i])
        if dist > FLOOD_INFLUENCE_M:
            return 0.0, dist
        # 가까울수록, 수심이 깊을수록 위험
        prox = 1.0 - dist / FLOOD_INFLUENCE_M
        depth_w = min(fdep[i] / 1.0, 2.0) if fdep[i] > 0 else 0.5
        return float(prox * depth_w), dist

    t0 = time.time()
    risky = 0
    for u, v, k, d in G.edges(keys=True, data=True):
        la = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
        lo = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
        r, dist = flood_risk(la, lo)
        d["_flood_risk"] = r
        d["_flood_dist"] = dist
        if r > 0:
            risky += 1
    print(f"  침수 영향권 엣지 {risky:,}개 / 전체 {G.number_of_edges():,}  ({time.time()-t0:.0f}s)")

    print("\n■ 4) 저장")
    with open(os.path.join(OUT, "graph_osong.pkl"), "wb") as f:
        pickle.dump(G, f)
    meta = dict(nodes=G.number_of_nodes(), edges=G.number_of_edges(),
                underpass=len(under), underpass_names=sorted(names),
                flood_points=len(fp), risky_edges=risky, bbox=BBOX)
    json.dump(meta, open(os.path.join(OUT, "graph_meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"  → data/routing/graph_osong.pkl")
    print(f"  → data/routing/graph_meta.json")


if __name__ == "__main__":
    main()
