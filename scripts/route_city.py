# -*- coding: utf-8 -*-
"""
도시별 지하차도 우회 사례 탐색 — 여러 도시에서 최적 사례를 자동으로 고른다

오송(농촌)에서는 도로망이 성겨 우회가 +5m(그림에 안 보임) 아니면
+2619m(메시지가 반대) 양극단만 나왔다. 도심은 도로망이 조밀해
"조금 돌아가면 지하차도를 피할 수 있다"는 사례가 나올 가능성이 높다.

대피소 데이터가 없는 도시에서도 동작하도록, 목적지를 대피소로 한정하지 않고
**지하차도 반대편 지점**으로 잡는다. 증명하려는 명제는 목적지 종류와 무관하다.
    "침수 시 지하차도는 가장 빠른 길이자 가장 위험한 길이다."

사용:
    python scripts/route_city.py busan_choryang
    python scripts/route_city.py            # 가능한 모든 도시
"""
import sys, os, json, math, pickle

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data", "routing")
INF = 1e9
DETOUR_MIN, DETOUR_MAX = 0.05, 1.00      # 우회 5~100%가 그림과 메시지 모두 성립하는 구간


def km(a, b):
    return math.hypot((a[0] - b[0]) * 111.0, (a[1] - b[1]) * 88.8)


def edge_names(d):
    n = d.get("name")
    if isinstance(n, str):
        return [n]
    return list(n) if n else []


def path_len(G, path):
    t = 0.0
    for u, v in zip(path[:-1], path[1:]):
        t += float(min(G[u][v].values(), key=lambda x: x.get("length", 1e9)).get("length", 0))
    return t


def _up(d):
    """지하차도 엣지 판정. 그래프마다 속성명이 다르다 (청주 underpass, 오송·부산 _underpass)."""
    return (str(d.get("_underpass", "")).lower() == "true"
            or str(d.get("underpass", "")).lower() == "true")


def count_underpass(G, path):
    names = []
    for u, v in zip(path[:-1], path[1:]):
        d = min(G[u][v].values(), key=lambda x: x.get("length", 1e9))
        if _up(d):
            nm = edge_names(d)
            names.append(nm[-1] if nm else "(무명 지하차도)")
    return names


def analyze(key, label, note):
    import networkx as nx
    p = os.path.join(OUT, f"graph_{key}.pkl")
    if not os.path.exists(p):
        return []
    G = pickle.load(open(p, "rb"))
    if not G.is_multigraph():            # 청주 그래프는 nx.Graph 로 저장돼 있다
        G = nx.MultiDiGraph(G.to_directed())
    und = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) if _up(d)]
    print(f"\n■ {label} — 노드 {G.number_of_nodes():,} · 지하차도 {len(und)}개")

    for u, v, k, d in G.edges(keys=True, data=True):
        d["_w"] = INF if _up(d) else float(d.get("length", 1.0))

    found = []
    for u, v, k, d in und:
        nm = edge_names(d)
        uname = nm[-1] if nm else "(무명 지하차도)"
        # 지하차도 양 끝에서 각각 바깥으로 몇 홉 떨어진 지점을 출발/도착으로 삼는다.
        # 지하차도 바로 앞뒤를 쓰면 우회로가 없어 비교가 성립하지 않는다.
        try:
            src_ring = nx.single_source_dijkstra_path_length(G, u, cutoff=900, weight="length")
            dst_ring = nx.single_source_dijkstra_path_length(G, v, cutoff=900, weight="length")
        except Exception:
            continue
        cands_s = [n for n, dd in src_ring.items() if 400 <= dd <= 900]
        cands_d = [n for n, dd in dst_ring.items() if 400 <= dd <= 900]
        for s in cands_s[:12]:
            for t in cands_d[:12]:
                if s == t:
                    continue
                try:
                    p1 = nx.shortest_path(G, s, t, weight="length")
                except nx.NetworkXNoPath:
                    continue
                un1 = count_underpass(G, p1)
                if not un1:
                    continue
                try:
                    p2 = nx.shortest_path(G, s, t, weight="_w")
                except nx.NetworkXNoPath:
                    continue
                un2 = count_underpass(G, p2)
                if un2:
                    continue
                L1, L2 = path_len(G, p1), path_len(G, p2)
                if L2 >= INF / 10 or L1 < 300:
                    continue
                det = (L2 - L1) / L1
                if not (DETOUR_MIN <= det <= DETOUR_MAX):
                    continue
                # 직선거리 대비 경로가 지나치게 길면 "왜 저렇게 도나" 싶은 그림이 된다.
                # 우회로의 존재가 아니라 도로망 구조 탓으로 보이므로 제외한다.
                sy, sx = G.nodes[s]["y"], G.nodes[s]["x"]
                ty, tx = G.nodes[t]["y"], G.nodes[t]["x"]
                straight = km((sy, sx), (ty, tx)) * 1000
                if straight < 250 or L1 / max(straight, 1) > 2.0:
                    continue
                found.append(dict(city=key, label=label, note=note, underpass=uname,
                                  detour=det, L1=L1, L2=L2,
                                  short=[[G.nodes[n]["y"], G.nodes[n]["x"]] for n in p1],
                                  safe=[[G.nodes[n]["y"], G.nodes[n]["x"]] for n in p2],
                                  un1=un1))
    # 이름 있는 지하차도를 우선. 그림에 "(무명 지하차도)"가 들어가면 설득력이 떨어진다.
    found.sort(key=lambda z: (z["underpass"].startswith("("), abs(z["detour"] - 0.25)))
    print(f"  조건 충족 사례 {len(found)}건 (우회 {DETOUR_MIN*100:.0f}~{DETOUR_MAX*100:.0f}%)")
    for f in found[:4]:
        print(f"    {f['underpass']:<18} 최단 {f['L1']:>6.0f}m → 안전 {f['L2']:>6.0f}m "
              f"(+{f['detour']*100:>5.1f}%)")
    return found[:6]


def main():
    # cities.json은 수집이 끝나야 저장된다. 아직 없을 수 있으므로
    # 실제로 존재하는 graph_*.pkl 에서 도시 목록을 직접 추론한다.
    import glob
    KNOWN = {
        "osong": ("청주 오송", "2023-07-15 궁평2지하차도 14명 사망"),
        "busan_choryang": ("부산 초량", "2020-07-23 초량지하차도 침수 3명 사망"),
        "seoul_gangnam": ("서울 강남", "2022-08-08 집중호우 침수"),
        "daejeon": ("대전 도심", "지하차도 다수"),
    }
    cities = {}
    for f in glob.glob(os.path.join(OUT, "graph_*.pkl")):
        key = os.path.basename(f)[6:-4]
        nm, note = KNOWN.get(key, (key, ""))
        cities[key] = dict(name=nm, note=note)
    if not cities:
        print("graph_*.pkl 이 없다 — build_road_graph.py 먼저 실행")
        return
    print(f"■ 대상 도시 {len(cities)}개: {', '.join(c['name'] for c in cities.values())}")

    only = sys.argv[1] if len(sys.argv) > 1 else None
    allf = []
    for key, c in cities.items():
        if only and key != only:
            continue
        allf += analyze(key, c.get("name", key), c.get("note", ""))

    if not allf:
        print("\n조건을 만족하는 사례가 없다")
        return
    allf.sort(key=lambda z: (z["underpass"].startswith("("), abs(z["detour"] - 0.25)))
    best = allf[0]
    print(f"\n■ 전체 최적 사례")
    print(f"  {best['label']} · {best['underpass']}")
    print(f"  최단 {best['L1']:.0f}m (지하차도 {len(best['un1'])}개 통과)")
    print(f"  안전 {best['L2']:.0f}m (지하차도 0개)")
    print(f"  → +{best['L2']-best['L1']:.0f}m ({best['detour']*100:+.0f}%)")
    json.dump(allf, open(os.path.join(OUT, "city_cases.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n  → data/routing/city_cases.json ({len(allf)}건)")


if __name__ == "__main__":
    main()
