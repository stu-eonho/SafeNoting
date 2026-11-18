# -*- coding: utf-8 -*-
"""
최단경로 vs 침수회피 안전경로 비교 — 예선 제출 이미지 ②

증명하려는 명제는 하나다.
    "가장 가까운 대피소로 가는 가장 짧은 길이, 가장 안전한 길은 아니다."

2023-07-15 궁평2지하차도에서 실제로 벌어진 일이 그것이다.
지하차도는 그 시각 '가장 빠른 길'이었고, 동시에 죽음의 길이었다.
카카오맵·네이버지도는 지금도 그 길을 안내한다. 최단거리만 계산하기 때문이다.

사용:
    python scripts/route_compare.py
"""
import sys, os, json, math, pickle
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(DATA, "routing")

OSONG = (36.62250963, 127.34523676)      # 궁평2지하차도
ALPHA = 4.0                               # 침수위험 가중
INF = 1e9


def km(a, b):
    return math.hypot((a[0] - b[0]) * 111.0, (a[1] - b[1]) * 88.8)


def nearest_node(G, lat, lon):
    """osmnx.nearest_nodes는 sklearn을 요구한다. 노드가 3천 개 수준이라 직접 찾는다."""
    best, bd = None, 9e18
    for n, nd in G.nodes(data=True):
        d = (nd["y"] - lat) ** 2 * 12321.0 + (nd["x"] - lon) ** 2 * 7885.0
        if d < bd:
            bd, best = d, n
    return best


def edge_names(d):
    n = d.get("name")
    if isinstance(n, str):
        return [n]
    return list(n) if n else []


def cost_safe(d):
    """안전경로 비용: 지하차도는 통행 불가, 침수 이력 근처는 비싸게."""
    L = float(d.get("length", 1.0))
    if d.get("_underpass"):
        return INF
    return L * (1.0 + ALPHA * float(d.get("_flood_risk", 0.0)))


def path_stats(G, path, label):
    total = 0.0
    risk_sum = 0.0
    unders, worst = [], (0.0, None)
    for u, v in zip(path[:-1], path[1:]):
        d = min(G[u][v].values(), key=lambda x: x.get("length", 1e9))
        L = float(d.get("length", 0)); total += L
        r = float(d.get("_flood_risk", 0.0)); risk_sum += r * L
        if d.get("_underpass"):
            nm = edge_names(d)
            unders.append(nm[-1] if nm else "(무명 지하차도)")
        if r > worst[0]:
            worst = (r, (edge_names(d) or ["(무명)"])[0])
    mean_risk = risk_sum / total if total else 0
    return dict(label=label, length=total, mean_risk=mean_risk,
                underpasses=unders, worst=worst)


def main():
    import networkx as nx
    import osmnx as ox

    G = pickle.load(open(os.path.join(OUT, "graph_osong.pkl"), "rb"))
    print(f"■ 그래프 노드 {G.number_of_nodes():,} · 엣지 {G.number_of_edges():,}")

    # ── 대피소 후보 ─────────────────────────────────────────
    sh = json.load(open(os.path.join(DATA, "osong_2023", "shelters_flood_joined.json"),
                        encoding="utf-8"))
    cand = []
    for s in sh:
        try:
            la, lo = float(s["LAT"]), float(s["LOT"])
        except (KeyError, ValueError):
            continue
        d = km((la, lo), OSONG)
        if d <= 3.0:
            cand.append(dict(name=s.get("REARE_NM"), lat=la, lon=lo, dist=d,
                             flood_km=s.get("_nearest_flood_km")))
    cand.sort(key=lambda z: z["dist"])
    print(f"\n■ 3km 이내 대피소 {len(cand)}곳 · 최근접 5곳")
    for c in cand[:5]:
        fk = c["flood_km"]
        print(f"  {c['dist']:.2f}km  {str(c['name'])[:20]:<22} "
              f"침수흔적까지 {f'{fk*1000:.0f}m' if fk else '-'}")

    # 안전경로용 가중치를 미리 부여
    for u, v, k, d in G.edges(keys=True, data=True):
        d["_w"] = cost_safe(d)

    # ── 출발점 탐색 ─────────────────────────────────────────
    # 궁평2지하차도를 출발점으로 고정하면 대피 경로가 애초에 지하차도를 지나지 않는다.
    # 실제로 문제가 되는 상황은 "대피소로 가려면 지하차도를 건너야 하는" 경우다.
    # 따라서 지하차도 양 끝을 후보 출발점으로 삼아, 최단경로가 그 지하차도를
    # 실제로 통과하는 조합만 골라낸다.
    und_edges = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True)
                 if d.get("_underpass")]
    starts = []
    for u, v, k, d in und_edges:
        la = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
        lo = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
        if km((la, lo), OSONG) <= 5.0:
            nm = edge_names(d)
            starts.append((u, v, nm[-1] if nm else "(무명 지하차도)"))
    print(f"\n■ 오송 5km 내 지하차도 {len(starts)}개를 경유 후보로 탐색")

    results = []
    for su, sv, uname in starts:
        for c in cand[:10]:
            dest = nearest_node(G, c["lat"], c["lon"])
            for orig in (sv, su):
                if dest == orig:
                    continue
                try:
                    p_short = nx.shortest_path(G, orig, dest, weight="length")
                except nx.NetworkXNoPath:
                    continue
                s1 = path_stats(G, p_short, "최단")
                if not s1["underpasses"]:
                    continue                      # 지하차도를 안 지나면 비교 의미 없음
                try:
                    p_safe = nx.shortest_path(G, orig, dest, weight="_w")
                except nx.NetworkXNoPath:
                    continue
                s2 = path_stats(G, p_safe, "안전")
                # 길이(m)로는 강제 통과를 알 수 없다. 차단 가중치를 포함한 비용으로 본다.
                if nx.path_weight(G, p_safe, "_w") >= INF / 10:
                    continue
                results.append((c, p_short, s1, p_safe, s2, orig, uname))
    # ── 사례 선별 ───────────────────────────────────────────
    # 좋은 사례의 조건은 '짧은 경로'가 아니다. 다음 세 가지를 모두 만족해야
    # "조금 돌아가면 안전해진다"는 주장이 성립한다.
    #   ① 안전경로가 지하차도를 실제로 피한다
    #   ② 안전경로의 침수위험이 실제로 낮아진다  ← 이걸 안 보면 우회로가 더 위험할 수 있다
    #   ③ 우회 대가가 현실적이다 (도보 대피가 가능한 범위)
    good = []
    for c, p1, s1, p2, s2, orig, uname in results:
        if s2["underpasses"]:
            continue
        detour = (s2["length"] - s1["length"]) / max(s1["length"], 1)
        risk_drop = s1["mean_risk"] - s2["mean_risk"]
        if detour > 1.5:                 # 2.5배 넘게 돌아가면 대피로로 비현실적
            continue
        if risk_drop <= 0:               # 우회로가 더 위험하면 제안할 수 없다
            continue
        good.append((detour, -risk_drop, len(good), (c, p1, s1, p2, s2, orig, uname)))
    # 동점일 때 dict끼리 비교하려 들지 않도록 정렬 키를 명시한다
    good.sort(key=lambda g: (g[0], g[1], g[2]))
    if good:
        results = [g[3] for g in good][:8]
        print(f"  → 조건을 만족하는 사례 {len(good)}건 (우회 ≤150% · 위험 감소)")
    else:
        print(f"  ⚠ 조건을 만족하는 사례 없음 — 전체에서 우회가 적은 순으로 제시")
        results.sort(key=lambda r: (len(r[4]["underpasses"]),
                                    (r[4]["length"] - r[2]["length"]) / max(r[2]["length"], 1)))
        results = results[:8]

    print(f"\n■ 경로 비교 (출발: 지하차도 양끝 노드)")
    print(f"  {'대피소':<20}{'최단(m)':>9}{'지하차도':>9}{'안전(m)':>9}{'지하차도':>9}{'우회':>8}")
    print("  " + "─" * 68)
    best = None
    for c, p1, s1, p2, s2, orig, uname in results:
        u1 = len(s1["underpasses"])
        if s2:
            u2 = len(s2["underpasses"])
            extra = (s2["length"] - s1["length"]) / s1["length"] * 100
            print(f"  {str(c['name'])[:18]:<20}{s1['length']:>9.0f}{u1:>9}"
                  f"{s2['length']:>9.0f}{u2:>9}{extra:>7.0f}%")
            if u1 > 0 and u2 == 0 and best is None:
                best = (c, p1, s1, p2, s2, orig, uname)
        else:
            print(f"  {str(c['name'])[:18]:<20}{s1['length']:>9.0f}{u1:>9}{'경로없음':>9}")

    if best is None and results:
        best = results[0]

    c, p1, s1, p2, s2, orig, uname = best
    print(f"\n■ 대표 사례: {c['name']}")
    print(f"  [최단경로] {s1['length']:.0f}m · 평균 침수위험 {s1['mean_risk']:.3f}")
    if s1["underpasses"]:
        print(f"     ⚠ 지하차도 통과: {', '.join(s1['underpasses'])}")
    print(f"  [안전경로] {s2['length']:.0f}m · 평균 침수위험 {s2['mean_risk']:.3f}")
    print(f"     지하차도 통과: {len(s2['underpasses'])}개")
    print(f"  → 거리 {s2['length']-s1['length']:+.0f}m "
          f"({(s2['length']/s1['length']-1)*100:+.0f}%) 늘리고 지하차도를 피한다")

    json.dump(dict(
        origin=[G.nodes[orig]["y"], G.nodes[orig]["x"]], underpass=uname, shelter=c,
        short=dict(length=s1["length"], risk=s1["mean_risk"], underpasses=s1["underpasses"],
                   coords=[[G.nodes[n]["y"], G.nodes[n]["x"]] for n in p1]),
        safe=dict(length=s2["length"], risk=s2["mean_risk"], underpasses=s2["underpasses"],
                  coords=[[G.nodes[n]["y"], G.nodes[n]["x"]] for n in p2]),
    ), open(os.path.join(OUT, "route_compare.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=2)
    print(f"\n  → data/routing/route_compare.json")


if __name__ == "__main__":
    main()
