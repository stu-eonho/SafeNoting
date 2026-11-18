# -*- coding: utf-8 -*-
"""
도시별 지하차도 우회 사례 시각화 — 예선 제출 이미지 ②

한 장으로 전달해야 하는 것:
    최단경로는 지하차도를 지나고(빨강), 안전경로는 그것을 피하며(파랑),
    그 대가는 몇백 미터에 불과하다.

색 설계: 지하차도를 최단경로와 같은 빨강으로 두면 구분이 안 된다.
        지하차도는 노랑으로 분리한다.

사용:
    python scripts/plot_city_case.py [사례번호]
"""
import sys, os, json, math, pickle

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data", "routing")


def setup_font():
    import matplotlib
    from matplotlib import font_manager
    for name in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
        except Exception:
            continue
    return None


def main():
    import matplotlib
    matplotlib.use("Agg")
    setup_font()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cases = json.load(open(os.path.join(OUT, "city_cases.json"), encoding="utf-8"))
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    c = cases[idx]
    G = pickle.load(open(os.path.join(OUT, f"graph_{c['city']}.pkl"), "rb"))

    short, safe = c["short"], c["safe"]
    pts = short + safe
    lats = [p[0] for p in pts]; lons = [p[1] for p in pts]
    pad = max(0.0018, (max(lats) - min(lats)) * 0.22)
    ymin, ymax = min(lats) - pad, max(lats) + pad
    xmin, xmax = min(lons) - pad * 1.25, max(lons) + pad * 1.25

    fig, ax = plt.subplots(figsize=(12.5, 11), dpi=145)
    ax.set_facecolor("#0f1419"); fig.patch.set_facecolor("#0f1419")

    # ── 맥락 레이어 (건물·수계·랜드마크) ─────────────────────
    # 도로만 그리면 선의 집합일 뿐이라 "여기가 어디인지" 읽히지 않는다.
    ctx_path = os.path.join(OUT, f"context_{c['city']}.json")
    ctx = json.load(open(ctx_path, encoding="utf-8")) if os.path.exists(ctx_path) else {}

    def inbox(la, lo):
        return xmin <= lo <= xmax and ymin <= la <= ymax

    for ring in ctx.get("water_areas", []):
        ys = [p[0] for p in ring]; xs = [p[1] for p in ring]
        if not any(inbox(a, b) for a, b in zip(ys, xs)):
            continue
        ax.fill(xs, ys, color="#16344d", alpha=0.9, zorder=0, linewidth=0)
    for line in ctx.get("water_lines", []):
        ys = [p[0] for p in line]; xs = [p[1] for p in line]
        if not any(inbox(a, b) for a, b in zip(ys, xs)):
            continue
        ax.plot(xs, ys, color="#1d4a6b", lw=2.6, alpha=0.9, zorder=0)

    nb = 0
    for ring in ctx.get("buildings", []):
        ys = [p[0] for p in ring]; xs = [p[1] for p in ring]
        if not (xmin <= xs[0] <= xmax and ymin <= ys[0] <= ymax):
            continue
        ax.fill(xs, ys, color="#2b3340", alpha=0.95, zorder=0, linewidth=0)
        nb += 1

    # 최단경로가 실제로 지나는 좌표쌍 — 이 구간의 지하차도만 강조한다.
    # 화면 안의 모든 지하차도를 같은 굵기로 칠하면, 경로와 무관한 것이
    # 더 눈에 띄어 "왜 저기가 노랗지?" 하는 혼란을 준다.
    on_route = set()
    for a, b in zip(short[:-1], short[1:]):
        on_route.add((round(a[0], 6), round(a[1], 6), round(b[0], 6), round(b[1], 6)))
        on_route.add((round(b[0], 6), round(b[1], 6), round(a[0], 6), round(a[1], 6)))

    # 배경 도로망
    for u, v, k, d in G.edges(keys=True, data=True):
        y1, x1 = G.nodes[u]["y"], G.nodes[u]["x"]
        y2, x2 = G.nodes[v]["y"], G.nodes[v]["x"]
        if not (xmin <= x1 <= xmax and ymin <= y1 <= ymax):
            continue
        if d.get("_underpass"):
            key = (round(y1, 6), round(x1, 6), round(y2, 6), round(x2, 6))
            if key in on_route:
                ax.plot([x1, x2], [y1, y2], color="#ffd60a", lw=9, alpha=1.0,
                        zorder=8, solid_capstyle="round")      # 경로가 지나는 지하차도
            else:
                ax.plot([x1, x2], [y1, y2], color="#8a7415", lw=3, alpha=0.55,
                        zorder=3, solid_capstyle="round")      # 그 외 지하차도
        else:
            ax.plot([x1, x2], [y1, y2], color="#495261", lw=0.9, alpha=0.75, zorder=1)

    sy = [p[0] for p in short]; sx = [p[1] for p in short]
    ay = [p[0] for p in safe]; axx = [p[1] for p in safe]
    ax.plot(sx, sy, color="#ff453a", lw=5.0, alpha=0.95, zorder=6, solid_capstyle="round")
    ax.plot(axx, ay, color="#0a84ff", lw=5.0, alpha=0.95, zorder=7,
            solid_capstyle="round", linestyle=(0, (7, 3.5)))

    # ── 랜드마크 ────────────────────────────────────────────
    # 경로 주변 것만 표시한다. 전부 찍으면 라벨이 겹쳐 읽을 수 없다.
    # 대피 판단에 의미 있는 시설만 남긴다.
    # 개인 의원·성형외과까지 찍으면 라벨이 겹칠 뿐 아니라
    # "이게 대피와 무슨 상관인가" 하는 인상을 준다.
    ICON = {"school": "학교", "university": "대학", "college": "대학",
            "hospital": "병원", "townhall": "관공서", "police": "경찰",
            "fire_station": "소방", "community_centre": "주민센터",
            "station": "역", "park": "공원"}
    route_pts = short + safe
    cands = []
    for m in ctx.get("landmarks", []):
        if m.get("kind") not in ICON or not inbox(m["lat"], m["lon"]):
            continue
        near = min(math.hypot((m["lat"] - p[0]) * 111000, (m["lon"] - p[1]) * 88800)
                   for p in route_pts)
        if near > 300:
            continue
        # 역·학교·관공서를 병원보다 우선 (대피소로 지정되는 시설)
        prio = 0 if m["kind"] in ("station", "school", "townhall",
                                  "community_centre", "park") else 1
        cands.append((prio, near, m))
    cands.sort(key=lambda z: (z[0], z[1]))

    placed = []
    for _, _, m in cands:
        # 이미 찍은 라벨과 120m 안쪽이면 건너뛴다 (겹침 방지)
        if any(math.hypot((m["lat"] - q[0]) * 111000, (m["lon"] - q[1]) * 88800) < 120
               for q in placed):
            continue
        ax.scatter([m["lon"]], [m["lat"]], s=58, c="#c9d1d9", edgecolors="#0f1419",
                   linewidths=1.3, zorder=5, marker="s")
        ax.annotate(f"{m['name']}\n({ICON[m['kind']]})", (m["lon"], m["lat"]),
                    xytext=(8, 5), textcoords="offset points", color="#b5c0cc",
                    fontsize=9, zorder=5, linespacing=1.15)
        placed.append((m["lat"], m["lon"]))
        if len(placed) >= 9:
            break

    ax.scatter([sx[0]], [sy[0]], s=330, c="#ffffff", edgecolors="#0f1419",
               linewidths=3, zorder=9)
    ax.scatter([sx[-1]], [sy[-1]], s=380, c="#30d158", edgecolors="#0f1419",
               linewidths=3, zorder=9, marker="*")
    ax.annotate("현재 위치", (sx[0], sy[0]), xytext=(13, 11), textcoords="offset points",
                color="white", fontsize=13, fontweight="bold", zorder=10)
    ax.annotate("대피 목적지", (sx[-1], sy[-1]), xytext=(13, -26),
                textcoords="offset points", color="#30d158", fontsize=13,
                fontweight="bold", zorder=10)

    gap = c["L2"] - c["L1"]
    ax.set_title(
        f"침수 시 지하차도는 '가장 빠른 길'이자 '가장 위험한 길'이다\n"
        f"{c['label']} · {c['underpass']}  —  우회 대가 {gap:+.0f}m "
        f"(도보 약 {gap/67:.0f}분)",
        color="white", fontsize=17.5, fontweight="bold", pad=20, linespacing=1.5)

    legend = [
        Line2D([0], [0], color="#ff453a", lw=5,
               label=f"최단경로 {c['L1']:.0f}m  ·  지하차도 통과 (위험)"),
        Line2D([0], [0], color="#0a84ff", lw=5, linestyle="--",
               label=f"안전경로 {c['L2']:.0f}m  ·  지하차도 회피"),
        Line2D([0], [0], color="#ffd60a", lw=5, label="지하차도 (침수 시 통행 차단)"),
    ]
    ax.legend(handles=legend, loc="lower left", facecolor="#1c2128", edgecolor="#3a4553",
              labelcolor="white", fontsize=12.5, framealpha=0.96, borderpad=1.0)

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    sub = c.get("note", "")
    fig.text(0.5, 0.028,
             f"{sub}    |    지하차도는 OpenStreetMap tunnel 태그로 자동 식별 "
             f"(이 지역 {sum(1 for _,_,_,d in G.edges(keys=True,data=True) if d.get('_underpass'))}개)",
             ha="center", color="#8b949e", fontsize=10)

    p = os.path.join(OUT, f"case_{c['city']}.png")
    fig.savefig(p, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  → {p}")
    print(f"  {c['label']} {c['underpass']} · 최단 {c['L1']:.0f}m → 안전 {c['L2']:.0f}m "
          f"({c['detour']*100:+.0f}%)")


if __name__ == "__main__":
    main()
