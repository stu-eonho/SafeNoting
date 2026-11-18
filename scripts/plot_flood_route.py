# -*- coding: utf-8 -*-
"""
침수 위험구역 회피 경로 시각화 — 예선 제출 이미지 ②

지하차도 버전과의 결정적 차이: 침수구역은 '점'이 아니라 '면'이다.
주황색 면이 실제 침수 이력 구역이고, 최단경로(빨강)는 그 안을 통과하며,
안전경로(파랑)는 면을 우회한다. 한 장으로 다음이 읽혀야 한다.

    "대피소로 가는 가장 빠른 길이 침수 구역 한가운데를 지난다"

사용:
    python scripts/plot_flood_route.py [사례번호]
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
    from matplotlib.patches import Patch

    R = json.load(open(os.path.join(OUT, "flood_route.json"), encoding="utf-8"))
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    c = R["cases"][idx]
    G = pickle.load(open(os.path.join(OUT, "graph_cheongju.pkl"), "rb"))

    short, safe = c["short"], c["safe"]
    pts = short + safe
    lats = [p[0] for p in pts]; lons = [p[1] for p in pts]
    pad = max(0.0022, (max(lats) - min(lats)) * 0.30)
    ymin, ymax = min(lats) - pad, max(lats) + pad
    xmin, xmax = min(lons) - pad * 1.3, max(lons) + pad * 1.3

    fig, ax = plt.subplots(figsize=(13, 11.5), dpi=145)
    ax.set_facecolor("#0f1419"); fig.patch.set_facecolor("#0f1419")

    def inbox(la, lo):
        return xmin <= lo <= xmax and ymin <= la <= ymax

    # ── 맥락 (건물) ─────────────────────────────────────────
    ctx_p = os.path.join(OUT, "context_cheongju.json")
    ctx = json.load(open(ctx_p, encoding="utf-8")) if os.path.exists(ctx_p) else {}
    for ring in ctx.get("buildings", []):
        ys = [p[0] for p in ring]; xs = [p[1] for p in ring]
        if not (xmin <= xs[0] <= xmax and ymin <= ys[0] <= ymax):
            continue
        ax.fill(xs, ys, color="#2b3340", alpha=0.9, zorder=0, linewidth=0)

    # ── 침수 위험구역 (면) ──────────────────────────────────
    # 이 그림의 주인공. 수심이 깊을수록 진하게.
    nfp = 0
    for p in R["polys"]:
        ring = p["ring"]
        ys = [q[0] for q in ring]; xs = [q[1] for q in ring]
        if not any(inbox(a, b) for a, b in zip(ys, xs)):
            continue
        dep = p.get("depth") or 0
        alpha = 0.30 + min(dep / 2.0, 1.0) * 0.35
        ax.fill(xs, ys, color="#ff9f0a", alpha=alpha, zorder=1, linewidth=0)
        ax.plot(xs + [xs[0]], ys + [ys[0]], color="#ffb340", lw=0.9, alpha=0.75, zorder=1)
        nfp += 1

    # ── 도로망 ──────────────────────────────────────────────
    for u, v, d in G.edges(data=True):
        if not (inbox(u[0], u[1]) or inbox(v[0], v[1])):
            continue
        ax.plot([u[1], v[1]], [u[0], v[0]], color="#4d5765", lw=1.0, alpha=0.8, zorder=2)

    # ── 경로 ────────────────────────────────────────────────
    # 두 경로는 상당 구간을 공유한다. 그대로 겹쳐 그리면 파선 틈으로
    # 아래 색이 비쳐 "겹친 건지 오류인지" 알 수 없다.
    # 공통 구간을 따로 칠해 어디서 갈라지는지 드러낸다.
    def seg_key(a, b):
        ka = (round(a[0], 7), round(a[1], 7))
        kb = (round(b[0], 7), round(b[1], 7))
        return (ka, kb) if ka <= kb else (kb, ka)

    short_segs = {seg_key(a, b) for a, b in zip(short[:-1], short[1:])}
    safe_segs = {seg_key(a, b) for a, b in zip(safe[:-1], safe[1:])}
    common = short_segs & safe_segs

    def draw(path, segset, color, lw, style, z):
        run = []
        for a, b in zip(path[:-1], path[1:]):
            if seg_key(a, b) in segset:
                if not run:
                    run = [a]
                run.append(b)
            elif run:
                ax.plot([p[1] for p in run], [p[0] for p in run], color=color,
                        lw=lw, alpha=0.97, zorder=z, solid_capstyle="round",
                        linestyle=style)
                run = []
        if run:
            ax.plot([p[1] for p in run], [p[0] for p in run], color=color,
                    lw=lw, alpha=0.97, zorder=z, solid_capstyle="round",
                    linestyle=style)

    # 마커용 좌표 (경로 양 끝)
    sy = [p[0] for p in short]; sx = [p[1] for p in short]

    # 공통 구간도 안전경로 색으로 칠한다.
    # 어차피 안전경로가 지나는 길이므로 파란 선이 끊기지 않고 이어져야
    # "대피는 이 파란 길을 따라간다"가 한눈에 읽힌다.
    draw(short, common, "#0a84ff", 5.4, "-", 5)
    # 최단경로만의 구간 (침수구역 통과)
    draw(short, short_segs - common, "#ff453a", 5.4, "-", 6)
    # 안전경로만의 구간 (우회)
    draw(safe, safe_segs - common, "#0a84ff", 5.4, (0, (7, 3.5)), 7)

    # 최단경로가 침수구역을 통과하는 구간을 표시
    for a, b in zip(short[:-1], short[1:]):
        ka, kb = (round(a[0], 7), round(a[1], 7)), (round(b[0], 7), round(b[1], 7))
        if G.has_edge(ka, kb) and G[ka][kb].get("flooded"):
            ax.plot([a[1], b[1]], [a[0], b[0]], color="#ffffff", lw=9,
                    alpha=0.30, zorder=5, solid_capstyle="round")

    # ── 대피소 (안전 / 부적합) ──────────────────────────────
    # 부적합 대피소를 함께 그려야 "왜 더 먼 곳으로 가는가"에 답이 된다.
    # 지정 대피소에는 침수 분류가 없어 야외 운동장·저지대가 섞여 있다.
    n_bad = n_good = 0
    for x in R.get("shelters_unsafe", []):
        if not inbox(x["lat"], x["lon"]):
            continue
        ax.scatter([x["lon"]], [x["lat"]], s=150, c="#6e7681", marker="X",
                   edgecolors="#0f1419", linewidths=1.6, zorder=6, alpha=0.9)
        n_bad += 1
    for x in R.get("shelters_safe", []):
        if not inbox(x["lat"], x["lon"]):
            continue
        ax.scatter([x["lon"]], [x["lat"]], s=115, c="#30d158", marker="o",
                   edgecolors="#0f1419", linewidths=1.5, zorder=6, alpha=0.75)
        n_good += 1

    ax.scatter([sx[0]], [sy[0]], s=340, c="#ffffff", edgecolors="#0f1419",
               linewidths=3, zorder=9)
    ax.scatter([sx[-1]], [sy[-1]], s=400, c="#30d158", edgecolors="#0f1419",
               linewidths=3, zorder=9, marker="*")
    ax.annotate("현재 위치", (sx[0], sy[0]), xytext=(14, 11), textcoords="offset points",
                color="white", fontsize=13.5, fontweight="bold", zorder=10)
    shname = c.get("shelter") or "대피소"
    ax.annotate("대피소\n" + shname, (sx[-1], sy[-1]), xytext=(14, -34),
                textcoords="offset points", color="#30d158", fontsize=13.5,
                fontweight="bold", zorder=10, linespacing=1.35)

    gap = c["L2"] - c["L1"]
    ax.set_title(
        f"대피 경로 한가운데 침수 구역이 있다\n"
        f"최단경로는 침수도로 {c['n_flood']}개를 지난다 (최대 수심 {c['max_depth']:.1f}m)  —  "
        f"우회 {gap:+.0f}m",
        color="white", fontsize=17, fontweight="bold", pad=20, linespacing=1.6)

    legend = [
        Patch(facecolor="#ff9f0a", alpha=0.55, label=f"침수 이력 구역 ({nfp}개 · 행정안전부 침수흔적도)"),
        Line2D([0], [0], color="#ff453a", lw=5,
               label=f"최단경로 {c['L1']:.0f}m  ·  침수구역 통과"),
        Line2D([0], [0], color="#0a84ff", lw=5,
               label=f"안전경로 {c['L2']:.0f}m  ·  침수구역 회피"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#30d158",
               markersize=11, label=f"침수 시 이용 가능 대피소 ({n_good}곳)"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#6e7681",
               markersize=12, label=f"부적합 대피소 ({n_bad}곳 · 야외/저지대)"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.015),
              ncol=3, facecolor="#1c2128", edgecolor="#3a4553", labelcolor="white",
              fontsize=11.5, framealpha=1.0, borderpad=1.0, columnspacing=2.2,
              handlelength=2.4)

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    fig.text(0.5, -0.055,
             "카카오맵·네이버지도는 침수 이력을 모른다 — 최단거리만 계산하기 때문이다    |    "
             "침수흔적도 행정안전부 · 도로망 OpenStreetMap",
             ha="center", color="#8b949e", fontsize=10)

    p = os.path.join(OUT, "flood_route.png")
    fig.savefig(p, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  → {p}")
    print(f"  최단 {c['L1']:.0f}m (침수도로 {c['n_flood']}개) → 안전 {c['L2']:.0f}m "
          f"({c['detour']*100:+.0f}%) · 침수구역 {nfp}개 표시")


if __name__ == "__main__":
    main()
