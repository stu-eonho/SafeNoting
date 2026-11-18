# -*- coding: utf-8 -*-
"""
경로 비교 시각화 — 예선 제출 이미지 ②

한 장으로 보여야 하는 것:
    ① 최단경로는 지하차도를 지난다 (빨강)
    ② 안전경로는 그것을 피해 돌아간다 (파랑)
    ③ 그 대가는 고작 몇 미터다

사용:
    python scripts/plot_route.py
"""
import sys, os, json, math, pickle

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(DATA, "routing")


def setup_font():
    """Windows 기본 한글 폰트. 없으면 경고만 하고 진행한다."""
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
    print("  ⚠ 한글 폰트를 찾지 못했다 — 라벨이 깨질 수 있다")
    return None


def main():
    import matplotlib
    matplotlib.use("Agg")
    font = setup_font()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    G = pickle.load(open(os.path.join(OUT, "graph_osong.pkl"), "rb"))
    R = json.load(open(os.path.join(OUT, "route_compare.json"), encoding="utf-8"))
    short = R["short"]; safe = R["safe"]

    # 그릴 범위: 두 경로를 모두 담되 여백을 준다
    allpts = short["coords"] + safe["coords"]
    lats = [p[0] for p in allpts]; lons = [p[1] for p in allpts]
    pad = 0.006
    ymin, ymax = min(lats) - pad, max(lats) + pad
    xmin, xmax = min(lons) - pad, max(lons) + pad

    fig, ax = plt.subplots(figsize=(13, 11), dpi=140)
    ax.set_facecolor("#0f1419")
    fig.patch.set_facecolor("#0f1419")

    # ── 배경 도로망 ─────────────────────────────────────────
    for u, v, k, d in G.edges(keys=True, data=True):
        y1, x1 = G.nodes[u]["y"], G.nodes[u]["x"]
        y2, x2 = G.nodes[v]["y"], G.nodes[v]["x"]
        if not (xmin <= x1 <= xmax and ymin <= y1 <= ymax):
            continue
        risk = float(d.get("_flood_risk", 0.0))
        if d.get("_underpass"):
            ax.plot([x1, x2], [y1, y2], color="#ff3b30", lw=3.2, alpha=0.9, zorder=3)
        elif risk > 0:
            # 침수 이력 근처일수록 진하게
            ax.plot([x1, x2], [y1, y2], color="#f5a623",
                    lw=0.8 + risk * 2.5, alpha=0.25 + min(risk, 1) * 0.45, zorder=2)
        else:
            ax.plot([x1, x2], [y1, y2], color="#3a4553", lw=0.5, alpha=0.55, zorder=1)

    # ── 경로 ────────────────────────────────────────────────
    sy = [p[0] for p in short["coords"]]; sx = [p[1] for p in short["coords"]]
    ay = [p[0] for p in safe["coords"]]; axx = [p[1] for p in safe["coords"]]
    ax.plot(sx, sy, color="#ff453a", lw=4.2, alpha=0.95, zorder=5,
            solid_capstyle="round")
    ax.plot(axx, ay, color="#0a84ff", lw=4.2, alpha=0.95, zorder=6,
            solid_capstyle="round", linestyle=(0, (6, 3)))

    # ── 지점 ────────────────────────────────────────────────
    o = R["origin"]; sh = R["shelter"]
    ax.scatter([o[1]], [o[0]], s=280, c="#ffffff", edgecolors="#0f1419",
               linewidths=2.5, zorder=8, marker="o")
    ax.scatter([sh["lon"]], [sh["lat"]], s=340, c="#30d158", edgecolors="#0f1419",
               linewidths=2.5, zorder=8, marker="*")
    ax.annotate("출발", (o[1], o[0]), xytext=(10, 10), textcoords="offset points",
                color="white", fontsize=12, fontweight="bold", zorder=9)
    ax.annotate(f"대피소\n{sh['name']}", (sh["lon"], sh["lat"]),
                xytext=(12, -28), textcoords="offset points",
                color="#30d158", fontsize=12, fontweight="bold", zorder=9)

    # ── 제목·범례 ───────────────────────────────────────────
    # 제목은 실제로 계산된 사례를 그대로 쓴다.
    # 참사 지명을 임의로 붙이면 그림과 내용이 어긋나 오해를 부른다.
    gap = safe["length"] - short["length"]
    pct = gap / max(short["length"], 1) * 100
    upname = R.get("underpass") or (short["underpasses"] or ["지하차도"])[0]
    ax.set_title(
        f"최단경로는 {upname}를 지난다 — 피하는 대가는 {gap:+.0f}m ({pct:+.0f}%)\n"
        f"침수 시 지하차도는 '가장 빠른 길'이자 '가장 위험한 길'이 된다",
        color="white", fontsize=17, fontweight="bold", pad=18)

    legend = [
        Line2D([0], [0], color="#ff453a", lw=4,
               label=f"최단경로 {short['length']:.0f}m · 지하차도 {len(short['underpasses'])}개 통과"),
        Line2D([0], [0], color="#0a84ff", lw=4, linestyle="--",
               label=f"안전경로 {safe['length']:.0f}m · 지하차도 0개"),
        Line2D([0], [0], color="#ff3b30", lw=3, label="지하차도 (통행 차단)"),
        Line2D([0], [0], color="#f5a623", lw=3, label="과거 침수흔적 구간"),
    ]
    ax.legend(handles=legend, loc="lower left", facecolor="#1c2128",
              edgecolor="#3a4553", labelcolor="white", fontsize=11.5, framealpha=0.95)

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    note = (f"침수흔적 {len([1 for _,_,_,d in G.edges(keys=True,data=True) if d.get('_flood_risk',0)>0]):,}개 구간 · "
            f"지하차도 {len([1 for _,_,_,d in G.edges(keys=True,data=True) if d.get('_underpass')])}개 식별 "
            f"(OpenStreetMap tunnel 태그) · 침수흔적도 행정안전부")
    fig.text(0.5, 0.022, note, ha="center", color="#8b949e", fontsize=9.5)

    p = os.path.join(OUT, "route_compare.png")
    fig.savefig(p, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  → {p}")
    print(f"  최단 {short['length']:.0f}m (지하차도 {len(short['underpasses'])}) / "
          f"안전 {safe['length']:.0f}m (지하차도 0) / 차이 {gap:+.0f}m")


if __name__ == "__main__":
    main()
