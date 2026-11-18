# -*- coding: utf-8 -*-
"""
오송 2023-07-15 — 흔적도만 볼 때 vs 예측을 겹칠 때

■ 이 그림이 말하려는 것
  침수흔적도에는 2023년 자료가 없다(최신 2022년). 참사 당일 기준으로
  과거 흔적만 보면 궁평2지하차도는 기록이 없어 '안전'해 보인다.
  실제로는 100m 옆에 2017년 흔적이 깔려 있고, 미호강교 관측소는 478m 거리다.

      흔적도   = 물이 어디로 가는가 (공간)
      수위예측 = 언제 그렇게 되는가 (시간)

■ 위험구역 판정
  경보수위(8.0m) 초과 예측  → 반경 1.5km 흔적 폴리곤을 켠다
  심각수위(9.38m) 초과 예측 → 반경 3.0km
  DEM이 없으므로 흔적 폴리곤을 침수 범위의 대리물로 쓴다.

사용:
    python scripts/osong_predictive_map.py 05
"""
import sys, os, json, re, math

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts"))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(DATA, "routing")

INC = (36.62250963467178, 127.34523676137114)     # 궁평2지하차도
ALM, SRS, GDT = 8.0, 9.38, 19.643
BBOX = (36.600, 36.650, 127.310, 127.385)


def m3857(x, y):
    lon = x / 20037508.34 * 180.0
    lat = y / 20037508.34 * 180.0
    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
    return lat, lon


def km(a, b, c, d):
    return math.hypot((a - c) * 111, (b - d) * 111 * math.cos(math.radians(a)))


def load_traces():
    s, n, w, e = BBOX
    out = []
    for f in json.load(open(os.path.join(DATA, "osong_2023",
                                         "fludmarks_chungbuk.json"), encoding="utf-8")):
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
        if dep > 5.0:          # 47m 같은 값이 섞여 있다 — 물리적으로 불가능
            dep = 0.0
        out.append(dict(ring=ring, cy=cy, cx=cx, depth=dep, year=str(f.get("FLDN_YR"))))
    return out


def predict_series(day="20230715"):
    import torch
    from forecast_model import build_model
    import predict_osong as po

    ck = os.path.join(DATA, "timeseries", "model_multi.pt")
    if not os.path.exists(ck):
        return None
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    model = build_model(9, po.LOOKBACK, 4, n_wl_ch=3,
                        **po.infer_arch(sd["state"]), dropout=0.2, use_revin=False)
    model.load_state_dict(sd["state"])
    model.eval()

    wl = po.load_wl()
    rain = po.load_rain()
    times = sorted(wl)
    idx = {t: k for k, t in enumerate(times)}
    hidx = [po.hour_idx(t) for t in times]
    res = {}
    for hh in range(24):
        t = f"{day}{hh:02d}00"
        if t not in idx:
            continue
        i = idx[t]
        if i < po.LOOKBACK or i + 3 >= len(times):
            continue
        if hidx[i] - hidx[i - po.LOOKBACK + 1] != po.LOOKBACK - 1:
            continue
        X = po.window(times, wl, rain, i)[None, ...]
        with torch.no_grad():
            o, _, _ = model(torch.tensor(X))
        res[f"{hh:02d}"] = dict(cur=wl[t], pred=wl[t] + float(o[0, 2]) * ALM,
                                act=wl[times[i + 3]])
    return res


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPoly
    from matplotlib.lines import Line2D
    from plot_flood_route import setup_font
    setup_font()

    hh = sys.argv[1] if len(sys.argv) > 1 else "05"
    tr = load_traces()
    ctx = json.load(open(os.path.join(OUT, "context_osong.json"), encoding="utf-8"))
    print(f"■ 침수흔적 {len(tr)}건 · 도로 {len(ctx['roads'])} · 건물 {len(ctx['buildings'])}")

    ps = predict_series()
    cur = pred = None
    if ps and hh in ps:
        cur, pred = ps[hh]["cur"], ps[hh]["pred"]
        print(f"■ {hh}시  현재 {cur:.2f}m → 3시간 뒤 {pred:.2f}m (실제 {ps[hh]['act']:.2f}m)")

    rad = 0.0 if pred is None else (3.0 if pred >= SRS else (1.5 if pred >= ALM else 0.0))
    act_set = {id(t) for t in tr if rad and km(t["cy"], t["cx"], *INC) <= rad}
    print(f"■ 예측 위험구역 {len(act_set)}건 (반경 {rad}km)")

    st = json.load(open(os.path.join(DATA, "timeseries", "stations_40km.json"),
                       encoding="utf-8"))
    sll = next(((x["_lat"], x["_lon"]) for x in st if x["wlobscd"] == "3011665"), None)

    s, n, w, e = BBOX
    fig, axes = plt.subplots(1, 2, figsize=(19.5, 9.6), facecolor="#0d1117")
    for ax, mode in zip(axes, ("past", "pred")):
        ax.set_facecolor("#0d1117")
        for b in ctx["buildings"]:
            ax.add_patch(MPoly([(q[1], q[0]) for q in b["pts"]], closed=True,
                               facecolor="#262c36", edgecolor="none", zorder=2))
        for r in ctx["roads"]:
            hw = r["tags"].get("highway", "")
            lw = 1.9 if hw in ("motorway", "trunk", "primary") else (
                 1.2 if hw in ("secondary", "tertiary") else 0.6)
            ax.plot([q[1] for q in r["pts"]], [q[0] for q in r["pts"]],
                    color="#59606a", lw=lw, zorder=4, solid_capstyle="round")
        for r in ctx["rail"]:
            ax.plot([q[1] for q in r["pts"]], [q[0] for q in r["pts"]],
                    color="#7d8590", lw=1.0, ls=(0, (6, 3)), zorder=4)
        for r in ctx["water"]:
            xs = [q[1] for q in r["pts"]]
            ys = [q[0] for q in r["pts"]]
            closed = (len(xs) > 2 and abs(xs[0] - xs[-1]) < 1e-9
                      and abs(ys[0] - ys[-1]) < 1e-9)
            if r["tags"].get("natural") == "water" or closed:
                ax.add_patch(MPoly(list(zip(xs, ys)), closed=True,
                                   facecolor="#123a5c", edgecolor="#1f5f96",
                                   lw=0.8, zorder=3))
            else:
                ax.plot(xs, ys, color="#2a7ab8", lw=2.6, zorder=3,
                        solid_capstyle="round")
        for t in tr:
            on = (mode == "pred") and (id(t) in act_set)
            ax.add_patch(MPoly([(q[1], q[0]) for q in t["ring"]], closed=True,
                               facecolor="#ff453a" if on else "#484f59",
                               edgecolor="#ff7b72" if on else "#5a626d",
                               alpha=0.80 if on else 0.55, lw=0.6, zorder=6))
        if sll:
            ax.plot(sll[1], sll[0], marker="^", ms=14, color="#2a7ab8",
                    mec="#0d1117", mew=1.6, zorder=9)
            ax.annotate("미호강교 관측소", (sll[1], sll[0]), xytext=(10, 6),
                        textcoords="offset points", color="#58a6ff",
                        fontsize=10.5, fontweight="bold", zorder=10)
        for nd in ctx.get("rail_nodes", []):
            nm = nd["tags"].get("name", "")
            if "오송" not in nm:
                continue
            ax.plot(nd["lon"], nd["lat"], marker="s", ms=9, color="#c9d1d9",
                    mec="#0d1117", mew=1.4, zorder=9)
            ax.annotate(nm, (nd["lon"], nd["lat"]), xytext=(9, -3),
                        textcoords="offset points", color="#c9d1d9",
                        fontsize=10, zorder=10)
        ax.plot(INC[1], INC[0], marker="X", ms=20, color="#ffd60a",
                mec="#0d1117", mew=2.2, zorder=11)
        ax.annotate("궁평2지하차도\n2023-07-15 08:40 침수 · 14명",
                    (INC[1], INC[0]), xytext=(13, -42), textcoords="offset points",
                    color="#ffd60a", fontsize=11.5, fontweight="bold",
                    zorder=12, linespacing=1.4)
        if mode == "past":
            ttl = "과거 침수흔적도만 볼 때"
            sub = "흔적 자료는 2022년까지 — 지하차도 본체에 기록 없음"
        else:
            ttl = f"수위 예측을 겹칠 때  ({hh}시 판단)"
            sub = (f"미호강교 {cur:.2f}m → 3시간 뒤 {pred:.2f}m 예측 (심각수위 {SRS}m 초과)"
                   if pred else "예측 미적용")
        ax.set_title(f"{ttl}\n{sub}", color="white", fontsize=15.5,
                     fontweight="bold", pad=15, linespacing=1.7)
        ax.set_xlim(w, e)
        ax.set_ylim(s, n)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#30363d")

    handles = [
        Line2D([], [], marker="X", ls="", ms=15, color="#ffd60a", label="참사 발생 지점"),
        Line2D([], [], marker="^", ls="", ms=12, color="#2a7ab8", label="수위 관측소 (478m)"),
        Line2D([], [], marker="s", ls="", ms=11, color="#484f59", label="과거 침수흔적 (2017)"),
        Line2D([], [], marker="s", ls="", ms=11, color="#ff453a", label="예측 위험구역"),
        Line2D([], [], color="#2a7ab8", lw=2.6, label="미호강"),
        Line2D([], [], color="#59606a", lw=1.6, label="도로"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.038),
               ncol=6, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9",
               fontsize=11.5, framealpha=1.0, borderpad=1.0, columnspacing=2.0)
    fig.suptitle("침수흔적도는 '어디'를 알려주고, 수위예측은 '언제'를 알려준다",
                 color="white", fontsize=18.5, fontweight="bold", y=0.975)
    fig.text(0.5, 0.008,
             "침수흔적도(행안부) · 수위 미호강교(3011665, 학습에 포함되지 않은 관측소) · "
             "예측 = 전국 39개 관측소로 학습한 Transformer",
             ha="center", color="#8b949e", fontsize=10.5)
    fig.tight_layout(rect=[0, 0.115, 1, 0.94])
    p = os.path.join(OUT, "osong_predictive.png")
    fig.savefig(p, dpi=145, facecolor="#0d1117")
    print(f"\n  → {p}")


if __name__ == "__main__":
    main()
