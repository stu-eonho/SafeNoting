# -*- coding: utf-8 -*-
"""
수위 예측 베이스라인 — Transformer를 쓸 가치가 있는지 판정하는 기준선

★ 평가 설계가 이 스크립트의 핵심이다.
   미호천 수위는 대부분의 시간 동안 평수위(1.5m 근처)에서 거의 변하지 않는다.
   전 구간 RMSE로 평가하면 "3시간 뒤 = 지금과 같다"(persistence)가 거의 만점을 받고,
   어떤 모델을 붙여도 개선폭이 소수점 아래에서 놀게 된다.
   따라서 반드시 ① 전체 ② 홍수기 ③ 급상승 구간을 나눠서 본다.
   우리가 실제로 맞혀야 하는 것은 ③이다.

사용:
    python scripts/baseline_forecast.py            # 기본 3시간 예측
    python scripts/baseline_forecast.py 6          # 6시간 예측
"""
import sys, os, csv, math
from datetime import datetime
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = os.path.join(BASE, "data", "timeseries")

TARGET = "3011665"                       # 미호강교
UPSTREAM = ["3011635", "3011660"]        # 팔결교(-2h), 환희교(-3h) — 시차상관으로 확정
# 미호교(3011685)는 lag +1h(하류)라 입력에 넣으면 미래정보 누수 → 영구 제외
ALMWL, SRSWL = 8.0, 9.38                 # 경보/위험 수위


def load_wl(sta):
    """정제본(_clean)이 있으면 우선 사용하고, 센서 오류로 표시된 행은 버린다."""
    clean = os.path.join(TS, f"wl_{sta}_1H_clean.csv")
    raw = os.path.join(TS, f"wl_{sta}_1H.csv")
    p = clean if os.path.exists(clean) else raw
    if not os.path.exists(p):
        return {}
    out, dropped = {}, 0
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("flag"):
                dropped += 1
                continue
            try:
                out[r["ymdhm"]] = float(r["wl"])
            except (ValueError, KeyError):
                pass
    if dropped:
        print(f"  [{sta}] 센서오류 {dropped}건 제외 (정제본 사용)")
    return out


def load_asos(stn="131"):
    p = os.path.join(TS, f"asos_{stn}_1H.csv")
    if not os.path.exists(p):
        return {}
    out = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tm = r.get("tm", "")
            if len(tm) < 13:
                continue
            key = tm[:4] + tm[5:7] + tm[8:10] + tm[11:13] + "00"
            try:
                out[key] = float(r.get("rn") or 0)
            except ValueError:
                out[key] = 0.0
    return out


def build(horizon=3, lookback=12):
    wl = load_wl(TARGET)
    if not wl:
        raise SystemExit(f"수위 데이터 없음: wl_{TARGET}_1H.csv — collect_waterlevel.py 먼저 실행")
    rain = load_asos()
    ups = {s: load_wl(s) for s in UPSTREAM}
    ups = {s: d for s, d in ups.items() if d}

    times = sorted(wl)
    # ⚠ 인접 레코드가 1시간 간격이라는 보장이 없다. 관측소에 따라 수개월이 비어 있고,
    #   그 구간을 가로지르는 창을 학습에 넣으면 존재하지 않는 급변을 학습하게 된다.
    #   절대 시각 인덱스를 만들어 창 전체가 실제로 연속인지 확인한다.
    hidx = [int(datetime.strptime(t[:12], "%Y%m%d%H%M").timestamp() // 3600)
            for t in times]
    X, Y, T = [], [], []
    for i, t in enumerate(times):
        if i < lookback or i + horizon >= len(times):
            continue
        span = lookback + horizon
        if hidx[i + horizon] - hidx[i - lookback] != span:
            continue          # 창 안에 결측 구간이 있음
        hist = [wl[times[j]] for j in range(i - lookback + 1, i + 1)]
        feat = list(hist)
        feat.append(hist[-1] - hist[-2])                       # 1시간 변화
        feat.append(hist[-1] - hist[-4] if lookback >= 4 else 0)  # 3시간 변화
        feat.append(float(np.mean(hist)))
        # 강수: 최근 1/3/6/12시간 누적
        for w in (1, 3, 6, 12):
            s = sum(rain.get(times[j], 0.0) for j in range(max(0, i - w + 1), i + 1))
            feat.append(s)
        # 상류 수위 + 변화량
        for s in UPSTREAM:
            d = ups.get(s, {})
            v = d.get(t)
            pv = d.get(times[i - 3]) if i >= 3 else None
            feat.append(v if v is not None else hist[-1])
            feat.append((v - pv) if (v is not None and pv is not None) else 0.0)
        X.append(feat)
        Y.append(wl[times[i + horizon]])
        T.append(t)
    return np.array(X, float), np.array(Y, float), T, np.array([x[lookback - 1] for x in X], float)


def fit_ridge(X, y, lam=1.0):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[-1, -1] -= lam
    return np.linalg.solve(A, Xb.T @ y)


def pred(w, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def metrics(y, p):
    e = p - y
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))),
                mae=float(np.mean(np.abs(e))),
                bias=float(np.mean(e)))


def report(name, y, p, masks):
    out = {"전체": metrics(y, p)}
    line = f"  {name:<26} RMSE {out['전체']['rmse']:>6.3f}  MAE {out['전체']['mae']:>6.3f}"
    for label, mk in masks.items():
        if mk.sum() >= 5:
            out[label] = metrics(y[mk], p[mk])
            line += f"   {label} RMSE {out[label]['rmse']:>6.3f}(n={int(mk.sum())})"
    # 재난 예측에서는 과소예측 비율이 RMSE만큼 중요하다
    if "급상승" in out:
        mk = masks["급상승"]
        under = float(np.mean(p[mk] < y[mk]) * 100)
        line += f"  과소{under:>5.1f}%"
    print(line)
    return out


def event_summary(Y, T, cur, label):
    """연도별 홍수·급상승 표본 분포. 분할 지점을 정하려면 이게 먼저 필요하다."""
    from collections import Counter
    fl, ri = Counter(), Counter()
    for y, t, c in zip(Y, T, cur):
        yr = t[:4]
        if y >= ALMWL:
            fl[yr] += 1
        if y - c >= 0.3:
            ri[yr] += 1
    yrs = sorted(set(t[:4] for t in T))
    print(f"  [{label}] 연도별 홍수기(≥{ALMWL}m) / 급상승(≥0.3m)")
    line = "    "
    for yr in yrs:
        if fl[yr] or ri[yr]:
            line += f"{yr}:{fl[yr]}/{ri[yr]}  "
    print(line if line.strip() else "    (해당 표본 없음)")


def main(horizon=3, split_year=None):
    print(f"■ 수위 예측 베이스라인 — {horizon}시간 후 예측\n")
    X, Y, T, cur = build(horizon)
    years = sorted(set(t[:4] for t in T))
    print(f"  표본 {len(X):,}개  피처 {X.shape[1]}개  기간 {T[0][:8]} ~ {T[-1][:8]}")
    print(f"  연도 {len(years)}개: {years[0]}~{years[-1]}\n")
    event_summary(Y, T, cur, "전체")
    print()

    # 연도 단위 시간순 분할 (셔플 절대 금지)
    # 계절성 때문에 비율 분할을 하면 검증 구간이 갈수기만 될 수 있다.
    if split_year is None:
        split_year = years[int(len(years) * 0.7)] if len(years) > 3 else years[-1]
    mask_tr = np.array([t[:4] < split_year for t in T])
    if mask_tr.sum() == 0 or (~mask_tr).sum() == 0:
        cut = int(len(X) * 0.75)
        mask_tr = np.zeros(len(X), bool)
        mask_tr[:cut] = True
        print(f"  ※ 연도 분할 불가 — 비율 분할로 대체\n")

    Xtr, Ytr = X[mask_tr], Y[mask_tr]
    Xte, Yte, cur_te = X[~mask_tr], Y[~mask_tr], cur[~mask_tr]
    Tte = [t for t, m in zip(T, mask_tr) if not m]
    print(f"  학습 {len(Xtr):,}건 (~{split_year} 이전)   검증 {len(Xte):,}건 ({split_year}~)\n")

    # 평가 구간 정의
    flood = Yte >= ALMWL                      # 경보수위 이상
    rise = (Yte - cur_te) >= 0.3              # 급상승 (h시간에 0.3m 이상)
    masks = {"홍수기": flood, "급상승": rise}
    print(f"  검증 구간 홍수기 {int(flood.sum())}건 / 급상승 {int(rise.sum())}건"
          f" / 전체 {len(Yte):,}건")
    if flood.sum() < 5:
        print(f"  ⚠ 검증 구간 홍수 표본 부족 — split_year를 앞당기세요 (인자 2번째)")
    print()

    print("─" * 100)
    results = {}
    # ① Persistence
    results["persistence"] = report("① Persistence(현재값 유지)", Yte, cur_te, masks)
    # ② 선형: 과거 수위만
    lb = 12
    w = fit_ridge(Xtr[:, :lb], Ytr)
    results["linear_wl"] = report("② 선형회귀(수위 lag만)", Yte, pred(w, Xte[:, :lb]), masks)
    # ③ 선형: 전체 피처(강수+상류)
    w = fit_ridge(Xtr, Ytr)
    results["linear_all"] = report("③ 선형회귀(강수+상류 포함)", Yte, pred(w, Xte), masks)
    # ④ 차분 예측(변화량을 학습) — 비정상 시계열에 강함
    w = fit_ridge(Xtr, Ytr - Xtr[:, lb - 1])
    results["linear_delta"] = report("④ 선형회귀(변화량 예측)", Yte,
                                     cur_te + pred(w, Xte), masks)
    print("─" * 100)

    base = results["persistence"]
    print("\n■ Persistence 대비 RMSE 개선율   ★ 급상승 구간이 진짜 성적")
    zones = [z for z in ("전체", "홍수기", "급상승") if z in base]
    print("    " + " " * 16 + "".join(f"{z:>12}" for z in zones))
    for k, v in results.items():
        if k == "persistence":
            continue
        row = f"    {k:<16}"
        for z in zones:
            if z in v and z in base and base[z]["rmse"] > 0:
                imp = (base[z]["rmse"] - v[z]["rmse"]) / base[z]["rmse"] * 100
                row += f"{imp:>+11.1f}%"
            else:
                row += f"{'—':>12}"
        print(row)

    if "급상승" in base:
        best = min((v["급상승"]["rmse"], k) for k, v in results.items() if "급상승" in v)
        print(f"\n  ▶ 급상승 구간 최고: {best[1]} (RMSE {best[0]:.3f})")
        print(f"    Transformer는 이 값을 유의하게 넘어야 채택한다.")
    print("\n※ 전체 RMSE 순위와 급상승 RMSE 순위가 어긋나면, 반드시 급상승을 기준으로 판단할 것.")


if __name__ == "__main__":
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    sy = sys.argv[2] if len(sys.argv) > 2 else None
    main(h, sy)
