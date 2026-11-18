# -*- coding: utf-8 -*-
"""
전국 다중 관측소 학습 데이터셋 구축

단일 관측소(미호강교) 학습의 한계가 명확했다 — 급상승 표본이 1,150개뿐이라
파라미터 15,757개 모델이 과적합했고, 여섯 번의 구조 변경이 모두 실패했다.
전국 59개 관측소를 합쳐 표본을 수십 배로 늘린다.

■ 스케일 통일이 핵심
  관측소마다 수위 범위가 다르다 (경보수위 7.1~12.0m, 영점표고 -2.6~197m).
  그대로 합치면 모델이 "3m"라는 값의 의미를 관측소마다 다르게 배워야 한다.

      정규화 수위 = 실제 수위 / 그 관측소의 경보수위
      → 1.0 = 경보수위. 어느 하천이든 같은 의미
      → 절대적 위험도는 보존되고 하천 규모 차이만 제거된다

  이는 RevIN(창별 평균 제거)과 다르다. RevIN은 "지금 몇 m인가"를 지워서
  평수위의 +0.3m와 위험수위의 +0.3m를 구별 못 하게 만들었다(26% 성능 손실).
  경보수위 정규화는 그 정보를 유지한 채 하천 간 스케일만 맞춘다.

■ 관측소 단위 분할
  시간 분할 대신 관측소를 통째로 나눈다. test 관측소는 학습에서 한 번도
  보지 못한 하천이므로, "처음 보는 하천에서도 작동하는가"를 검증할 수 있다.

사용:
    python scripts/build_multi_dataset.py
"""
import sys, os, csv, json, math
from datetime import datetime
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = os.path.join(BASE, "data", "timeseries")

HORIZONS = [1, 2, 3, 6]
LOOKBACK = 48
MIN_ROWS = 1500          # 관측소 확보량이 고르지 않아 기준을 낮춘다
RAIN_SCALE = 30.0        # mm


def load_wl(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("flag"):
                continue
            try:
                out[r["ymdhm"]] = float(r["wl"])
            except (ValueError, KeyError):
                pass
    return out


def load_rain(stn):
    p = os.path.join(TS, f"asos_{stn}_1H.csv")
    out = {}
    if not os.path.exists(p):
        return out
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tm = r.get("tm", "")
            if len(tm) >= 13:
                k = tm[:4] + tm[5:7] + tm[8:10] + tm[11:13] + "00"
                try:
                    out[k] = float(r.get("rn") or 0)
                except ValueError:
                    out[k] = 0.0
    return out


def hour_idx(t):
    return int(datetime.strptime(t[:12], "%Y%m%d%H%M").timestamp() // 3600)


def build_one(wl, rain, almwl):
    """한 관측소 → (X, Y, cur) . 수위는 경보수위로 나눠 정규화한다."""
    times = sorted(wl)
    if len(times) < LOOKBACK + max(HORIZONS) + 100:
        return None
    hidx = [hour_idx(t) for t in times]
    maxh = max(HORIZONS)

    rain_seq = np.array([rain.get(t, 0.0) for t in times], dtype=np.float64)
    rcum = np.concatenate([[0.0], np.cumsum(rain_seq)])

    Xs, Ys, curs, Ts = [], [], [], []
    for i in range(LOOKBACK, len(times) - maxh):
        if hidx[i + maxh] - hidx[i - LOOKBACK] != LOOKBACK + maxh:
            continue
        lo = i - LOOKBACK + 1
        hist = [wl[times[j]] / almwl for j in range(lo, i + 1)]   # ← 경보수위 정규화

        ch = [hist,
              [0.0] + [hist[k] - hist[k - 1] for k in range(1, len(hist))],
              [0.0] * 3 + [hist[k] - hist[k - 3] for k in range(3, len(hist))],
              (rain_seq[lo:i + 1] / RAIN_SCALE).tolist()]
        for w in (3, 6, 12, 24):
            ch.append([(rcum[k + 1] - rcum[max(0, k - w + 1)]) / RAIN_SCALE
                       for k in range(lo, i + 1)])
        ch.append([math.sin(2 * math.pi * int(times[j][4:6]) / 12)
                   for j in range(lo, i + 1)])

        Xs.append(ch)
        Ys.append([(wl[times[i + h]] - wl[times[i]]) / almwl for h in HORIZONS])
        curs.append(wl[times[i]] / almwl)
        Ts.append(times[i])
    if not Xs:
        return None
    return (np.array(Xs, np.float32), np.array(Ys, np.float32),
            np.array(curs, np.float32), Ts)


def main():
    stations = json.load(open(os.path.join(TS, "train_stations.json"), encoding="utf-8"))
    rain_cache = {}
    parts = []
    print("■ 관측소별 데이터셋 구축")
    for s in stations:
        code, alm = s["code"], s.get("alm")
        if not alm or alm <= 0:
            continue
        clean = os.path.join(TS, f"wl_{code}_1H_flood_clean.csv")
        raw = os.path.join(TS, f"wl_{code}_1H_flood.csv")
        path = clean if os.path.exists(clean) else raw
        wl = load_wl(path)
        if len(wl) < MIN_ROWS:
            print(f"  {code} {str(s.get('name'))[:14]:<16} 건너뜀 ({len(wl)}건)")
            continue
        stn = s.get("asos", 131)
        if stn not in rain_cache:
            rain_cache[stn] = load_rain(stn)
        got = build_one(wl, rain_cache[stn], alm)
        if got is None:
            print(f"  {code} 표본 생성 실패")
            continue
        X, Y, cur, T = got
        rise = int((Y[:, 2] >= 0.3 / alm).sum())
        print(f"  {code} {str(s.get('name'))[:14]:<16} {len(X):>7,}표본  급상승 {rise:>5}")
        parts.append((code, X, Y, cur, T))

    if not parts:
        raise SystemExit("사용 가능한 관측소 없음 — 수집이 끝났는지 확인")

    # 관측소 단위 분할 (코드 해시로 재현 가능하게)
    codes = sorted(p[0] for p in parts)
    n = len(codes)
    n_te = max(1, int(n * 0.15)); n_va = max(1, int(n * 0.15))
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    te = {codes[i] for i in perm[:n_te]}
    va = {codes[i] for i in perm[n_te:n_te + n_va]}

    out = {}
    for split in ("train", "val", "test"):
        sel = [p for p in parts
               if (p[0] in te if split == "test" else
                   p[0] in va if split == "val" else
                   p[0] not in te and p[0] not in va)]
        if not sel:
            continue
        X = np.concatenate([p[1] for p in sel])
        Y = np.concatenate([p[2] for p in sel])
        C = np.concatenate([p[3] for p in sel])
        out[split] = (X, Y, C, [p[0] for p in sel])

    print(f"\n■ 관측소 단위 분할")
    for k, (X, Y, C, cs) in out.items():
        rise = int((Y[:, 2] >= 0.03).sum())     # 정규화 기준 3% ≈ 경보수위의 3%
        print(f"  {k:<6} 관측소 {len(cs):>2}개  표본 {len(X):>8,}  급상승 {rise:>6,}")

    np.savez_compressed(
        os.path.join(TS, "multi_dataset.npz"),
        **{f"{k}_{n}": v for k, (X, Y, C, cs) in out.items()
           for n, v in (("X", X), ("Y", Y), ("cur", C))},
        meta=json.dumps({k: v[3] for k, v in out.items()}, ensure_ascii=False))
    print(f"\n  → {os.path.join(TS, 'multi_dataset.npz')}")


if __name__ == "__main__":
    main()
