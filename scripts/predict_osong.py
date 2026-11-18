# -*- coding: utf-8 -*-
"""
오송 참사 당일 재현 — 2023-07-15 각 시각에 모델이 예측한 3시간 뒤 수위

■ 관측소
  미호강교(3011665). 궁평2지하차도에서 약 478m.
  이 관측소 자체는 학습에 포함되지 않았다. 단, 같은 하천 하류의 미호교(3011685)는
  학습 관측소이며 같은 날 자료가 학습 데이터에 들어 있다.

■ 그날의 실측 (미호강교 1시간 자료)
  03시  7.02m
  05시  8.27m   경보수위 8.0m 초과
  08시  9.91m   심각수위 9.38m 초과
  08:40         궁평2지하차도 침수
  10시 10.08m   정점

■ 무엇을 재는가
  매 시각 t에서 48시간 과거만 보고 t+3h 수위를 예측하고, 단순 기준선과 비교한다.
    유지            3시간 뒤 = 현재
    3h 차분 외삽     현재 + (현재 - 3시간 전)
    1h 기울기 외삽   현재 + 3 x (현재 - 1시간 전)

사용:
    python scripts/predict_osong.py            # 표 출력
    python scripts/predict_osong.py --json     # data/routing/osong_replay.json 저장
"""
import sys, os, csv, json, math
from datetime import datetime
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = os.path.join(BASE, "data", "timeseries")
sys.path.insert(0, os.path.join(BASE, "scripts"))

STA = "3011665"
ALM = 8.0            # 경보수위
SRS = 9.38           # 심각수위
GDT = 19.643         # 영점표고 — 수면표고 = gdt + 수위
LOOKBACK = 48
RAIN_SCALE = 30.0
HORIZONS = [1, 2, 3, 6]
EVENT = "202307150840"


def infer_arch(state):
    """체크포인트에서 d_model·n_layer를 읽어낸다.

    구조를 코드에 박아두면 학습 설정을 바꿀 때마다 어긋난다.
    실제로 dim 32로 짜둔 채 dim 64 체크포인트를 읽어 전부 size mismatch가 났다.
    """
    d = int(state["b.enc.layers.0.norm1.weight"].shape[0])
    n = 1 + max((int(m.group(1)) for k in state
                 for m in [__import__("re").match(r"b\.enc\.layers\.(\d+)\.", k)] if m),
                default=0)
    return dict(d_model=d, n_layer=n)


def hour_idx(t):
    return int(datetime.strptime(t[:12], "%Y%m%d%H%M").timestamp() // 3600)


def load_wl():
    p = os.path.join(TS, f"wl_{STA}_1H_clean.csv")
    if not os.path.exists(p):
        p = os.path.join(TS, f"wl_{STA}_1H.csv")
    out = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r.get("flag"):
            continue
        try:
            out[r["ymdhm"]] = float(r["wl"])
        except (ValueError, KeyError):
            pass
    return out


def load_rain(stn=131):
    p = os.path.join(TS, f"asos_{stn}_1H.csv")
    out = {}
    if os.path.exists(p):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            tm = r.get("tm", "")
            if len(tm) >= 13:
                k = tm[:4] + tm[5:7] + tm[8:10] + tm[11:13] + "00"
                try:
                    out[k] = float(r.get("rn") or 0)
                except ValueError:
                    out[k] = 0.0
    return out


def window(times, wl, rain, i):
    """build_multi_dataset.build_one 과 동일한 9채널을 만든다."""
    lo = i - LOOKBACK + 1
    hist = [wl[times[j]] / ALM for j in range(lo, i + 1)]
    seq = np.array([rain.get(times[j], 0.0) for j in range(lo, i + 1)], np.float64)
    # 누적 강수는 창 바깥까지 필요하므로 원본에서 다시 뽑는다
    full = np.array([rain.get(t, 0.0) for t in times], np.float64)
    rcum = np.concatenate([[0.0], np.cumsum(full)])
    ch = [hist,
          [0.0] + [hist[k] - hist[k - 1] for k in range(1, len(hist))],
          [0.0] * 3 + [hist[k] - hist[k - 3] for k in range(3, len(hist))],
          (seq / RAIN_SCALE).tolist()]
    for w in (3, 6, 12, 24):
        ch.append([(rcum[k + 1] - rcum[max(0, k - w + 1)]) / RAIN_SCALE
                   for k in range(lo, i + 1)])
    ch.append([math.sin(2 * math.pi * int(times[j][4:6]) / 12)
               for j in range(lo, i + 1)])
    return np.array(ch, np.float32)


def main():
    import torch
    from forecast_model import build_model

    ck = os.path.join(TS, "model_multi.pt")
    if not os.path.exists(ck):
        raise SystemExit("model_multi.pt 없음 — 학습이 끝나야 한다")
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    model = build_model(9, LOOKBACK, len(HORIZONS), n_wl_ch=3,
                        **infer_arch(sd["state"]), dropout=0.2, use_revin=False)
    model.load_state_dict(sd["state"]); model.eval()
    print(f"■ 모델 로드 (학습 시 val_rise {sd.get('val', float('nan')):.4f})")

    wl = load_wl(); rain = load_rain()
    times = sorted(wl)
    idx = {t: k for k, t in enumerate(times)}
    hidx = [hour_idx(t) for t in times]

    print(f"■ 미호강교 {len(times):,}건 · 경보 {ALM}m · 심각 {SRS}m · 영점표고 {GDT}m")
    print(f"\n■ 2023-07-15 시간별 3시간 예측\n")
    print(f"  {'예측시각':<12}{'현재수위':>8}{'3h뒤예측':>10}{'실제':>8}{'오차':>8}   판정")

    rows = []
    for hh in range(0, 13):
        t = f"202307{15:02d}{hh:02d}00"
        if t not in idx:
            continue
        i = idx[t]
        if i < LOOKBACK or i + 3 >= len(times):
            continue
        if hidx[i] - hidx[i - LOOKBACK + 1] != LOOKBACK - 1:
            continue
        if hidx[i + 3] - hidx[i] != 3:      # 3시간 뒤 실측이 정확히 3칸 뒤여야 한다
            continue
        X = window(times, wl, rain, i)[None, ...]
        with torch.no_grad():
            out, _, _ = model(torch.tensor(X))
        d3 = float(out[0, 2]) * ALM
        cur = wl[t]
        pred = cur + d3
        act = wl[times[i + 3]]
        lv = "심각" if pred >= SRS else ("경보" if pred >= ALM else "-")
        base = {"유지": cur,
                "3h 차분 외삽": cur + (cur - wl[times[i - 3]]),
                "1h 기울기 외삽": cur + 3 * (cur - wl[times[i - 1]])}
        rows.append((t, cur, pred, act, lv, base))
        print(f"  {t[4:6]}-{t[6:8]} {t[8:10]}시   {cur:>7.2f}m{pred:>9.2f}m"
              f"{act:>7.2f}m{pred-act:>+7.2f}m   {lv}")

    act = np.array([r[3] for r in rows])
    print(f"\n■ 방법별 비교 (00~12시 {len(rows)}개 시점, 3시간 예측)\n")
    print(f"  {'방법':<12}{'RMSE':>8}{'최대오차':>9}   경보예측  심각예측")
    summary = {}
    for m in ["모델", "유지", "3h 차분 외삽", "1h 기울기 외삽"]:
        p = np.array([r[2] if m == "모델" else r[5][m] for r in rows])
        rmse = float(np.sqrt(np.mean((p - act) ** 2)))
        mx = float(np.max(np.abs(p - act)))
        fa = next((r[0][8:10] for r, v in zip(rows, p) if v >= ALM), None)
        fs = next((r[0][8:10] for r, v in zip(rows, p) if v >= SRS), None)
        summary[m] = {"rmse": round(rmse, 3), "max_err": round(mx, 3),
                      "first_alert_hour": fa, "first_severe_hour": fs}
        print(f"  {m:<12}{rmse:>7.3f}m{mx:>8.2f}m   {fa or '-':>4}시    {fs or '-':>4}시")
    print("\n  실제 궁평2지하차도 침수 08:40")

    if "--json" in sys.argv:
        out = {"station": STA, "alert_m": ALM, "severe_m": SRS, "horizon_h": 3,
               "source": "scripts/predict_osong.py, data/timeseries/model_multi.pt",
               "rows": [{"time": r[0], "current_m": round(r[1], 2),
                         "pred_m": round(r[2], 2), "actual_m": round(r[3], 2),
                         "baselines": {k: round(v, 2) for k, v in r[5].items()}}
                        for r in rows],
               "summary": summary}
        p = os.path.join(BASE, "data", "routing", "osong_replay.json")
        json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n  -> {p}")


if __name__ == "__main__":
    main()
