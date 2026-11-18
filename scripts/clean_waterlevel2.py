# -*- coding: utf-8 -*-
"""
수위 이상치 탐지 v2 — 하천 규모에 맞춰 판정한다

■ v1의 두 결함
  ① 강수 대조 지점이 청주(131)로 고정돼 있었다.
     관측소 59곳은 ASOS 25지점에 흩어져 있다. 춘천 하천을 청주 비와 대조하면
     "비 안 왔는데 물이 올랐다"가 아무 의미도 없는 판정이 된다.
  ② 변화율 임계 2.0m/h가 미호천 하나에서 뽑은 값이었다.
     실측 결과 정상 변동폭(p99.9)이 하천마다 0.27~0.89m/h로 3배 넘게 차이난다.
       양평교(경보 12.0m)  p99.9 0.27  최대 0.66   → 2.0은 너무 느슨
       학야교(경보  7.2m)  p99.9 0.89  최대 2.87   → 2.0은 진짜 홍수를 지운다

■ v2 판정 절차
  임계 = max(2.0, 4 x 그 관측소의 p99.9)   ← 낮추지 않는다. 급한 하천만 올린다
  급등이 임계를 넘고, 72시간 안에 급락이 오고,
  **급락 후 수위가 급등 전 수위 부근(스파이크 높이의 25% 이내)으로 복귀**하면 오류.

  진짜 홍수는 유출곡선을 그리며 서서히 빠지므로 72시간 안에 원위치로 돌아오지 않는다.
  센서 오류는 튀었다가 정확히 원래 값으로 떨어진다. 이 차이가 판별의 핵심이다.

  단, 직전 24시간 강수가 30mm 이상이면 판정을 보류한다(그 관측소의 ASOS 기준).
  비가 그만큼 왔다면 물이 오를 이유가 실재한다.

사용:
    python scripts/clean_waterlevel2.py            # 전체
    python scripts/clean_waterlevel2.py 3011665    # 한 곳
"""
import sys, os, csv, json
from datetime import datetime
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = os.path.join(BASE, "data", "timeseries")

RATE_FLOOR = 2.0     # 임계 하한 (미호천에서 검증된 값)
RATE_K = 4.0         # 그 하천 p99.9의 몇 배를 임계로 볼 것인가
WINDOW = 72          # 급등-급락 쌍을 찾을 범위 (시간)
RETURN_TOL = 0.25    # 복귀 판정: 스파이크 높이의 이 비율 이내면 '원위치'
RAIN_VETO = 30.0     # 직전 24h 강수가 이 이상이면 오류 판정 보류 (mm)

_rain_cache = {}


def load_rain(stn):
    if stn in _rain_cache:
        return _rain_cache[stn]
    p = os.path.join(TS, f"asos_{stn}_1H.csv")
    out = {}
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                tm = r.get("tm", "")
                if len(tm) >= 13:
                    k = tm[:4] + tm[5:7] + tm[8:10] + tm[11:13] + "00"
                    try:
                        out[k] = float(r.get("rn") or 0)
                    except ValueError:
                        out[k] = 0.0
    _rain_cache[stn] = out
    return out


def hour_idx(t):
    return int(datetime.strptime(t[:12], "%Y%m%d%H%M").timestamp() // 3600)


def clean_one(code, asos, verbose=True):
    src = os.path.join(TS, f"wl_{code}_1H_flood.csv")
    if not os.path.exists(src):
        return None
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    n = len(rows)
    ts = [r["ymdhm"] for r in rows]
    wl = [float(r["wl"]) if r.get("wl") not in (None, "") else None for r in rows]
    hi = [hour_idx(t) for t in ts]
    rain = load_rain(asos)

    # ── 그 하천의 정상 변동폭에서 임계를 정한다 ──────────────
    d = [abs(wl[i] - wl[i - 1]) for i in range(1, n)
         if wl[i] is not None and wl[i - 1] is not None and hi[i] - hi[i - 1] == 1]
    if len(d) < 500:
        return None
    p999 = float(np.percentile(d, 99.9))
    th = max(RATE_FLOOR, RATE_K * p999)

    def rain24(i):
        return sum(rain.get(ts[j], 0.0) for j in range(max(0, i - 23), i + 1))

    spikes = []
    for i in range(1, n):
        if wl[i] is None or wl[i - 1] is None or hi[i] - hi[i - 1] != 1:
            continue
        if abs(wl[i] - wl[i - 1]) >= th:
            spikes.append((i, wl[i] - wl[i - 1]))

    suspect, pairs = set(), []
    for i, delta in spikes:
        if delta <= 0:
            continue
        pre = wl[i - 1]                       # 급등 직전 수위
        for j, d2 in spikes:
            if not (j > i and d2 < 0 and hi[j] - hi[i] <= WINDOW):
                continue
            post = wl[j]
            # 핵심 — 급락 후 원위치로 돌아왔는가
            if abs(post - pre) <= RETURN_TOL * abs(delta):
                if rain24(i) >= RAIN_VETO:    # 비가 왔다면 판정 보류
                    break
                suspect.update(range(i, j + 1))
                pairs.append((ts[i], ts[j], pre, wl[i], post, rain24(i)))
                break

    flags = [""] * n
    for i in suspect:
        flags[i] = "SENSOR_ERR"

    dst = os.path.join(TS, f"wl_{code}_1H_flood_clean.csv")
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ymdhm", "wl", "fw", "flag"])
        for i, r in enumerate(rows):
            w.writerow([r["ymdhm"], r.get("wl", ""), r.get("fw", ""), flags[i]])
    return dict(code=code, n=n, th=th, p999=p999,
                flagged=len(suspect), pairs=pairs)


def main():
    sts = json.load(open(os.path.join(TS, "train_stations.json"), encoding="utf-8"))
    only = sys.argv[1] if len(sys.argv) > 1 else None
    tot_f, tot_n, changed = 0, 0, []
    print(f"{'관측소':<9}{'하천':<16}{'p99.9':>7}{'임계':>7}{'flag':>6}")
    for s in sts:
        if only and s["code"] != only:
            continue
        r = clean_one(s["code"], s.get("asos", 131))
        if not r:
            continue
        tot_f += r["flagged"]; tot_n += r["n"]
        if r["flagged"]:
            changed.append((s, r))
            print(f"{s['code']:<9}{s['name'][:14]:<16}{r['p999']:>7.2f}"
                  f"{r['th']:>7.2f}{r['flagged']:>6}")
    print(f"\n■ 총 flag {tot_f}건 / {tot_n:,}건 ({tot_f/max(tot_n,1)*100:.4f}%)")
    print(f"  v1 결과: 342건 · 10개소")
    print(f"  v2 결과: {tot_f}건 · {len(changed)}개소")

    if changed:
        print(f"\n■ 판정된 스파이크 (급등→급락 복귀 쌍)")
        for s, r in changed[:6]:
            for (t1, t2, pre, peak, post, rn) in r["pairs"][:2]:
                print(f"    {s['name'][:12]:<14} {t1[:10]} {pre:.2f}→{peak:.2f}m "
                      f"→ {t2[:10]} {post:.2f}m  (직전24h강수 {rn:.1f}mm)")


if __name__ == "__main__":
    main()
