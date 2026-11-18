# -*- coding: utf-8 -*-
"""
수위 예측 모델 — Linear 병렬 + Transformer 게이트 구조

설계 근거는 docs/11-forecast-model-design.md 참조. 요지만 옮기면:

  ┌ 경로 A: DLinear (추세)          ┐
  │                                 ├─ 출력 = A + α·B    (α는 학습되는 게이트)
  └ 경로 B: iTransformer (비선형 잔차)┘

  · Transformer가 학습에 실패해도 α→0이 되어 DLinear 성능이 하한으로 남는다
  · 학습 후 α와 경로 B의 기여도를 재면 "Transformer가 필요했는가"에 답할 수 있다
  · 변수를 토큰으로 쓰는 iTransformer 방식 — 상류→하류 관계가 이 문제의 본질이므로
    시간 패치를 쓰는 PatchTST(변수 독립)는 부적합하다

사용:
    python scripts/forecast_model.py                 # 학습 + 평가
    python scripts/forecast_model.py --epochs 30
"""
import sys, os, csv, math, time, argparse
from datetime import datetime
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = os.path.join(BASE, "data", "timeseries")

TARGET = "3011665"
# 시차상관으로 확정된 상류 (docs/11 §4). 미호교 3011685는 lag +1h(하류)라 영구 제외.
UPSTREAM = {"3011635": 2, "3011660": 3, "3011645": 4, "3011653": 4, "3011663": 6}
ALMWL = 8.0
HORIZONS = [1, 2, 3, 6]
LOOKBACK = 48


# ────────────────────────────────────────────────────────────
# 데이터
# ────────────────────────────────────────────────────────────
def load_wl(sta):
    clean = os.path.join(TS, f"wl_{sta}_1H_clean.csv")
    raw = os.path.join(TS, f"wl_{sta}_1H.csv")
    p = clean if os.path.exists(clean) else raw
    if not os.path.exists(p):
        return {}
    out = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("flag"):
                continue
            try:
                out[r["ymdhm"]] = float(r["wl"])
            except (ValueError, KeyError):
                pass
    return out


def load_rain(stn="131"):
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


def build_dataset():
    """
    반환: X (N, V, L)  변수 V개 × 시계열 길이 L  ← iTransformer는 변수를 토큰으로 쓴다
          Y (N, H)     각 horizon의 Δ수위
          cur (N,)     현재 수위
          T  (N,)      시각 문자열
    """
    wl = load_wl(TARGET)
    if not wl:
        raise SystemExit("수위 데이터 없음 — collect_waterlevel.py 먼저 실행")
    rain = load_rain()
    # 실제로 수집된 상류만 채널로 쓴다.
    # 미수집 관측소를 0으로 채워 넣으면 상수 채널이 되어 학습에 해가 된다.
    ups = {s: d for s, d in ((s, load_wl(s)) for s in UPSTREAM) if d}
    active = list(ups.keys())
    skipped = [s for s in UPSTREAM if s not in ups]
    print(f"  타겟 {len(wl):,}건 · 강수 {len(rain):,}건 · 상류 {len(active)}/{len(UPSTREAM)}곳 사용 {active}")
    if skipped:
        print(f"  ※ 미수집으로 제외된 상류: {skipped}")

    times = sorted(wl)
    maxh = max(HORIZONS)

    # 누적 강수는 표본마다 다시 더하면 O(N·L·W)로 느려진다.
    # 전체 시계열의 누적합을 한 번 만들어두고 차분으로 O(1) 조회한다.
    rain_seq = np.array([rain.get(t, 0.0) for t in times], dtype=np.float64)
    rain_cum = np.concatenate([[0.0], np.cumsum(rain_seq)])

    def rain_acc(k, w):
        lo = max(0, k - w + 1)
        return float(rain_cum[k + 1] - rain_cum[lo])

    # ⚠ 인접 레코드가 1시간 간격이라는 보장이 없다 (상류 관측소는 수개월 결측이 있다).
    #   절대 시각 인덱스로 창 전체의 연속성을 확인한다.
    hidx = [int(datetime.strptime(t[:12], "%Y%m%d%H%M").timestamp() // 3600)
            for t in times]

    Xs, Ys, curs, Ts = [], [], [], []
    skipped_gap = 0
    for i in range(LOOKBACK, len(times) - maxh):
        if hidx[i + maxh] - hidx[i - LOOKBACK] != LOOKBACK + maxh:
            skipped_gap += 1
            continue          # 창 안에 결측 구간이 있음
        t_now = times[i]

        chans = []
        # ① 타겟 수위
        hist = [wl[times[j]] for j in range(i - LOOKBACK + 1, i + 1)]
        chans.append(hist)
        # ①-b 수위 변화량 — 상승 국면인지 하강 국면인지는 절대 수위보다 중요하다
        chans.append([0.0] + [hist[k] - hist[k - 1] for k in range(1, len(hist))])
        chans.append([0.0] * 3 + [hist[k] - hist[k - 3] for k in range(3, len(hist))])
        # ② 상류 수위 — 물리적으로 측정된 lag만큼 시프트해서 넣는다 (docs/11 §4)
        for s in active:
            d, lag = ups[s], UPSTREAM[s]
            ch = []
            last = 0.0
            for j in range(i - LOOKBACK + 1, i + 1):
                k = j - lag
                v = d.get(times[k]) if 0 <= k < len(times) else None
                # 결측은 직전 관측값으로 유지 (0으로 두면 가짜 급락 신호가 된다)
                last = v if v is not None else last
                ch.append(last)
            chans.append(ch)
        # ③ 강수 — 시간당 raw
        lo = i - LOOKBACK + 1
        chans.append(rain_seq[lo: i + 1].tolist())
        # ③-b 누적 강수. 수위를 움직이는 것은 순간 강수가 아니라 누적량이다.
        #     baseline이 이 피처를 명시적으로 갖고 있었으므로 동등한 조건을 만든다.
        for w in (3, 6, 12, 24):
            chans.append([rain_acc(k, w) for k in range(lo, i + 1)])
        # ④ 시간 인코딩 (연중 주기)
        mo = [math.sin(2 * math.pi * int(times[j][4:6]) / 12)
              for j in range(i - LOOKBACK + 1, i + 1)]
        chans.append(mo)

        Xs.append(chans)
        Ys.append([wl[times[i + h]] - wl[t_now] for h in HORIZONS])
        curs.append(wl[t_now])
        Ts.append(t_now)

    if skipped_gap:
        print(f"  결측 구간을 가로지르는 창 {skipped_gap:,}개 제외")
    X = np.array(Xs, dtype=np.float32)
    Y = np.array(Ys, dtype=np.float32)
    # 채널 배치: [수위, Δ1h, Δ3h, 상류…] [강수 raw, 3h, 6h, 12h, 24h] [월]
    # 앞쪽 수위 계열만 RevIN 대상이므로 개수를 함께 넘긴다.
    n_wl_ch = 3 + len(active)
    return X, Y, np.array(curs, np.float32), Ts, n_wl_ch


# ────────────────────────────────────────────────────────────
# 모델
# ────────────────────────────────────────────────────────────
def build_model(n_var, lookback, n_out, n_wl_ch, d_model=64, n_head=4,
                n_layer=2, dropout=0.1, use_revin=True, use_transformer=True):
    """use_transformer=False 면 경로 A(DLinear)만 만든다. 결합 구조와의 비교 실험용."""
    import torch
    import torch.nn as nn

    class RevIN(nn.Module):
        """
        비정상 시계열 대응 — 입력 창마다 정규화하고, 출력에서 반드시 되돌린다.

        ⚠ norm()과 denorm()은 반드시 쌍으로 쓴다. 역정규화를 빠뜨리면
          모델이 표본마다 다른 스케일을 억지로 맞추게 되어 성능이 무너진다.
          (실제로 이 버그로 DLinear가 선형회귀보다 나쁘게 나왔다)

        ⚠ 수위 계열 채널에만 적용한다. 강수는 대부분 0이라 창 내 표준편차가
          0에 가깝고, 그걸로 나누면 미세한 값이 폭발적으로 증폭된다.
          (누적강수 채널에 RevIN을 걸었다가 홍수기 RMSE가 0.97→1.43으로 악화됨)
          강수는 고정 스케일로 나누고, 시간 인코딩은 이미 [-1,1]이라 그대로 둔다.
        """
        RAIN_SCALE = 30.0                        # mm — 고정 스케일
        WL_SCALE = 5.0                           # m — RevIN을 끌 때 쓰는 고정 스케일
        # 평수위 구간은 48시간 창의 표준편차가 0.005m 수준까지 내려간다(전체 창의 32%가 0.01m 미만).
        # 그대로 나누면 관측 잡음이 100배로 증폭되므로 하한을 둔다.
        SD_FLOOR = 0.05                          # m

        def __init__(self, use_revin=True):
            super().__init__()
            self.use_revin = use_revin

        def norm(self, x, n_wl):                 # x: (B, V, L)
            wl = x[:, :n_wl, :]
            if self.use_revin:
                self.mu = wl.mean(-1, keepdim=True).detach()
                sd = (wl.var(-1, keepdim=True, unbiased=False) + 1e-5).sqrt()
                self.sd = sd.clamp(min=self.SD_FLOOR).detach()
                z_wl = (wl - self.mu) / self.sd
            else:
                # ⚠ 창별 평균 제거는 '지금 몇 m인가'를 지운다.
                #   평수위 1.5m에서의 +0.3m와 위험수위 9m에서의 +0.3m가
                #   정규화 후 구별되지 않는데, 하천에서 절대 수위는 결정적 정보다.
                #   (baseline 선형회귀는 절대 수위를 그대로 써서 0.373을 낸다)
                #   고정 스케일만 적용해 절대 수위를 보존한다.
                z_wl = wl / self.WL_SCALE
                self.sd = torch.full_like(wl[:, :1, :1], self.WL_SCALE)
            rest = x[:, n_wl:, :]
            # 마지막 채널(시간 인코딩)은 이미 정규화되어 있다
            z_rain = rest[:, :-1, :] / self.RAIN_SCALE
            return torch.cat([z_wl, z_rain, rest[:, -1:, :]], dim=1)

        def denorm_delta(self, d):
            """정규화 공간의 Δ → 원래 스케일. Δ이므로 평균은 더하지 않고 표준편차만 곱한다."""
            return d * self.sd[:, 0, :]          # 타겟 채널(0)의 스케일

    class DLinearPath(nn.Module):
        """
        경로 A — 추세/잔차 분해 후 선형 사상. 이 문제의 성능 대부분을 담당한다.

        ⚠ 타겟 채널만 보게 만들면 안 된다.
          baseline ③(0.373)이 강한 이유는 선형 모델이 상류·강수를 함께 쓰기 때문이다.
          A를 타겟 전용으로 두면 A의 한계가 baseline ②(0.542) 수준에 묶이고,
          상류·강수 처리를 Transformer 혼자 떠맡게 되어 과적합한다.
          (실제로 그렇게 만들었다가 A 단독 0.60, 전체 0.517에 그쳤다)
          따라서 전 채널을 입력받아 채널별 선형 사상을 합산한다.
        """
        def __init__(self, n_var, lookback, n_out, kernel=25):
            super().__init__()
            self.kernel = kernel
            self.trend = nn.Linear(n_var * lookback, n_out)
            self.seasonal = nn.Linear(n_var * lookback, n_out)

        def forward(self, x):                    # x: (B, V, L) 전 채널
            pad = self.kernel // 2
            xp = torch.nn.functional.pad(x, (pad, pad), mode="replicate")
            tr = torch.nn.functional.avg_pool1d(xp, self.kernel, stride=1)
            tr = tr[..., :x.shape[-1]]
            se = x - tr
            return self.trend(tr.flatten(1)) + self.seasonal(se.flatten(1))

    class ITransformerPath(nn.Module):
        """경로 B — 변수를 토큰으로 두고 변수 간 attention. 상류→하류 관계를 학습한다."""
        def __init__(self, n_var, lookback, n_out, d_model, n_head, n_layer, dropout):
            super().__init__()
            self.embed = nn.Linear(lookback, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_head, dim_feedforward=d_model * 2,
                dropout=dropout, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, n_layer)
            self.head = nn.Linear(d_model * n_var, n_out)

        def forward(self, x):                    # x: (B, V, L)
            z = self.embed(x)                    # (B, V, d)  변수 = 토큰
            z = self.enc(z)
            return self.head(z.flatten(1))

    class GatedHybrid(nn.Module):
        def __init__(self):
            super().__init__()
            self.revin = RevIN(use_revin)
            self.a = DLinearPath(n_var, lookback, n_out)
            if use_transformer:
                self.b = ITransformerPath(n_var, lookback, n_out,
                                          d_model, n_head, n_layer, dropout)
                # α를 0에서 출발시킨다 — 처음에는 순수 DLinear이고,
                # Transformer가 실제로 기여할 때만 게이트가 열린다
                self.alpha = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            z = self.revin.norm(x, n_wl_ch)
            out_a = self.revin.denorm_delta(self.a(z))
            if not use_transformer:
                return out_a, out_a, torch.zeros_like(out_a)
            out_b = self.revin.denorm_delta(self.b(z))
            return out_a + self.alpha * out_b, out_a, out_b

    return GatedHybrid()


def asymmetric_loss(pred, true, w_under=3.0, w_rise=2.0):
    """
    재난 예측은 오차가 대칭이 아니다.
      과소예측 → 대피 판단 실패 (치명적)
      과대예측 → 헛걸음
    급상승 구간에도 가중을 걸어 평수위 표본에 묻히지 않게 한다.
    """
    import torch
    err = pred - true
    w = torch.where(err < 0, torch.tensor(w_under, device=err.device),
                    torch.tensor(1.0, device=err.device))
    w = w * (1.0 + w_rise * torch.clamp(true, min=0))
    return (err.pow(2) * w).mean()


# ────────────────────────────────────────────────────────────
# 평가 — baseline_forecast.py와 동일 기준
# ────────────────────────────────────────────────────────────
def evaluate(name, pred_d, true_d, cur, hi=2):
    """hi: HORIZONS 인덱스 (기본 3시간)"""
    p = cur + pred_d[:, hi]
    y = cur + true_d[:, hi]
    rise = (true_d[:, hi] >= 0.3)
    flood = (y >= ALMWL)
    def rmse(m):
        return float(np.sqrt(np.mean((p[m] - y[m]) ** 2))) if m.sum() else float("nan")
    under = float(np.mean(p[rise] < y[rise]) * 100) if rise.sum() else float("nan")
    print(f"  {name:<24} 전체 {rmse(np.ones_like(rise,bool)):.3f}"
          f"  홍수기 {rmse(flood):.3f}(n={int(flood.sum())})"
          f"  급상승 {rmse(rise):.3f}(n={int(rise.sum())})"
          f"  과소 {under:.1f}%")
    return rmse(rise), under


def mse_loss(pred, true):
    return (pred - true).pow(2).mean()


def rise_weighted_mse(pred, true, w_rise=8.0):
    """
    급상승 표본에 가중을 준 MSE.

    급상승(Δ3h≥0.3m) 표본은 전체의 0.96%뿐이라, 균등 MSE에서는 기울기가
    나머지 99%에 완전히 지배된다. 모델이 평수위를 소수점 아래로 다듬는 데
    용량을 쓰고 정작 필요한 구간은 학습되지 않는다.
    변화량 크기에 비례해 가중하여 이 편향을 교정한다.
    (과소예측 방향 가중은 별개 문제 — asymmetric_loss 참조)
    """
    import torch
    w = 1.0 + w_rise * true.abs()
    return ((pred - true).pow(2) * w).mean()


def main(epochs=25, batch=256, lr=1e-3, split_year="2020", loss_kind="mse",
         d_model=64, n_layer=2, dropout=0.1, use_revin=True):
    try:
        import torch
    except ImportError:
        raise SystemExit("torch 미설치 — pip install torch --index-url "
                         "https://download.pytorch.org/whl/cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    print("■ 데이터 준비")
    X, Y, cur, T, n_wl_ch = build_dataset()
    print(f"  표본 {len(X):,} · 변수 {X.shape[1]} (수위계열 {n_wl_ch}) "
          f"· 창 {X.shape[2]} · 출력 {Y.shape[1]}")

    tr = np.array([t[:4] < split_year for t in T])
    print(f"  학습 {int(tr.sum()):,} / 검증 {int((~tr).sum()):,}  (분할 {split_year})\n")

    Xtr = torch.tensor(X[tr]); Ytr = torch.tensor(Y[tr])
    Xte = torch.tensor(X[~tr])
    Yte_np, cur_te = Y[~tr], cur[~tr]

    model = build_model(X.shape[1], X.shape[2], Y.shape[1], n_wl_ch,
                        d_model=d_model, n_layer=n_layer, dropout=dropout,
                        use_revin=use_revin)
    n_param = sum(p.numel() for p in model.parameters())
    rise_n = int((Y[tr][:, 2] >= 0.3).sum())
    print(f"■ 모델 파라미터 {n_param:,}개 (d={d_model}, layer={n_layer}, "
          f"drop={dropout}, RevIN={'ON' if use_revin else 'OFF'})")
    print(f"  급상승 학습표본 {rise_n:,}개 → 파라미터/표본 = {n_param/max(rise_n,1):.0f}배")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(Xtr)

    # baseline은 MSE로 적합된 선형모델이다. 구조의 효과만 보려면 손실도 MSE여야 한다.
    # 비대칭 손실은 구조 검증이 끝난 뒤 별도 단계에서 얹는다 (docs/11 §9 단계 6).
    lossfn = {"mse": mse_loss, "rise": rise_weighted_mse,
              "asym": asymmetric_loss}[loss_kind]
    print(f"■ 학습 {epochs} epoch (CPU) · 손실 {loss_kind}", flush=True)
    print(f"    {'ep':>3} {'loss':>9} {'α':>8} {'검증RMSE':>9} {'급상승':>8} "
          f"{'과소%':>7} {'경과':>7}", flush=True)

    def quick_eval():
        """학습 중 검증 지표 — 진행 상황을 눈으로 확인하기 위한 것"""
        model.eval()
        with torch.no_grad():
            ps = []
            for i in range(0, len(Xte), 2048):
                o, _, _ = model(Xte[i:i + 2048])
                ps.append(o.numpy())
        pd_ = np.concatenate(ps)
        model.train()
        p = cur_te + pd_[:, 2]
        y = cur_te + Yte_np[:, 2]
        rise = Yte_np[:, 2] >= 0.3
        all_rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        rise_rmse = float(np.sqrt(np.mean((p[rise] - y[rise]) ** 2)))
        under = float(np.mean(p[rise] < y[rise]) * 100)
        return all_rmse, rise_rmse, under

    t0 = time.time()
    best = (9e9, -1)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            out, _, _ = model(Xtr[idx])
            loss = lossfn(out, Ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.detach().item() * len(idx)
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0 or ep == epochs - 1:
            ar, rr, un = quick_eval()
            flag = ""
            if rr < best[0]:
                best = (rr, ep + 1)
                flag = " ★"
            print(f"    {ep+1:>3} {tot/n:>9.5f} {float(model.alpha):>+8.4f} "
                  f"{ar:>9.3f} {rr:>8.3f} {un:>6.1f}% {time.time()-t0:>6.0f}s{flag}",
                  flush=True)
    print(f"\n  최고 급상승 RMSE {best[0]:.3f} (epoch {best[1]}) · "
          f"baseline 0.373 대비 {(0.373-best[0])/0.373*100:+.1f}%", flush=True)

    print(f"\n■ 평가 (3시간 예측)")
    model.eval()
    with torch.no_grad():
        preds, pa, pb = [], [], []
        for i in range(0, len(Xte), 1024):
            o, a, b = model(Xte[i:i + 1024])
            preds.append(o.numpy()); pa.append(a.numpy()); pb.append(b.numpy())
    pred = np.concatenate(preds); path_a = np.concatenate(pa); path_b = np.concatenate(pb)

    # 경로 A는 타겟 수위 채널만 보므로 baseline ②(수위 lag만)와 대응한다.
    # 전체 모델(A+αB)이 강수·상류까지 쓰므로 baseline ③과 대응한다.
    print(f"  [기준] baseline ② 수위만 0.542 · ③ 강수+상류 0.373 (과소 76.5%)")
    evaluate("경로A 단독(DLinear)", path_a, Yte_np, cur_te)
    r, u = evaluate("A + α·B (제안구조)", pred, Yte_np, cur_te)

    alpha = float(model.alpha)
    # 기여도는 "최종 출력의 변동 중 Transformer가 만든 몫"으로 잰다.
    # |αB|/|A| 같은 비는 A가 작을 때 무한대로 튀어 오해를 준다.
    contrib = float(np.std(alpha * path_b[:, 2]) /
                    (np.std(pred[:, 2]) + 1e-9) * 100)
    print(f"\n■ Transformer 기여도 판정")
    print(f"    α = {alpha:+.4f}")
    print(f"    std(α·B) / std(출력) = {contrib:.1f}%")
    if abs(alpha) < 0.01 or contrib < 5:
        print(f"    ▶ 게이트가 닫혔다 — 이 문제에 Transformer는 불필요.")
        print(f"      DLinear(경로 A) 단독으로 확정하고 그 근거로 α를 제시한다.")
    else:
        print(f"    ▶ Transformer가 실질적으로 기여 중 — 채택 검토.")
    if r <= 0.373:
        print(f"    ▶ 급상승 RMSE {r:.3f} ≤ 0.373 — baseline 상회 ✅")
    else:
        print(f"    ▶ 급상승 RMSE {r:.3f} > 0.373 — baseline 미달 ❌")

    torch.save(model.state_dict(), os.path.join(TS, "model_hybrid.pt"))
    print(f"\n  → 저장: data/timeseries/model_hybrid.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--split", default="2020")
    ap.add_argument("--loss", default="mse", choices=["mse", "rise", "asym"])
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-revin", action="store_true",
                    help="RevIN 끄기 — 절대 수위 정보를 보존한다")
    a = ap.parse_args()
    main(a.epochs, a.batch, a.lr, a.split, a.loss, a.dim, a.layers, a.dropout,
         not a.no_revin)
