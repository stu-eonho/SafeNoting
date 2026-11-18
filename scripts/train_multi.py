# -*- coding: utf-8 -*-
"""
전국 다중 관측소 학습 + learning rate 탐색

이전 단일 관측소 학습의 두 결함을 여기서 바로잡는다.

  ① validation이 없어 test로 모델을 골랐다 (데이터 누수)
     → train/val/test를 관측소 단위로 나누고, 하이퍼파라미터·조기종료는 val로만 결정.
       test는 최종 1회만 평가한다.

  ② learning rate를 탐색하지 않았다 (1e-3 관행값 고정)
     → 후보를 val 기준으로 비교한다.

수위는 이미 경보수위로 정규화되어 있으므로(build_multi_dataset.py),
RevIN은 쓰지 않는다. 창별 평균 제거는 절대 수위 정보를 지워
성능을 26% 떨어뜨린다는 것이 실험으로 확인됐다.

사용:
    python scripts/train_multi.py --epochs 30
    python scripts/train_multi.py --epochs 15 --no-transformer   # DLinear 단독 비교
"""
import sys, os, json, time, argparse
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = os.path.join(BASE, "data", "timeseries")
sys.path.insert(0, os.path.join(BASE, "scripts"))

RISE_TH = 0.03      # 정규화 기준 급상승 = 경보수위의 3%


def load():
    p = os.path.join(TS, "multi_dataset.npz")
    if not os.path.exists(p):
        raise SystemExit("multi_dataset.npz 없음 — build_multi_dataset.py 먼저 실행")
    d = np.load(p, allow_pickle=True)
    out = {}
    for k in ("train", "val", "test"):
        if f"{k}_X" in d:
            out[k] = (d[f"{k}_X"], d[f"{k}_Y"], d[f"{k}_cur"])
    meta = json.loads(str(d["meta"]))
    return out, meta


def rise_weighted_mse(pred, true, w_rise=15.0, w_big=40.0):
    """
    급상승 표본 가중 MSE.

    ⚠ 가중을 |true|에 비례시키면 정규화 단위에서 무력해진다.
      수위를 경보수위로 나누면서 Y가 1/8로 작아졌는데(0.3m → 0.037),
      w=8인 비례 가중은 3.4배에서 1.3배로 떨어져 사실상 사라졌다.
      급상승 표본을 15배 늘려놓고 정작 가중이 걸리지 않아 loss가 3 epoch 만에 멈췄다.
      따라서 크기 비례가 아니라 '임계 기반'으로 명확히 가중한다.
    """
    import torch
    a = true.abs()
    w = torch.ones_like(a)
    w = torch.where(a >= 0.03, torch.full_like(a, w_rise), w)   # 경보수위 3% 이상
    w = torch.where(a >= 0.10, torch.full_like(a, w_big), w)    # 경보수위 10% 이상
    return ((pred - true).pow(2) * w).mean()


def evaluate(model, X, Y, cur, batch=4096):
    import torch
    model.eval()
    ps = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            o, _, _ = model(torch.tensor(X[i:i + batch]))
            ps.append(o.numpy())
    model.train()
    p = cur + np.concatenate(ps)[:, 2]
    y = cur + Y[:, 2]
    rise = Y[:, 2] >= RISE_TH
    res = {"all": float(np.sqrt(np.mean((p - y) ** 2)))}
    if rise.sum():
        res["rise"] = float(np.sqrt(np.mean((p[rise] - y[rise]) ** 2)))
        res["under"] = float(np.mean(p[rise] < y[rise]) * 100)
        res["n"] = int(rise.sum())
    return res


def run(lr, data, epochs, batch, dim, layers, dropout, seed=42, use_transformer=True):
    import torch
    from forecast_model import build_model
    torch.manual_seed(seed); np.random.seed(seed)

    Xtr, Ytr, Ctr = data["train"]
    n_var, lookback, n_out = Xtr.shape[1], Xtr.shape[2], Ytr.shape[1]
    # 수위 계열 3채널(정규화수위, Δ1h, Δ3h) — 나머지는 강수·시간
    model = build_model(n_var, lookback, n_out, n_wl_ch=3,
                        d_model=dim, n_layer=layers, dropout=dropout,
                        use_revin=False, use_transformer=use_transformer)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt = torch.from_numpy(Xtr); Yt = torch.from_numpy(Ytr)   # 복사 없이 메모리 공유
    n = len(Xt)
    best = {"rise": 9e9, "ep": -1, "state": None}

    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            out, _, _ = model(Xt[idx])
            loss = rise_weighted_mse(out, Yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.detach().item() * len(idx)
        sched.step()
        if (ep + 1) % 3 == 0 or ep == 0 or ep == epochs - 1:
            v = evaluate(model, *data["val"])
            mark = ""
            if v.get("rise", 9e9) < best["rise"]:
                best = {"rise": v["rise"], "ep": ep + 1,
                        "state": {k: t.clone() for k, t in model.state_dict().items()},
                        "under": v.get("under")}
                mark = " ★"
            print(f"    lr={lr:<7g} ep {ep+1:>3} loss {tot/n:.5f} "
                  f"val_all {v['all']:.4f} val_rise {v.get('rise', float('nan')):.4f} "
                  f"under {v.get('under', float('nan')):.1f}%{mark}", flush=True)
    return model, best


def main(epochs=30, batch=512, dim=64, layers=2, dropout=0.2, lrs=(1e-4,3e-4,1e-3,3e-3),
         use_transformer=True):
    tag = "multi" if use_transformer else "dlinear"
    try:
        import torch
    except ImportError:
        raise SystemExit("torch 미설치")
    data, meta = load()
    print("■ 데이터")
    for k, (X, Y, C) in data.items():
        rise = int((Y[:, 2] >= RISE_TH).sum())
        print(f"  {k:<6} 표본 {len(X):>8,}  급상승 {rise:>7,}  관측소 {len(meta.get(k, []))}개")
    if "val" not in data or "test" not in data:
        raise SystemExit("val/test 분할이 없음")

    print(f"\n■ learning rate 탐색 (val 기준)")
    results = {}
    for lr in lrs:
        t0 = time.time()
        model, best = run(lr, data, epochs, batch, dim, layers, dropout,
                          use_transformer=use_transformer)
        results[lr] = (model, best)
        # lr마다 즉시 저장한다 — 전부 끝나야 저장되면 중간에 끊을 수 없다.
        # 실제로 "lr 차이가 없으니 나머지는 건너뛰자"는 판단을 못 살렸다.
        if best["state"]:
            torch.save({"state": best["state"], "lr": lr, "val": best["rise"],
                        "ep": best["ep"], "dim": dim, "layers": layers},
                       os.path.join(TS, f"model_{tag}_lr{lr:g}.pt"))
        print(f"  → lr={lr:g} 최고 val_rise {best['rise']:.4f} "
              f"(ep {best['ep']}, under {best.get('under', float('nan')):.1f}%) "
              f"[{time.time()-t0:.0f}s]\n", flush=True)

    best_lr = min(results, key=lambda k: results[k][1]["rise"])
    model, best = results[best_lr]
    print(f"■ 선택된 lr = {best_lr:g}  (val_rise {best['rise']:.4f})")

    # ── test는 여기서 딱 한 번만 평가한다 ──
    if best["state"]:
        model.load_state_dict(best["state"])
    te = evaluate(model, *data["test"])
    print(f"\n■ TEST (처음 보는 관측소, 최종 1회 평가)")
    print(f"  전체 RMSE   {te['all']:.4f}")
    print(f"  급상승 RMSE {te.get('rise', float('nan')):.4f}  (n={te.get('n', 0)})")
    print(f"  과소예측    {te.get('under', float('nan')):.1f}%")
    print(f"\n  ※ 값은 '경보수위 대비 비율' 단위다. 실제 미터로 보려면 관측소 경보수위를 곱한다.")
    print(f"     예: 경보 8.0m 하천에서 0.05 → 0.40m 오차")

    torch.save({"state": model.state_dict(), "lr": best_lr,
                "val": best["rise"], "test": te, "dim": dim, "layers": layers,
                "use_transformer": use_transformer},
               os.path.join(TS, f"model_{tag}.pt"))
    print(f"\n  → data/timeseries/model_{tag}.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lrs", default="1e-4,3e-4,1e-3,3e-3")
    ap.add_argument("--no-transformer", action="store_true",
                    help="경로 A(DLinear)만 학습한다. 결과는 model_dlinear.pt")
    a = ap.parse_args()
    main(a.epochs, a.batch, a.dim, a.layers, a.dropout,
         tuple(float(x) for x in a.lrs.split(",")), use_transformer=not a.no_transformer)
