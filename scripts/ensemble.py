# -*- coding: utf-8 -*-
"""
GBDT 튜닝 + 3모델 앙상블 (선형 / GBDT / Transformer)

세 모델이 각각 다른 것을 잘한다는 관찰에서 출발한다.
    선형회귀    경보 판정 정밀도가 높다 (헛경보가 적다)
    GBDT       균형이 좋고 설명 가능하다
    Transformer 급상승 RMSE와 과소예측률이 우수하다

앙상블 가중치는 **val에서만** 탐색하고 test는 최종 1회만 본다.
튜닝 전/후 앙상블을 각각 측정해 "튜닝 효과"와 "앙상블 효과"를 분리한다.

사용:
    python scripts/ensemble.py
"""
import sys, os, json, time, itertools
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = os.path.join(BASE, "data", "timeseries")
sys.path.insert(0, os.path.join(BASE, "scripts"))
from train_gbdt import featurize, NAMES, HORIZON_IDX, RISE_TH

# ⚠ 경보수위(1.0) 초과는 train 0.027%, val 0건으로 학습·평가가 불가능하다.
#   실제 하천 경보도 관심/주의/경계/심각 단계로 운영되므로,
#   '주의 단계 = 경보수위의 70%'를 판정 기준으로 삼는다. (val 150건 / test 364건)
ALARM = 0.70


def evaluate(name, p_lvl, t_lvl, rise, quiet=False):
    all_rmse = float(np.sqrt(np.mean((p_lvl - t_lvl) ** 2)))
    rise_rmse = float(np.sqrt(np.mean((p_lvl[rise] - t_lvl[rise]) ** 2)))
    under = float(np.mean(p_lvl[rise] < t_lvl[rise]) * 100)
    pe, te = p_lvl >= ALARM, t_lvl >= ALARM
    tp = int((pe & te).sum()); fp = int((pe & ~te).sum()); fn = int((~pe & te).sum())
    rec = tp / (tp + fn) if tp + fn else float("nan")
    pre = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * pre * rec / (pre + rec) if pre + rec else float("nan")
    if not quiet:
        print(f"  {name:<26}{all_rmse:>8.4f}{rise_rmse:>10.4f}{under:>8.1f}%"
              f"{rec*100:>8.1f}%{pre*100:>8.1f}%{f1:>7.3f}")
    return dict(all=all_rmse, rise=rise_rmse, under=under, rec=rec, pre=pre, f1=f1)


def ridge(X, y, lam=1.0):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1]); A[-1, -1] -= lam
    return np.linalg.solve(A, Xb.T @ y)


def main():
    import lightgbm as lgb
    d = np.load(os.path.join(TS, "multi_dataset.npz"), allow_pickle=True)
    F, Y, C = {}, {}, {}
    for k in ("train", "val", "test"):
        F[k] = featurize(d[f"{k}_X"]); Y[k] = d[f"{k}_Y"][:, HORIZON_IDX]; C[k] = d[f"{k}_cur"]
    Xraw = {k: d[f"{k}_X"] for k in ("train", "val", "test")}

    w_tr = np.ones(len(Y["train"]), np.float32)
    w_tr[np.abs(Y["train"]) >= RISE_TH] = 15.0
    w_tr[np.abs(Y["train"]) >= 0.10] = 40.0

    def fit_gbdt(params, rounds=3000):
        ds_tr = lgb.Dataset(F["train"], Y["train"], weight=w_tr, feature_name=NAMES)
        ds_va = lgb.Dataset(F["val"], Y["val"], reference=ds_tr, feature_name=NAMES)
        m = lgb.train(params, ds_tr, num_boost_round=rounds, valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        return m

    BASE_P = dict(objective="regression", metric="l2", learning_rate=0.05,
                  num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbosity=-1, num_threads=8)

    # ── 1. 튜닝 전 GBDT ─────────────────────────────────────
    print("■ 1단계: 튜닝 전 GBDT")
    t0 = time.time()
    gbdt0 = fit_gbdt(BASE_P)
    print(f"  best_iter {gbdt0.best_iteration} · {time.time()-t0:.0f}s")

    # ── 2. GBDT 튜닝 (val 기준) ─────────────────────────────
    print(f"\n■ 2단계: GBDT 튜닝 (val 급상승 RMSE 기준)")
    grid = list(itertools.product([0.01, 0.02, 0.05], [31, 63, 127], [50, 200]))
    rise_va = Y["val"] >= RISE_TH
    best = (9e9, None, None)
    for lr, leaves, mdl in grid:
        p = dict(BASE_P, learning_rate=lr, num_leaves=leaves, min_data_in_leaf=mdl)
        m = fit_gbdt(p)
        pv = C["val"] + m.predict(F["val"], num_iteration=m.best_iteration)
        tv = C["val"] + Y["val"]
        r = float(np.sqrt(np.mean((pv[rise_va] - tv[rise_va]) ** 2)))
        mark = ""
        if r < best[0]:
            best = (r, p, m); mark = " ★"
        print(f"    lr={lr:<5} leaves={leaves:<4} min_leaf={mdl:<4} "
              f"iter={m.best_iteration:<5} val_rise {r:.4f}{mark}", flush=True)
    gbdt1 = best[2]
    print(f"\n  선택: {dict((k,v) for k,v in best[1].items() if k in ('learning_rate','num_leaves','min_data_in_leaf'))}")
    print(f"  val 급상승 RMSE {best[0]:.4f}")

    # ── 3. 나머지 모델 예측 ─────────────────────────────────
    w_lin = ridge(F["train"], Y["train"])
    def lin_pred(k): return np.hstack([F[k], np.ones((len(F[k]), 1))]) @ w_lin

    tr_pred = {}
    p_model = os.path.join(TS, "model_multi.pt")
    if os.path.exists(p_model):
        import torch
        from forecast_model import build_model
        ck = torch.load(p_model, weights_only=False)
        net = build_model(Xraw["train"].shape[1], Xraw["train"].shape[2], 4,
                          n_wl_ch=3, d_model=64, n_layer=2, dropout=0.2, use_revin=False)
        net.load_state_dict(ck["state"]); net.eval()
        for k in ("val", "test"):
            outs = []
            with torch.no_grad():
                for i in range(0, len(Xraw[k]), 4096):
                    o, _, _ = net(torch.tensor(Xraw[k][i:i + 4096]))
                    outs.append(o.numpy())
            tr_pred[k] = np.concatenate(outs)[:, HORIZON_IDX]

    # ── 4. 앙상블 가중치 탐색 (val) ─────────────────────────
    def preds(k, gb):
        out = {"lin": lin_pred(k), "gbdt": gb.predict(F[k], num_iteration=gb.best_iteration)}
        if k in tr_pred: out["trf"] = tr_pred[k]
        return out

    def f2_score(p_lvl, t_lvl, th=ALARM):
        """재난에서는 놓치는 비용이 헛경보보다 크므로 재현율에 2배 가중한 F2를 쓴다."""
        pe, te = p_lvl >= th, t_lvl >= ALARM
        tp = int((pe & te).sum()); fp = int((pe & ~te).sum()); fn = int((~pe & te).sum())
        if tp == 0: return 0.0
        rec = tp / (tp + fn); pre = tp / (tp + fp)
        return 5 * pre * rec / (4 * pre + rec) if (4 * pre + rec) else 0.0

    def search_w(gb, objective="rmse"):
        """
        objective='rmse'  → 급상승 RMSE 최소화
        objective='f2'    → 경보 판정 F2 최대화 (문턱도 함께 탐색)

        두 목적의 최적 조합이 다르다. 시스템이 실제로 수행하는 일이
        '수위를 맞히는 것'이 아니라 '위험 여부를 판정하는 것'이므로 f2가 본령이다.
        """
        pv = preds("val", gb); tv = C["val"] + Y["val"]; r = Y["val"] >= RISE_TH
        keys = [k for k in ("lin", "gbdt", "trf") if k in pv]
        bw, bs, bth = None, (9e9 if objective == "rmse" else -1), ALARM
        for combo in itertools.product(np.arange(0, 1.01, 0.1), repeat=len(keys)):
            s = sum(combo)
            if s == 0: continue
            wts = np.array(combo) / s
            p = C["val"] + sum(wt * pv[k] for wt, k in zip(wts, keys))
            if objective == "rmse":
                sc = float(np.sqrt(np.mean((p[r] - tv[r]) ** 2)))
                if sc < bs: bs, bw = sc, dict(zip(keys, wts))
            else:
                for th in (0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50):
                    sc = f2_score(p, tv, th)
                    if sc > bs: bs, bw, bth = sc, dict(zip(keys, wts)), th
        return bw, bs, bth

    print(f"\n■ 3단계: 앙상블 가중치 탐색 (val)")
    w0, s0, _ = search_w(gbdt0, "rmse")
    w1, s1, _ = search_w(gbdt1, "rmse")
    print(f"  [RMSE 기준] 튜닝전 {({k:round(v,2) for k,v in w0.items()})} val {s0:.4f}")
    print(f"  [RMSE 기준] 튜닝후 {({k:round(v,2) for k,v in w1.items()})} val {s1:.4f}")
    wf0, sf0, th0 = search_w(gbdt0, "f2")
    wf1, sf1, th1 = search_w(gbdt1, "f2")
    print(f"  [F2 기준]   튜닝전 {({k:round(v,2) for k,v in wf0.items()})} 문턱 {th0:.2f} val F2 {sf0:.3f}")
    print(f"  [F2 기준]   튜닝후 {({k:round(v,2) for k,v in wf1.items()})} 문턱 {th1:.2f} val F2 {sf1:.3f}")

    # ── 5. TEST 최종 평가 ───────────────────────────────────
    t_lvl = C["test"] + Y["test"]; rise = Y["test"] >= RISE_TH
    print(f"\n■ TEST 최종 (처음 보는 관측소 7곳 · 급상승 {int(rise.sum()):,}건)")
    print(f"  {'모델':<26}{'전체':>8}{'급상승':>10}{'과소':>9}{'재현율':>8}{'정밀도':>8}{'F1':>7}")
    print("  " + "─" * 76)
    evaluate("Persistence", C["test"], t_lvl, rise)
    evaluate("선형회귀", C["test"] + lin_pred("test"), t_lvl, rise)
    p0 = preds("test", gbdt0); p1 = preds("test", gbdt1)
    evaluate("GBDT (튜닝 전)", C["test"] + p0["gbdt"], t_lvl, rise)
    evaluate("GBDT (튜닝 후)", C["test"] + p1["gbdt"], t_lvl, rise)
    if "trf" in p0:
        evaluate("Transformer", C["test"] + p0["trf"], t_lvl, rise)
    print("  " + "─" * 76)
    evaluate("앙상블 RMSE기준(튜닝전)", C["test"] + sum(w0[k]*p0[k] for k in w0), t_lvl, rise)
    evaluate("앙상블 RMSE기준(튜닝후)", C["test"] + sum(w1[k]*p1[k] for k in w1), t_lvl, rise)
    mf0 = C["test"] + sum(wf0[k]*p0[k] for k in wf0)
    mf1 = C["test"] + sum(wf1[k]*p1[k] for k in wf1)
    evaluate("앙상블 F2기준(튜닝전)", mf0, t_lvl, rise)
    evaluate("앙상블 F2기준(튜닝후)", mf1, t_lvl, rise)

    print(f"\n■ 경보 판정 — F2 기준 앙상블에 최적 문턱 적용")
    print(f"  {'모델':<26}{'문턱':>6}{'재현율':>9}{'정밀도':>9}{'놓침':>7}{'헛경보':>8}{'F2':>7}")
    for nm, p_lvl, th in [("선형회귀", C["test"]+lin_pred("test"), ALARM),
                          ("GBDT(튜닝전)", C["test"]+p0["gbdt"], ALARM),
                          ("GBDT(튜닝후)", C["test"]+p1["gbdt"], ALARM),
                          ("Transformer", C["test"]+p0["trf"], ALARM),
                          ("앙상블 F2기준(튜닝전)", mf0, th0),
                          ("앙상블 F2기준(튜닝후)", mf1, th1)]:
        pe, te = p_lvl >= th, t_lvl >= ALARM
        tp = int((pe & te).sum()); fp = int((pe & ~te).sum()); fn = int((~pe & te).sum())
        rec = tp/(tp+fn) if tp+fn else 0; pre = tp/(tp+fp) if tp+fp else 0
        f2 = 5*pre*rec/(4*pre+rec) if (4*pre+rec) else 0
        print(f"  {nm:<26}{th:>6.2f}{rec*100:>8.1f}%{pre*100:>8.1f}%{fn:>7}{fp:>8}{f2:>7.3f}")

    print(f"\n■ 피처 중요도 상위 10 (튜닝 후 GBDT)")
    imp = sorted(zip(NAMES, gbdt1.feature_importance("gain")), key=lambda z: -z[1])
    tot = sum(v for _, v in imp) or 1
    for nm, v in imp[:10]:
        print(f"  {nm:<14} {v/tot*100:>5.1f}%  {'█'*int(v/tot*50)}")


if __name__ == "__main__":
    main()
