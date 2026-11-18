# -*- coding: utf-8 -*-
"""
LightGBM(GBDT) 수위 예측 — Transformer·선형회귀와 동일 조건 비교

■ 왜 지금 트리 모델인가
  앞서 LightGBM을 폐기한 것은 '침수흔적도로 침수 여부를 학습'하려던 때였고,
  독립 강우사건이 3개뿐이라 성능 측정 자체가 불가능했기 때문이다.
  수위 예측은 조건이 전혀 다르다 — 표본 66만, 급상승 1.7만, 그리고
  시계열을 lag/누적/변화율로 펴면 전형적인 테이블 문제가 된다.
  이 영역에서 GBDT가 딥러닝을 능가하는 사례는 널리 보고돼 있다.

■ 알려진 약점
  트리는 학습 범위를 벗어난 값을 외삽하지 못한다. 기록적 홍수에 취약하다.
  다만 우리는 절대 수위가 아니라 '변화량(Δ)'을 예측하므로 상당히 완화된다.

사용:
    python scripts/train_gbdt.py
"""
import sys, os, json, time
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = os.path.join(BASE, "data", "timeseries")

HORIZON_IDX = 2      # 3시간
RISE_TH = 0.03
ALARM = 1.0


def featurize(X):
    """
    (N, C, 48) 시계열 → 테이블 피처.
    채널: 0=정규화수위 1=Δ1h 2=Δ3h 3=강수raw 4~7=강수누적(3/6/12/24h) 8=월
    """
    wl = X[:, 0, :]
    f = [
        wl[:, -1:], wl[:, -2:-1], wl[:, -3:-2], wl[:, -6:-5],
        wl[:, -12:-11], wl[:, -24:-23], wl[:, -48:-47],       # 수위 lag
        wl.mean(1, keepdims=True), wl.std(1, keepdims=True),
        wl.max(1, keepdims=True), wl.min(1, keepdims=True),
        (wl[:, -1:] - wl[:, -2:-1]),                          # 1h 변화
        (wl[:, -1:] - wl[:, -4:-3]),                          # 3h 변화
        (wl[:, -1:] - wl[:, -7:-6]),                          # 6h 변화
        (wl[:, -1:] - wl[:, -13:-12]),                        # 12h 변화
        (wl[:, -1:] - wl.mean(1, keepdims=True)),             # 평균 대비
        X[:, 1, -1:], X[:, 2, -1:],                           # 변화량 채널
        X[:, 1, -6:].mean(1, keepdims=True),                  # 최근 6h 평균 변화
        X[:, 3, -1:], X[:, 3, -6:].sum(1, keepdims=True),     # 강수
        X[:, 4, -1:], X[:, 5, -1:], X[:, 6, -1:], X[:, 7, -1:],
        X[:, 8, -1:],                                          # 월
    ]
    return np.hstack(f).astype(np.float32)


NAMES = ["wl_t", "wl_t1", "wl_t2", "wl_t5", "wl_t11", "wl_t23", "wl_t47",
         "wl_mean", "wl_std", "wl_max", "wl_min",
         "d1h", "d3h", "d6h", "d12h", "wl_vs_mean",
         "ch_d1", "ch_d3", "ch_d1_mean6",
         "rain_now", "rain_6h_sum", "rain_c3", "rain_c6", "rain_c12", "rain_c24",
         "month"]


def main():
    try:
        import lightgbm as lgb
    except ImportError:
        raise SystemExit("lightgbm 미설치")
    d = np.load(os.path.join(TS, "multi_dataset.npz"), allow_pickle=True)
    sets = {}
    for k in ("train", "val", "test"):
        sets[k] = (featurize(d[f"{k}_X"]), d[f"{k}_Y"][:, HORIZON_IDX], d[f"{k}_cur"])
        print(f"  {k:<6} {sets[k][0].shape}")

    Xtr, ytr, ctr = sets["train"]
    Xva, yva, cva = sets["val"]
    Xte, yte, cte = sets["test"]

    # 급상승 구간에 가중 — Transformer와 동일한 사상
    w = np.ones(len(ytr), np.float32)
    w[np.abs(ytr) >= RISE_TH] = 15.0
    w[np.abs(ytr) >= 0.10] = 40.0

    print(f"\n■ LightGBM 학습 (급상승 가중 15/40배)")
    t0 = time.time()
    ds_tr = lgb.Dataset(Xtr, ytr, weight=w, feature_name=NAMES)
    ds_va = lgb.Dataset(Xva, yva, reference=ds_tr, feature_name=NAMES)
    params = dict(objective="regression", metric="l2", learning_rate=0.05,
                  num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbosity=-1, num_threads=8)
    model = lgb.train(params, ds_tr, num_boost_round=800, valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(200)])
    print(f"  best_iter {model.best_iteration} · {time.time()-t0:.0f}s")

    pred = model.predict(Xte, num_iteration=model.best_iteration)
    p_lvl, t_lvl = cte + pred, cte + yte
    rise = yte >= RISE_TH

    print(f"\n■ TEST 회귀 성능 (처음 보는 관측소 7곳)")
    print(f"  전체 RMSE   {np.sqrt(np.mean((p_lvl-t_lvl)**2)):.4f}")
    print(f"  급상승 RMSE {np.sqrt(np.mean((p_lvl[rise]-t_lvl[rise])**2)):.4f}  (n={int(rise.sum())})")
    print(f"  과소예측    {np.mean(p_lvl[rise] < t_lvl[rise])*100:.1f}%")
    print(f"\n  [비교] 선형회귀 0.0678 · Transformer 0.0567")

    print(f"\n■ TEST 경보 판정 (3시간 뒤 경보수위 초과)")
    for th, lb in [(1.00, "경보 100%"), (0.95, "경보 95%"), (0.90, "경보 90%")]:
        pe, te_ = p_lvl >= th, t_lvl >= ALARM
        tp = int((pe & te_).sum()); fp = int((pe & ~te_).sum()); fn = int((~pe & te_).sum())
        rec = tp/(tp+fn) if tp+fn else float("nan")
        pre = tp/(tp+fp) if tp+fp else float("nan")
        print(f"  {lb:<10} 재현율 {rec*100:>5.1f}%  정밀도 {pre*100:>5.1f}%  놓침 {fn:>3}  헛경보 {fp:>4}")

    print(f"\n■ 피처 중요도 상위 12 (설명 가능성)")
    imp = sorted(zip(NAMES, model.feature_importance("gain")), key=lambda z: -z[1])
    tot = sum(v for _, v in imp) or 1
    for nm, v in imp[:12]:
        print(f"  {nm:<14} {v/tot*100:>5.1f}%  {'█'*int(v/tot*60)}")

    model.save_model(os.path.join(TS, "model_gbdt.txt"))
    print(f"\n  → data/timeseries/model_gbdt.txt")


if __name__ == "__main__":
    main()
