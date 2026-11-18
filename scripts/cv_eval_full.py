# -*- coding: utf-8 -*-
"""
카메라 단위 재평가 (전체 데이터 · 위험 표본 9배)

앞선 결과:
    Validation만(카메라 13개) · 클립 분할   F2 0.884  ← 데이터 누수
    Validation만 · 카메라 분할              F2 0.000  ← 처음 보는 하천에서 전멸

이번에는 경계이상 표본이 371 → 3,358로 늘었다.
카메라 수(14개)는 거의 그대로이므로, 표본 증가만으로 일반화가 되는지 본다.

사용:
    python scripts/cv_eval_full.py
"""
import sys, os, json
import numpy as np
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(BASE, "data", "aihub_135", "dataset")


def report(y, p, name):
    acc = float((y == p).mean())
    yb, pb = (y >= 2), (p >= 2)
    tp = int((yb & pb).sum()); fp = int((~yb & pb).sum()); fn = int((yb & ~pb).sum())
    rec = tp / (tp + fn) if tp + fn else float("nan")
    pre = tp / (tp + fp) if tp + fp else float("nan")
    f2 = 5 * pre * rec / (4 * pre + rec) if (pre and rec) else 0.0
    print(f"  {name:<22} 정확도 {acc*100:>5.1f}%  재현율 {rec*100:>5.1f}%  "
          f"정밀도 {pre*100:>5.1f}%  F2 {f2:.3f}  놓침 {fn:>4}  헛경보 {fp:>4}")
    return f2


def main():
    import lightgbm as lgb
    d = np.load(os.path.join(DS, "features_full.npz"))
    X, y, R = d["X"], d["y"], d["R"]
    C = np.array(json.load(open(os.path.join(DS, "features_full_cam.json"),
                                encoding="utf-8")))
    print(f"■ 전체 {len(y):,}프레임 · 카메라 {len(set(C))}개")
    print(f"  등급 {dict(sorted(Counter(y.tolist()).items()))}")

    # 경계이상을 가진 카메라를 양쪽에 배분한다.
    # 한쪽에만 몰리면 '학습에 없던 위험 상황'을 평가할 수 없다.
    info = {c: int((y[C == c] >= 2).sum()) for c in sorted(set(C))}
    hi = [c for c, v in sorted(info.items(), key=lambda z: -z[1]) if v > 0]
    lo = [c for c, v in info.items() if v == 0]
    val_cams = set(hi[1::3]) | set(lo[::2][:2])
    tr = ~np.isin(C, list(val_cams))
    va = np.isin(C, list(val_cams))

    print(f"\n■ 카메라 단위 분할")
    print(f"  train 카메라 {len(set(C[tr]))}개 · {int(tr.sum()):,}프레임 "
          f"· 경계이상 {int((y[tr]>=2).sum()):,}")
    print(f"  val   카메라 {len(set(C[va]))}개 · {int(va.sum()):,}프레임 "
          f"· 경계이상 {int((y[va]>=2).sum()):,}")
    print(f"  val: {sorted(val_cams)}")

    w = np.ones(int(tr.sum()), np.float32)
    yt = y[tr]
    w[yt == 1] = 3.0
    w[yt >= 2] = 2.0        # 이제 경계이상이 충분해 과한 가중은 불필요
    P = dict(objective="multiclass", num_class=4, metric="multi_logloss",
             learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
             feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
             lambda_l2=1.0, verbosity=-1, num_threads=8)

    def fit(A):
        ds = lgb.Dataset(A[tr], yt, weight=w)
        dv = lgb.Dataset(A[va], y[va], reference=ds)
        m = lgb.train(P, ds, num_boost_round=600, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(60, verbose=False)])
        return m.predict(A[va], num_iteration=m.best_iteration).argmax(1)

    print(f"\n■ 처음 보는 하천에서의 성능")
    report(y[va], fit(R), "① 강우량만")
    report(y[va], fit(X), "② 영상만")
    p3 = fit(np.hstack([X, R]))
    report(y[va], p3, "③ 영상 + 강우량")
    report(y[va], np.zeros_like(y[va]), "[기준] 항상 평상")

    print(f"\n■ 혼동 행렬 (③)")
    cm = np.zeros((4, 4), int)
    for a, b in zip(y[va], p3):
        cm[a][b] += 1
    print("        예측→    0     1     2     3")
    for i in range(4):
        print(f"    실제 {i}   " + "".join(f"{cm[i][j]:>6}" for j in range(4)))

    print(f"\n■ 이전 결과와 비교")
    print(f"    Validation만·클립분할   F2 0.884 (누수)")
    print(f"    Validation만·카메라분할 F2 0.000")
    print(f"    전체·카메라분할         위 표")


if __name__ == "__main__":
    main()
