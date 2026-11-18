# -*- coding: utf-8 -*-
"""
카메라 단위 분할 재평가 — "처음 보는 하천에서도 되는가"

■ 왜 다시 재는가
  클립 단위로 나눴더니 val 카메라 11개가 전부 train에도 있었다.
  같은 카메라의 같은 배경을 이미 본 상태라 F2 0.884는 낙관적으로 편향된 값이다.
  수위 예측에서 관측소 단위로 나눴던 것과 같은 검증이 필요하다.

  실제 배포 상황은 "학습에 없던 하천 CCTV에 모델을 붙이는" 것이다.
  그 조건에서 얼마나 되는지가 진짜 성능이다.

■ 분할 방침
  카메라 13개뿐이라 무작위로 나누면 등급 2·3이 한쪽에 몰린다.
  경계이상 표본을 가진 카메라를 양쪽에 배분한다.

사용:
    python scripts/cv_eval_camsplit.py
"""
import sys, os, json
import numpy as np
from collections import Counter, defaultdict

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
    print(f"  {name:<24} 정확도 {acc*100:>5.1f}%  재현율 {rec*100:>5.1f}%  "
          f"정밀도 {pre*100:>5.1f}%  F2 {f2:.3f}  놓침 {fn:>3}  헛경보 {fp:>3}")
    return f2


def main():
    import lightgbm as lgb
    d = np.load(os.path.join(DS, "features.npz"))
    cams = json.load(open(os.path.join(DS, "features_cam.json"), encoding="utf-8"))
    X = np.vstack([d["Xtr"], d["Xva"]])
    y = np.concatenate([d["ytr"], d["yva"]])
    R = np.vstack([d["rtr"], d["rva"]])
    C = np.array(cams["train_cam"] + cams["val_cam"])
    print(f"■ 전체 {len(y):,}프레임 · 카메라 {len(set(C))}개")

    # 카메라별 등급 구성
    info = {}
    for c in sorted(set(C)):
        m = C == c
        cnt = Counter(y[m].tolist())
        info[c] = dict(n=int(m.sum()), hi=int((y[m] >= 2).sum()),
                       lv3=int((y[m] == 3).sum()), dist=dict(sorted(cnt.items())))
    print(f"\n■ 카메라별 구성")
    for c, v in sorted(info.items(), key=lambda z: -z[1]["hi"]):
        print(f"    {c[:30]:<32} n={v['n']:>4}  경계이상 {v['hi']:>3}  등급3 {v['lv3']:>3}  {v['dist']}")

    # 경계이상을 가진 카메라를 양쪽에 나눠 담는다.
    # 한쪽에 몰리면 "학습에 없던 위험 상황"을 평가할 수 없다.
    hi_cams = [c for c, v in info.items() if v["hi"] > 0]
    lo_cams = [c for c, v in info.items() if v["hi"] == 0]
    hi_cams.sort(key=lambda c: -info[c]["hi"])
    val_cams = set(hi_cams[1::2][:3]) | set(lo_cams[::3][:1])
    train_cams = set(info) - val_cams

    tr = np.isin(C, list(train_cams))
    va = np.isin(C, list(val_cams))
    print(f"\n■ 카메라 단위 분할")
    print(f"  train 카메라 {len(train_cams)}개 · {int(tr.sum()):,}프레임 "
          f"· 경계이상 {int((y[tr]>=2).sum())}")
    print(f"  val   카메라 {len(val_cams)}개 · {int(va.sum()):,}프레임 "
          f"· 경계이상 {int((y[va]>=2).sum())}")
    print(f"  val 카메라: {sorted(val_cams)}")
    assert not (train_cams & val_cams)

    if (y[va] >= 2).sum() < 10:
        print("  ⚠ val 경계이상 표본이 너무 적어 평가가 불안정하다")

    w = np.ones(int(tr.sum()), np.float32)
    ytr_ = y[tr]
    w[ytr_ == 1] = 3.0
    w[ytr_ >= 2] = 8.0
    P = dict(objective="multiclass", num_class=4, metric="multi_logloss",
             learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
             feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
             lambda_l2=1.0, verbosity=-1, num_threads=8)

    def fit(A, B):
        ds = lgb.Dataset(A[tr], ytr_, weight=w)
        dv = lgb.Dataset(A[va], y[va], reference=ds)
        m = lgb.train(P, ds, num_boost_round=600, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(60, verbose=False)])
        return m.predict(A[va], num_iteration=m.best_iteration).argmax(1)

    print(f"\n■ 처음 보는 카메라에서의 성능")
    report(y[va], fit(R, None), "① 강우량만")
    report(y[va], fit(X, None), "② 영상만")
    p3 = fit(np.hstack([X, R]), None)
    report(y[va], p3, "③ 영상 + 강우량")
    report(y[va], np.zeros_like(y[va]), "[기준] 항상 평상")

    print(f"\n■ 혼동 행렬 (③)")
    cm = np.zeros((4, 4), int)
    for a, b in zip(y[va], p3):
        cm[a][b] += 1
    print("        예측→   0     1     2     3")
    for i in range(4):
        print(f"    실제 {i}   " + "".join(f"{cm[i][j]:>6}" for j in range(4)))

    print(f"\n■ 앞서 클립 분할 결과와 비교")
    print(f"    클립 단위 (같은 카메라 포함)  F2 0.884  재현율 85.9%")
    print(f"    카메라 단위 (처음 보는 하천)  위 표 참조")


if __name__ == "__main__":
    main()
