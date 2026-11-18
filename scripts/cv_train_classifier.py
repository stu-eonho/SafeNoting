# -*- coding: utf-8 -*-
"""
수위 등급 분류기 — 영상 특징 + 강우량

■ 무엇을 맞히는가
  CCTV 한 장면을 보고 그 지점의 수위가 평상시 대비 어느 단계인지 판정한다.
      0 평상 (<1.3배)  1 증가 (1.3~2배)  2 경계 (2~3배)  3 위험 (3배↑)

■ 평가 기준을 정확도로 삼으면 안 된다
  등급 0이 전체의 86%다. "항상 평상"이라고 답해도 정확도 86%가 나온다.
  수위 예측에서 겪은 것과 같은 함정이다.
  → **경계이상(2·3등급) 재현율**을 기준으로 본다. 놓치면 사람이 위험해진다.

■ 영상만 vs 강우량 결합
  강우량은 이미 확보한 데이터이므로 함께 넣을 수 있다.
  다만 강우량만으로도 어느 정도 맞는다면 "영상이 실제로 기여하는가"를
  따로 확인해야 한다. 세 가지를 나란히 비교한다.

사용:
    python scripts/cv_train_classifier.py
"""
import sys, os, json
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(BASE, "data", "aihub_135", "dataset")


def metrics(y, p, name):
    """경계이상(>=2)을 양성으로 보는 이진 지표 + 4단계 정확도"""
    acc = float((y == p).mean())
    yb, pb = (y >= 2), (p >= 2)
    tp = int((yb & pb).sum()); fp = int((~yb & pb).sum())
    fn = int((yb & ~pb).sum())
    rec = tp / (tp + fn) if tp + fn else float("nan")
    pre = tp / (tp + fp) if tp + fp else float("nan")
    f2 = 5 * pre * rec / (4 * pre + rec) if (pre + rec) and (4 * pre + rec) else float("nan")
    print(f"  {name:<26} 정확도 {acc*100:>5.1f}%   경계이상 재현율 {rec*100:>5.1f}%  "
          f"정밀도 {pre*100:>5.1f}%  F2 {f2:.3f}   놓침 {fn:>3}  헛경보 {fp:>3}")
    return dict(acc=acc, rec=rec, pre=pre, f2=f2, fn=fn, fp=fp)


def main():
    import lightgbm as lgb
    p = os.path.join(DS, "features.npz")
    if not os.path.exists(p):
        raise SystemExit("features.npz 없음 — cv_extract_features.py 먼저 실행")
    d = np.load(p)
    Xtr, ytr, rtr = d["Xtr"], d["ytr"], d["rtr"]
    Xva, yva, rva = d["Xva"], d["yva"], d["rva"]
    print(f"■ train {Xtr.shape} · val {Xva.shape}")
    from collections import Counter
    print(f"  train 등급 {dict(sorted(Counter(ytr.tolist()).items()))}")
    print(f"  val   등급 {dict(sorted(Counter(yva.tolist()).items()))}")

    # 등급 0이 86%라 그대로 학습하면 전부 0으로 답한다.
    # 경계이상 표본에 가중을 준다.
    w = np.ones(len(ytr), np.float32)
    w[ytr == 1] = 3.0
    w[ytr >= 2] = 8.0

    P = dict(objective="multiclass", num_class=4, metric="multi_logloss",
             learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
             feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
             lambda_l2=1.0, verbosity=-1, num_threads=8)

    def fit(Xa, Xb, tag):
        ds = lgb.Dataset(Xa, ytr, weight=w)
        dv = lgb.Dataset(Xb, yva, reference=ds)
        m = lgb.train(P, ds, num_boost_round=600, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(60, verbose=False)])
        pred = m.predict(Xb, num_iteration=m.best_iteration).argmax(1)
        return m, pred

    print(f"\n■ 검증 결과 (val {len(yva):,}프레임 · 경계이상 {int((yva>=2).sum())}건)")

    # ① 강우량만 — 영상 없이 얼마나 되는가
    _, p_rain = fit(rtr, rva, "rain")
    metrics(yva, p_rain, "① 강우량만 (2피처)")

    # ② 영상 특징만
    _, p_img = fit(Xtr, Xva, "img")
    metrics(yva, p_img, "② 영상만 (ResNet18 512)")

    # ③ 결합
    m3, p_both = fit(np.hstack([Xtr, rtr]), np.hstack([Xva, rva]), "both")
    r3 = metrics(yva, p_both, "③ 영상 + 강우량")

    # 항상 0으로 답하는 기준선 — 정확도가 얼마나 무의미한지 보여준다
    metrics(yva, np.zeros_like(yva), "[기준] 항상 '평상'")

    print(f"\n■ 혼동 행렬 (③ 영상+강우량)")
    cm = np.zeros((4, 4), int)
    for a, b in zip(yva, p_both):
        cm[a][b] += 1
    print("        예측→   0     1     2     3")
    for i in range(4):
        print(f"    실제 {i}   " + "".join(f"{cm[i][j]:>6}" for j in range(4)))

    imp = m3.feature_importance("gain")
    n_img = Xtr.shape[1]
    print(f"\n■ 피처 기여도")
    print(f"    영상 특징 {imp[:n_img].sum()/imp.sum()*100:>5.1f}%")
    print(f"    강우량    {imp[n_img:].sum()/imp.sum()*100:>5.1f}%")

    json.dump(dict(rain=None, img=None, both=r3),
              open(os.path.join(DS, "cls_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)
    print(f"\n  → {os.path.join(DS, 'cls_result.json')}")


if __name__ == "__main__":
    main()
