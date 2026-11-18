# -*- coding: utf-8 -*-
"""
이진 분류 재평가 — 평상(0,1) vs 경계이상(2,3)

■ 왜 4단계를 버리는가
  등급 3이 439건(val 150건)뿐이라 4단계로는 학습이 되지 않았다.
  실제로 등급 3의 98%를 '평상'으로 오판했다.
  그리고 대피 판단에 필요한 것은 "얼마나 위험한가"가 아니라 "위험한가"다.
  두 등급을 합치면 양성 표본이 3,310건으로 늘고 목적에도 맞는다.

■ 강우량 단독 모델의 F2 0.891은 착시였다
  헛경보 606 + 정답 991 = val 전체. 전부 '경계이상'이라 답한 결과다.
  이진에서도 같은 함정이 생기므로, 예측이 한쪽으로 쏠렸는지 함께 확인한다.

사용:
    python scripts/cv_eval_binary.py
"""
import sys, os, json
import numpy as np
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(BASE, "data", "aihub_135", "dataset")


def report(y, prob, name, th=0.5):
    p = (prob >= th).astype(int)
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    acc = (tp + tn) / len(y)
    rec = tp / (tp + fn) if tp + fn else float("nan")
    pre = tp / (tp + fp) if tp + fp else float("nan")
    f2 = 5 * pre * rec / (4 * pre + rec) if (pre and rec) else 0.0
    ratio = p.mean() * 100
    flag = "  ← 쏠림" if ratio > 95 or ratio < 5 else ""
    print(f"  {name:<22} 정확도 {acc*100:>5.1f}%  재현율 {rec*100:>5.1f}%  "
          f"정밀도 {pre*100:>5.1f}%  F2 {f2:.3f}  놓침 {fn:>4}  헛경보 {fp:>4}  "
          f"경보율 {ratio:>4.0f}%{flag}")
    return dict(rec=rec, pre=pre, f2=f2, fn=fn, fp=fp)


def main():
    import lightgbm as lgb
    d = np.load(os.path.join(DS, "features_full.npz"))
    X, y4, R = d["X"], d["y"], d["R"]
    C = np.array(json.load(open(os.path.join(DS, "features_full_cam.json"),
                                encoding="utf-8")))
    y = (y4 >= 2).astype(int)          # 이진화
    print(f"■ 전체 {len(y):,}프레임 · 카메라 {len(set(C))}개")
    print(f"  평상 {int((y==0).sum()):,} · 경계이상 {int((y==1).sum()):,}")

    info = {c: int(y[C == c].sum()) for c in sorted(set(C))}
    hi = [c for c, v in sorted(info.items(), key=lambda z: -z[1]) if v > 0]
    lo = [c for c, v in info.items() if v == 0]
    val_cams = set(hi[1::3]) | set(lo[::2][:2])
    tr = ~np.isin(C, list(val_cams))
    va = np.isin(C, list(val_cams))
    print(f"\n■ 카메라 단위 분할 (val 카메라는 학습에서 한 번도 보지 않는다)")
    print(f"  train {len(set(C[tr]))}개 카메라 · {int(tr.sum()):,}프레임 "
          f"· 양성 {int(y[tr].sum()):,} ({y[tr].mean()*100:.0f}%)")
    print(f"  val   {len(set(C[va]))}개 카메라 · {int(va.sum()):,}프레임 "
          f"· 양성 {int(y[va].sum()):,} ({y[va].mean()*100:.0f}%)")

    P = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
             num_leaves=31, min_data_in_leaf=20, feature_fraction=0.7,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
             verbosity=-1, num_threads=8)

    def fit(A):
        ds = lgb.Dataset(A[tr], y[tr])
        dv = lgb.Dataset(A[va], y[va], reference=ds)
        m = lgb.train(P, ds, num_boost_round=600, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(60, verbose=False)])
        return m, m.predict(A[va], num_iteration=m.best_iteration)

    print(f"\n■ 처음 보는 하천에서의 성능 (문턱 0.5)")
    _, pr_rain = fit(R); report(y[va], pr_rain, "① 강우량만")
    _, pr_img = fit(X); report(y[va], pr_img, "② 영상만")
    m3, pr_both = fit(np.hstack([X, R])); r3 = report(y[va], pr_both, "③ 영상 + 강우량")
    report(y[va], np.zeros(int(va.sum())), "[기준] 항상 평상")
    report(y[va], np.ones(int(va.sum())), "[기준] 항상 경보")

    print(f"\n■ 문턱 조정 (③ 영상+강우량) — 재난에서는 놓침이 헛경보보다 비싸다")
    for th in (0.7, 0.5, 0.35, 0.25, 0.15):
        report(y[va], pr_both, f"  문턱 {th:.2f}", th)

    imp = m3.feature_importance("gain")
    n = X.shape[1]
    print(f"\n■ 피처 기여도  영상 {imp[:n].sum()/imp.sum()*100:.1f}% · "
          f"강우량 {imp[n:].sum()/imp.sum()*100:.1f}%")

    print(f"\n■ 경과")
    print(f"    Validation만·클립분할    F2 0.884  (데이터 누수)")
    print(f"    Validation만·카메라분할  F2 0.000  (전멸)")
    print(f"    전체·카메라분할·4단계    F2 0.775")
    print(f"    전체·카메라분할·이진     F2 {r3['f2']:.3f}")

    json.dump({k: float(v) for k, v in r3.items()},
              open(os.path.join(DS, "binary_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
