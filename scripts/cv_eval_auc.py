# -*- coding: utf-8 -*-
"""
AUC 기반 재평가 — F2가 속인 것을 바로잡는다

■ F2를 버리는 이유
  val 양성 비율이 62%라 '항상 경보'가 F2 0.891로 1등이 됐다.
  아무 판단도 하지 않는 전략이 최고 점수를 받는 지표는 쓸 수 없다.
  실제 분포에서 경계이상은 10%뿐인데 샘플링이 비율을 왜곡했다.

  ROC-AUC는 '항상 경보'가 정의상 0.5다. 판별력을 그대로 드러낸다.
  PR-AUC는 양성 비율(기저선)과 비교해 읽는다.

■ 실제 분포 환산
  운영 환경(양성 10%)에서 이 문턱을 쓰면 헛경보가 몇 건 나오는지 함께 계산한다.
  62% 환경의 헛경보 수치는 현장 감각과 맞지 않는다.

사용:
    python scripts/cv_eval_auc.py
"""
import sys, os, json
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(BASE, "data", "aihub_135", "dataset")
REAL_POS = 0.101          # 원본 분포의 경계이상 비율 (3,358/33,291)


def roc_auc(y, s):
    """순위 기반. sklearn 없이 계산한다."""
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # 동점 처리
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pr_auc(y, s):
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    rec = tp / max(y.sum(), 1)
    pre = tp / np.maximum(tp + fp, 1)
    return float(np.sum(np.diff(np.concatenate([[0], rec])) * pre))


def main():
    import lightgbm as lgb
    d = np.load(os.path.join(DS, "features_full.npz"))
    X, y4, R = d["X"], d["y"], d["R"]
    C = np.array(json.load(open(os.path.join(DS, "features_full_cam.json"),
                                encoding="utf-8")))
    y = (y4 >= 2).astype(int)

    info = {c: int(y[C == c].sum()) for c in sorted(set(C))}
    hi = [c for c, v in sorted(info.items(), key=lambda z: -z[1]) if v > 0]
    lo = [c for c, v in info.items() if v == 0]
    val_cams = set(hi[1::3]) | set(lo[::2][:2])
    tr = ~np.isin(C, list(val_cams))
    va = np.isin(C, list(val_cams))
    yv = y[va]
    print(f"■ val {int(va.sum()):,}프레임 · 양성 {int(yv.sum()):,} ({yv.mean()*100:.0f}%)")
    print(f"  ※ 실제 운영 분포의 양성 비율은 {REAL_POS*100:.1f}% — 아래 환산치 참고\n")

    P = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
             num_leaves=31, min_data_in_leaf=20, feature_fraction=0.7,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
             verbosity=-1, num_threads=8)

    def fit(A):
        ds = lgb.Dataset(A[tr], y[tr])
        dv = lgb.Dataset(A[va], yv, reference=ds)
        m = lgb.train(P, ds, num_boost_round=600, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(60, verbose=False)])
        return m.predict(A[va], num_iteration=m.best_iteration)

    print("■ 판별력 (클래스 비율에 흔들리지 않는 지표)")
    print(f"  {'모델':<20}{'ROC-AUC':>9}{'PR-AUC':>9}   해석")
    base_pr = yv.mean()
    results = {}
    for nm, A in (("① 강우량만", R), ("② 영상만", X),
                  ("③ 영상+강우량", np.hstack([X, R]))):
        s = fit(A)
        ra, pa = roc_auc(yv, s), pr_auc(yv, s)
        results[nm] = s
        judge = "무작위와 같음" if ra < 0.55 else ("약함" if ra < 0.7 else
                ("쓸만함" if ra < 0.85 else "우수"))
        print(f"  {nm:<20}{ra:>9.3f}{pa:>9.3f}   {judge}")
    print(f"  {'[기준] 항상 경보':<20}{0.5:>9.3f}{base_pr:>9.3f}   판별력 없음")
    print(f"  {'[기준] 무작위':<20}{0.5:>9.3f}{base_pr:>9.3f}")

    s = results["③ 영상+강우량"]
    print(f"\n■ 문턱별 성능 + 실제 분포(양성 {REAL_POS*100:.0f}%) 환산")
    print(f"  {'문턱':>6}{'재현율':>8}{'정밀도':>8}{'놓침':>7}{'헛경보':>7}"
          f"   실제분포 정밀도  1000건당 헛경보")
    for th in (0.7, 0.5, 0.35, 0.25, 0.15):
        p = (s >= th).astype(int)
        tp = int(((yv == 1) & (p == 1)).sum()); fp = int(((yv == 0) & (p == 1)).sum())
        fn = int(((yv == 1) & (p == 0)).sum()); tn = int(((yv == 0) & (p == 0)).sum())
        rec = tp / max(tp + fn, 1)
        pre = tp / max(tp + fp, 1)
        fpr = fp / max(fp + tn, 1)
        # 실제 분포에서의 정밀도 = 재현율*p / (재현율*p + fpr*(1-p))
        real_pre = rec * REAL_POS / max(rec * REAL_POS + fpr * (1 - REAL_POS), 1e-9)
        per1000 = fpr * (1 - REAL_POS) * 1000
        print(f"  {th:>6.2f}{rec*100:>7.1f}%{pre*100:>7.1f}%{fn:>7}{fp:>7}"
              f"{real_pre*100:>14.1f}%{per1000:>16.0f}건")

    print(f"\n  ※ 실제 분포에서는 양성이 드물어 같은 문턱이라도 헛경보 비중이 커진다.")
    print(f"    운영 시에는 이 표의 오른쪽 두 열을 기준으로 문턱을 정해야 한다.")


if __name__ == "__main__":
    main()
