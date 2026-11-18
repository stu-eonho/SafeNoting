# -*- coding: utf-8 -*-
"""
새 test 분할에서의 baseline 재측정

■ 왜 다시 재는가
  관측소가 47개 → 55개로 늘면서 train/val/test 분할이 통째로 바뀌었다.
  test 관측소가 달라졌으므로 예전 baseline(급상승 0.0678)과 직접 비교할 수 없다.
  같은 test에서 baseline을 다시 재야 "데이터를 2배로 늘린 효과"를 정직하게 말할 수 있다.

  ① Persistence  — 지금 수위가 그대로 유지된다고 본다 (변화량 0 예측)
  ② 선형회귀     — 최근 48시간 전 채널을 펼쳐 Ridge

사용:
    python scripts/baseline_multi.py
"""
import sys, os, json
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = os.path.join(BASE, "data", "timeseries")
RISE_TH = 0.03          # 경보수위 대비 3% 이상 상승 = 급상승
H = 2                   # 3시간 후 (HORIZONS = [1,2,3,6] 중 index 2)


def rep(name, yt, yp, rise):
    all_r = float(np.sqrt(((yp - yt) ** 2).mean()))
    r_r = float(np.sqrt(((yp[rise] - yt[rise]) ** 2).mean()))
    under = float((yp[rise] < yt[rise]).mean() * 100)
    print(f"  {name:<22}{all_r:>10.4f}{r_r:>12.4f}{under:>11.1f}%")
    return all_r, r_r, under


def main():
    d = np.load(os.path.join(TS, "multi_dataset.npz"), allow_pickle=True)
    Xtr, Ytr = d["train_X"], d["train_Y"][:, H]
    Xte, Yte = d["test_X"], d["test_Y"][:, H]
    rise = Yte >= RISE_TH
    print(f"■ TEST 표본 {len(Yte):,} · 급상승 {int(rise.sum()):,}")
    print(f"\n  {'모델':<22}{'전체 RMSE':>10}{'급상승 RMSE':>12}{'과소예측':>12}")

    rep("① Persistence", Yte, np.zeros_like(Yte), rise)

    # ② Ridge — 채널 x 시간을 그대로 펼친다
    n_tr = min(len(Xtr), 400_000)          # 메모리 보호. 40만이면 충분히 수렴한다
    idx = np.random.RandomState(0).choice(len(Xtr), n_tr, replace=False)
    A = Xtr[idx].reshape(n_tr, -1).astype(np.float64)
    b = Ytr[idx].astype(np.float64)
    A = np.hstack([A, np.ones((n_tr, 1))])
    lam = 1.0
    G = A.T @ A + lam * np.eye(A.shape[1])
    w = np.linalg.solve(G, A.T @ b)
    B = np.hstack([Xte.reshape(len(Xte), -1).astype(np.float64),
                   np.ones((len(Xte), 1))])
    rep("② 선형회귀(Ridge)", Yte, (B @ w).astype(np.float32), rise)

    print(f"\n  ※ 단위는 경보수위 대비 비율. 이전 test(관측소 7곳)에서는")
    print(f"     Persistence 0.0947 · Ridge 0.0678 이었다 (다른 하천이므로 참고용).")


if __name__ == "__main__":
    main()
