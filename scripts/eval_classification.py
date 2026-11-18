# -*- coding: utf-8 -*-
"""
회귀 예측을 '경보 판정'으로 바꿔 평가한다.

■ 왜 분류로 보는가
  대피 판단에 실제로 쓰이는 정보는 "3시간 뒤 수위가 9.4m"가 아니라
  "3시간 안에 위험수위를 넘는가"다. RMSE는 이 물음에 직접 답하지 못한다.

  분류로 보면 재난 시스템에서 가장 중요한 수치가 바로 나온다.
      재현율(Recall)  = 실제 위험 중 몇 %를 잡아냈는가   ← 놓치면 사람이 죽는다
      정밀도(Precision) = 경보 중 몇 %가 진짜였는가       ← 낮으면 양치기 소년

  두 지표의 비용이 다르므로 F1보다 F2(재현율 2배 가중)를 함께 본다.

사용:
    python scripts/eval_classification.py
"""
import sys, os, json
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = os.path.join(BASE, "data", "timeseries")
sys.path.insert(0, os.path.join(BASE, "scripts"))

# 수위는 경보수위로 정규화되어 있으므로 1.0 = 경보수위
ALARM = 1.0
HORIZON_IDX = 2          # HORIZONS = [1,2,3,6] → 3시간


def metrics(pred_exceed, true_exceed):
    tp = int((pred_exceed & true_exceed).sum())
    fp = int((pred_exceed & ~true_exceed).sum())
    fn = int((~pred_exceed & true_exceed).sum())
    tn = int((~pred_exceed & ~true_exceed).sum())
    rec = tp / (tp + fn) if tp + fn else float("nan")
    pre = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * pre * rec / (pre + rec) if pre + rec else float("nan")
    # F2: 재난에서는 놓치는 것(재현율)이 헛경보보다 훨씬 비싸다
    f2 = 5 * pre * rec / (4 * pre + rec) if (4 * pre + rec) else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, rec=rec, pre=pre, f1=f1, f2=f2)


def report(name, pred_lvl, true_lvl, th=ALARM):
    m = metrics(pred_lvl >= th, true_lvl >= th)
    print(f"  {name:<24} 재현율 {m['rec']*100:>5.1f}%  정밀도 {m['pre']*100:>5.1f}%  "
          f"F1 {m['f1']:.3f}  F2 {m['f2']:.3f}   놓침 {m['fn']:>4}  헛경보 {m['fp']:>5}")
    return m


def main():
    d = np.load(os.path.join(TS, "multi_dataset.npz"), allow_pickle=True)
    Xtr, Ytr, Ctr = d["train_X"], d["train_Y"], d["train_cur"]
    Xte, Yte, Cte = d["test_X"], d["test_Y"], d["test_cur"]
    true_lvl = Cte + Yte[:, HORIZON_IDX]
    n_pos = int((true_lvl >= ALARM).sum())
    print(f"■ TEST {len(Xte):,}표본 · 실제 경보수위 초과 {n_pos:,}건 "
          f"({n_pos/len(Xte)*100:.2f}%)\n")
    if n_pos < 10:
        print("  ⚠ 양성 표본이 너무 적어 분류 평가가 불안정하다")

    print("■ 3시간 뒤 경보수위 초과 판정")
    # ① Persistence
    report("① Persistence", Cte, true_lvl)

    # ② 선형회귀
    def feat(X):
        return np.hstack([X[:, 0, -12:], X[:, 1, -1:], X[:, 2, -1:],
                          X[:, 0, :].mean(1, keepdims=True), X[:, 3:8, -1]]).astype(np.float64)
    def ridge(X, y, lam=1.0):
        Xb = np.hstack([X, np.ones((len(X), 1))])
        A = Xb.T @ Xb + lam * np.eye(Xb.shape[1]); A[-1, -1] -= lam
        return np.linalg.solve(A, Xb.T @ y)
    Ftr, Fte = feat(Xtr), feat(Xte)
    w = ridge(Ftr, Ytr[:, HORIZON_IDX])
    lin = Cte + (np.hstack([Fte, np.ones((len(Fte), 1))]) @ w)
    report("② 선형회귀", lin, true_lvl)

    # ③ Transformer
    p = os.path.join(TS, "model_multi.pt")
    if os.path.exists(p):
        import torch
        from forecast_model import build_model
        ck = torch.load(p, weights_only=False)
        model = build_model(Xtr.shape[1], Xtr.shape[2], Ytr.shape[1], n_wl_ch=3,
                            d_model=64, n_layer=2, dropout=0.2, use_revin=False)
        model.load_state_dict(ck["state"]); model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(Xte), 4096):
                o, _, _ = model(torch.tensor(Xte[i:i + 4096]))
                outs.append(o.numpy())
        tr_lvl = Cte + np.concatenate(outs)[:, HORIZON_IDX]
        report("③ Transformer", tr_lvl, true_lvl)

        print(f"\n■ 경보 문턱을 낮추면 (조기경보 관점)")
        print(f"  실제 운영에서는 경보수위 도달 '전에' 알려야 하므로 문턱을 낮춰 잡는다.")
        for th, lb in [(0.95, "경보의 95%"), (0.90, "경보의 90%"), (0.80, "경보의 80%")]:
            m = metrics(tr_lvl >= th, true_lvl >= ALARM)
            print(f"    {lb:<12} 재현율 {m['rec']*100:>5.1f}%  정밀도 {m['pre']*100:>5.1f}%  "
                  f"놓침 {m['fn']:>4}  헛경보 {m['fp']:>5}")
    else:
        print("  ③ Transformer: model_multi.pt 없음")


if __name__ == "__main__":
    main()
