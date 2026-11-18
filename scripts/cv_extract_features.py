# -*- coding: utf-8 -*-
"""
CCTV 프레임 → 특징 벡터 추출 (ImageNet 사전학습 ResNet18)

■ 왜 백본을 통째로 학습하지 않는가
  GPU가 없다(AMD 내장). ResNet18을 3천 장에 fine-tune 하려면 CPU로 수십 분이 걸리고,
  하이퍼파라미터를 한 번 바꿀 때마다 그만큼 반복해야 한다.
  특징을 한 번만 뽑아두면 분류기 학습은 초 단위로 끝나 실험을 여러 번 돌릴 수 있다.
  표본이 2,519장뿐이라 백본 전체를 미세조정할 만큼 데이터가 많지도 않다.

■ 입력 전처리
  하천 CCTV는 화면 아래쪽에 수면이, 위쪽에 하늘·건물이 잡힌다.
  수위 판단에 중요한 것은 아래쪽이므로 원본 비율을 유지한 채 224로 리사이즈한다.
  (중앙 크롭을 하면 수면이 잘려나갈 수 있다)

사용:
    python scripts/cv_extract_features.py
"""
import sys, os, json, time
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(BASE, "data", "aihub_135")
DS = os.path.join(ROOT, "dataset")
IMGROOT = os.path.join(ROOT, "val_image", "침수위험 지역 라벨링 데이터")


def build_index():
    """파일명 → 실제 경로. 하위 폴더가 깊어 매번 탐색하면 느리다."""
    idx = {}
    for dirpath, _, files in os.walk(IMGROOT):
        for f in files:
            if f.lower().endswith(".jpg"):
                idx[f] = os.path.join(dirpath, f)
    return idx


def main():
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18, ResNet18_Weights
    from PIL import Image

    torch.set_num_threads(8)
    split = json.load(open(os.path.join(DS, "split.json"), encoding="utf-8"))
    print(f"■ train {len(split['train']):,} · val {len(split['val']):,}")

    print("  이미지 경로 색인 중...")
    t0 = time.time()
    path_of = build_index()
    print(f"    {len(path_of):,}장 ({time.time()-t0:.0f}s)")

    weights = ResNet18_Weights.IMAGENET1K_V1
    net = resnet18(weights=weights)
    net.fc = nn.Identity()          # 512차원 특징만 뽑는다
    net.eval()
    tf = weights.transforms()

    def run(rows, tag):
        feats, keep = [], []
        t0 = time.time()
        batch, meta = [], []
        for i, r in enumerate(rows):
            p = path_of.get(r["file"] if r["file"].endswith(".jpg") else r["file"] + ".jpg")
            if not p:
                continue
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                continue
            batch.append(tf(im))
            meta.append(r)
            if len(batch) == 32:
                with torch.no_grad():
                    feats.append(net(torch.stack(batch)).numpy())
                keep += meta
                batch, meta = [], []
                if len(keep) % 512 == 0:
                    print(f"    {tag} {len(keep):,}/{len(rows):,} "
                          f"({time.time()-t0:.0f}s)", flush=True)
        if batch:
            with torch.no_grad():
                feats.append(net(torch.stack(batch)).numpy())
            keep += meta
        X = np.concatenate(feats) if feats else np.zeros((0, 512), np.float32)
        y = np.array([r["level"] for r in keep], np.int64)
        rain = np.array([[r["rain_1h"], r["rain_3h"]] for r in keep], np.float32)
        cam = [r["cam"] for r in keep]
        print(f"    {tag} 완료 {X.shape} ({time.time()-t0:.0f}s)")
        return X, y, rain, cam

    Xtr, ytr, rtr, ctr = run(split["train"], "train")
    Xva, yva, rva, cva = run(split["val"], "val")

    np.savez_compressed(os.path.join(DS, "features.npz"),
                        Xtr=Xtr, ytr=ytr, rtr=rtr,
                        Xva=Xva, yva=yva, rva=rva)
    json.dump(dict(train_cam=ctr, val_cam=cva),
              open(os.path.join(DS, "features_cam.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"\n  → {os.path.join(DS, 'features.npz')}")
    from collections import Counter
    print(f"  train 등급 {dict(sorted(Counter(ytr.tolist()).items()))}")
    print(f"  val   등급 {dict(sorted(Counter(yva.tolist()).items()))}")


if __name__ == "__main__":
    main()
