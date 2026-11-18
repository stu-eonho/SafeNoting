# -*- coding: utf-8 -*-
"""
zip에서 직접 읽어 특징 추출 — 33.8GB를 디스크에 풀지 않는다

■ 왜 압축을 풀지 않는가
  Training 원천데이터는 33.8GB지만 실제로 쓸 프레임은 샘플링 후 수천 장뿐이다.
  zipfile.open()으로 필요한 항목만 스트림으로 읽으면
  디스크 사용 0, 해제 대기 0으로 끝난다.

사용:
    python scripts/cv_extract_full.py
"""
import sys, os, json, time, zipfile, io
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(BASE, "data", "aihub_135")
DS = os.path.join(ROOT, "dataset")
SRC = os.path.join(BASE, "data", "cctv_snapshots",
                   "135.부산시 침수위험 복합 데이터", "3.개방데이터", "1.데이터")
ZIPS = [os.path.join(SRC, "Training", "01.원천데이터", "TS.zip"),
        os.path.join(SRC, "Validation", "01.원천데이터", "VS.zip")]


def main():
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18, ResNet18_Weights
    from PIL import Image

    torch.set_num_threads(8)
    rows = json.load(open(os.path.join(DS, "full_index.json"), encoding="utf-8"))
    want = {r["file"] if r["file"].endswith(".jpg") else r["file"] + ".jpg": r
            for r in rows}
    print(f"■ 대상 {len(want):,}프레임")

    # zip 내 경로 색인 — 파일명만으로는 어느 zip에 있는지 모른다
    loc = {}
    for zp in ZIPS:
        if not os.path.exists(zp):
            print(f"  ! 없음: {zp}")
            continue
        t0 = time.time()
        with zipfile.ZipFile(zp) as z:
            n = 0
            for name in z.namelist():
                if not name.lower().endswith(".jpg"):
                    continue
                base = name.rsplit("/", 1)[-1]
                if base in want and base not in loc:
                    loc[base] = (zp, name)
                    n += 1
        print(f"  {os.path.basename(zp)}: {n:,}개 매칭 ({time.time()-t0:.0f}s)")
    print(f"  총 매칭 {len(loc):,}/{len(want):,}")

    weights = ResNet18_Weights.IMAGENET1K_V1
    net = resnet18(weights=weights)
    net.fc = nn.Identity()
    net.eval()
    tf = weights.transforms()

    # zip을 반복해서 여닫으면 느리다. zip별로 묶어 한 번씩만 연다.
    by_zip = {}
    for base, (zp, name) in loc.items():
        by_zip.setdefault(zp, []).append((base, name))

    feats, keep = [], []
    t0 = time.time()
    for zp, items in by_zip.items():
        with zipfile.ZipFile(zp) as z:
            batch, meta = [], []
            for base, name in items:
                try:
                    with z.open(name) as fp:
                        im = Image.open(io.BytesIO(fp.read())).convert("RGB")
                except Exception:
                    continue
                batch.append(tf(im))
                meta.append(want[base])
                if len(batch) == 32:
                    with torch.no_grad():
                        feats.append(net(torch.stack(batch)).numpy())
                    keep += meta
                    batch, meta = [], []
                    if len(keep) % 1024 == 0:
                        print(f"    {len(keep):,}/{len(loc):,} "
                              f"({time.time()-t0:.0f}s)", flush=True)
            if batch:
                with torch.no_grad():
                    feats.append(net(torch.stack(batch)).numpy())
                keep += meta

    X = np.concatenate(feats) if feats else np.zeros((0, 512), np.float32)
    y = np.array([r["level"] for r in keep], np.int64)
    R = np.array([[r["rain_1h"], r["rain_3h"]] for r in keep], np.float32)
    cam = [r["cam"] for r in keep]
    print(f"\n■ 특징 {X.shape} ({time.time()-t0:.0f}s)")
    from collections import Counter
    print(f"  등급 {dict(sorted(Counter(y.tolist()).items()))}")
    print(f"  카메라 {len(set(cam))}개")

    np.savez_compressed(os.path.join(DS, "features_full.npz"), X=X, y=y, R=R)
    json.dump(cam, open(os.path.join(DS, "features_full_cam.json"), "w",
                        encoding="utf-8"), ensure_ascii=False)
    print(f"  → {os.path.join(DS, 'features_full.npz')}")


if __name__ == "__main__":
    main()
