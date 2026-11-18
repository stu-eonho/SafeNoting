# -*- coding: utf-8 -*-
"""
Training 라벨 → 전체 데이터셋 재구성 (카메라 30개)

■ 왜 다시 만드는가
  Validation만(카메라 13개) 쓴 결과, 카메라 단위로 나누자 재현율이 0%로 무너졌다.
  모델이 물이 아니라 '이 카메라의 배경'을 외운 탓이다.
  카메라 수를 30개로 늘려 배경 암기가 통하지 않게 만든다.

■ 33.8GB를 전부 풀지 않는 이유
  80,000프레임의 87%가 등급 0(평상)이다. 전부 특징 추출하면 70분이 걸리는데
  대부분이 같은 장면의 반복이다.
    · 등급 2·3(경계·위험)은 희소하므로 전부 사용
    · 등급 0·1은 클립당 소수만 샘플링
  필요한 이미지만 zip에서 직접 뽑아내면 디스크도 아낀다.

사용:
    python scripts/cv_build_full.py
"""
import sys, os, json, re, random, time, zipfile
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(BASE, "data", "aihub_135")
DS = os.path.join(ROOT, "dataset")
TRAIN_LAB = os.path.join(ROOT, "train_label", "침수위험 지역 라벨링 데이터")
VAL_LAB = os.path.join(ROOT, "val_label", "침수위험 지역 라벨링 데이터")

FRAME_PX = 1280 * 720
MIN_BASE_RATIO = 0.01
PER_CLIP_LOW = 3          # 등급 0·1은 클립당 이 개수만


def parse_clip(fn):
    m = re.match(r"([A-Za-z]+)_+(\d+)_(\d+)\^(\d{8})-(\d{6})", fn)
    if not m:
        return None
    return f"{m.group(3)}_{m.group(4)}_{m.group(5)}"


def scan(labroot, src_tag):
    rows = []
    for dirpath, _, files in os.walk(labroot):
        for f in files:
            if not f.endswith(".json"):
                continue
            try:
                d = json.load(open(os.path.join(dirpath, f), encoding="utf-8"))
            except Exception:
                continue
            info = d.get("info") or {}
            im = (d.get("images") or [{}])[0]
            rel = os.path.relpath(dirpath, labroot)
            cam = rel.split(os.sep)[1] if os.sep in rel else rel
            area = defaultdict(float)
            names = {c["id"]: c["name"] for c in d.get("categories", [])}
            for a in d.get("annotations") or []:
                area[names.get(a.get("category_id"), "?")] += float(a.get("area") or 0)
            rows.append(dict(
                file=im.get("file_name") or f.replace(".json", ".jpg"),
                cam=cam, clip=parse_clip(f), src=src_tag,
                rain_1h=float(info.get("1hr_rainfall") or 0),
                rain_3h=float(info.get("3hr_rainfall") or 0),
                water=area.get("water_area", 0.0)))
    return rows


def main():
    if not os.path.isdir(TRAIN_LAB):
        raise SystemExit(f"라벨 폴더 없음: {TRAIN_LAB}")
    t0 = time.time()
    rows = scan(TRAIN_LAB, "train") + scan(VAL_LAB, "val")
    print(f"■ 라벨 {len(rows):,}건 ({time.time()-t0:.0f}s)")

    by_cam = defaultdict(list)
    for r in rows:
        by_cam[r["cam"]].append(r)

    # 카메라별 기준면적 (0 제외 중앙값) — Validation 때 확립한 방식
    kept = {}
    for cam, rs in by_cam.items():
        pos = sorted(x["water"] for x in rs if x["water"] > 0)
        if not pos:
            continue
        base = pos[len(pos) // 2]
        if base < FRAME_PX * MIN_BASE_RATIO:
            continue
        kept[cam] = base
        for r in rs:
            r["water_rel"] = r["water"] / base
            rel = r["water_rel"]
            r["level"] = 0 if rel < 1.3 else (1 if rel < 2.0 else (2 if rel < 3.0 else 3))
    rows = [r for r in rows if r["cam"] in kept]
    print(f"■ 사용 카메라 {len(kept)}개 / 전체 {len(by_cam)}개 · 프레임 {len(rows):,}")

    gc = Counter(r["level"] for r in rows)
    print(f"  등급 분포 {dict(sorted(gc.items()))}")

    # ── 샘플링 ──────────────────────────────────────────────
    # 등급 2·3은 전부, 0·1은 클립당 소수만. 같은 클립은 거의 같은 장면이다.
    by_clip = defaultdict(list)
    for r in rows:
        by_clip[(r["cam"], r["clip"])].append(r)
    picked = []
    rng = random.Random(42)
    for key, rs in by_clip.items():
        hi = [r for r in rs if r["level"] >= 2]
        lo = [r for r in rs if r["level"] < 2]
        picked += hi
        rng.shuffle(lo)
        picked += lo[:PER_CLIP_LOW]
    print(f"\n■ 샘플링 후 {len(picked):,}프레임 (클립 {len(by_clip):,}개)")
    print(f"  등급 분포 {dict(sorted(Counter(r['level'] for r in picked).items()))}")

    print(f"\n■ 카메라별 (경계이상 보유 순)")
    cam_stat = defaultdict(Counter)
    for r in picked:
        cam_stat[r["cam"]][r["level"]] += 1
    ranked = sorted(cam_stat.items(), key=lambda z: -(z[1][2] + z[1][3]))
    for cam, c in ranked:
        hi = c[2] + c[3]
        print(f"    {cam[:34]:<36} n={sum(c.values()):>5}  경계이상 {hi:>4}  {dict(sorted(c.items()))}")

    os.makedirs(DS, exist_ok=True)
    json.dump(picked, open(os.path.join(DS, "full_index.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"\n  → {os.path.join(DS, 'full_index.json')}")
    print(f"  다음: cv_extract_full.py 로 이미지 선별 추출 + 특징 계산")


if __name__ == "__main__":
    main()
