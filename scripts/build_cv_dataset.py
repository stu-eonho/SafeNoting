# -*- coding: utf-8 -*-
"""
AI Hub 135 '부산시 침수위험 복합 데이터' → 학습 데이터셋 구성

■ 이 데이터가 왜 맞는가
  같은 CCTV의 평상시와 호우 시가 라벨과 함께 들어 있다.
      2023-08-24 00:00  강우 약함 · 하천 바닥의 징검다리가 보임
      2023-08-24 05:00  강우 53mm/h · 징검다리 수몰, 산책로 침수
  카메라가 고정이므로 "같은 각도에서 무엇이 달라졌는가"를 학습할 수 있다.
  각도·조명이 다른 남의 사진은 대조군이 되지 못한다.

■ 라벨 (COCO 포맷)
  1 driveway  2 pole  3 pier  4 water_side  5 water_area  6 rain_gutter
  water_area 가 핵심. pier(교각)가 함께 라벨되어 있어 수위의 기준 척도가 된다.

■ 만드는 것
  ① 프레임별 인덱스 (강우량·수역면적·지점·시각)
  ② 수위 등급 라벨 — water_area 면적을 그 카메라의 평상시 대비 비율로 환산
     절대 면적은 카메라마다 화각이 달라 비교할 수 없다.
  ③ 같은 카메라의 (평상시, 호우) 짝

사용:
    python scripts/build_cv_dataset.py
"""
import sys, os, json, re, math
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(BASE, "data", "aihub_135")
LAB = os.path.join(ROOT, "val_label", "침수위험 지역 라벨링 데이터")
IMG = os.path.join(ROOT, "val_image", "침수위험 지역 라벨링 데이터")
OUT = os.path.join(ROOT, "dataset")

CAT = {1: "driveway", 2: "pole", 3: "pier", 4: "water_side",
       5: "water_area", 6: "rain_gutter"}


def poly_area(seg):
    """COCO segmentation(평면 좌표 나열) → 면적. 신발끈 공식."""
    if not seg:
        return 0.0
    total = 0.0
    for ring in (seg if isinstance(seg[0], list) else [seg]):
        xs = ring[0::2]
        ys = ring[1::2]
        if len(xs) < 3:
            continue
        s = 0.0
        for i in range(len(xs)):
            j = (i + 1) % len(xs)
            s += xs[i] * ys[j] - xs[j] * ys[i]
        total += abs(s) / 2.0
    return total


def parse_name(fn):
    """
    E__006_112^20230824-050000-20230824-060000_000012.json
      → site=112, start=20230824-050000, frame=12
    """
    m = re.match(r"([A-Za-z]+)_+(\d+)_(\d+)\^(\d{8})-(\d{6})-\d{8}-\d{6}_(\d+)", fn)
    if not m:
        return None
    kind, seq, site, ymd, hms, frame = m.groups()
    return dict(kind=kind, site=site, ymd=ymd, hms=hms, frame=int(frame),
                clip=f"{site}_{ymd}_{hms}")


def main():
    if not os.path.isdir(LAB):
        raise SystemExit(f"라벨 폴더 없음: {LAB}\n압축 해제가 끝났는지 확인하세요")
    os.makedirs(OUT, exist_ok=True)

    rows = []
    n_bad = 0
    for dirpath, _, files in os.walk(LAB):
        for f in files:
            if not f.endswith(".json"):
                continue
            p = os.path.join(dirpath, f)
            try:
                d = json.load(open(p, encoding="utf-8"))
            except Exception:
                n_bad += 1
                continue
            info = d.get("info") or {}
            imgs = d.get("images") or [{}]
            im = imgs[0]
            meta = parse_name(f) or {}
            # 카메라 지점: 경로에 '112_사상구_학장교' 형태로 들어 있다
            m = re.search(r"(\d+_[^\\/]+)", dirpath.replace(LAB, ""))
            cam = m.group(1) if m else meta.get("site", "?")

            area = defaultdict(float)
            for a in d.get("annotations") or []:
                cid = a.get("category_id")
                area[CAT.get(cid, str(cid))] += float(a.get("area") or
                                                      poly_area(a.get("segmentation")))
            W = float(im.get("width") or 1280)
            H = float(im.get("height") or 720)
            rows.append(dict(
                file=im.get("file_name") or f.replace(".json", ".jpg"),
                cam=cam, clip=meta.get("clip"), ymd=meta.get("ymd"),
                hms=meta.get("hms"), frame=meta.get("frame"),
                rain_10m=float(info.get("10m_rainfall") or 0),
                rain_1h=float(info.get("1hr_rainfall") or 0),
                rain_3h=float(info.get("3hr_rainfall") or 0),
                water=area.get("water_area", 0.0),
                water_side=area.get("water_side", 0.0),
                pier=area.get("pier", 0.0),
                driveway=area.get("driveway", 0.0),
                water_ratio=area.get("water_area", 0.0) / (W * H),
                label_path=os.path.relpath(p, ROOT)))
    print(f"■ 라벨 {len(rows):,}건 파싱 (실패 {n_bad})")

    # ── 카메라별 정규화 ─────────────────────────────────────
    # 절대 면적은 화각이 달라 비교 불가하므로 카메라별 기준면적으로 나눈다.
    #
    # ⚠ 처음에는 '하위 20% 면적'을 기준으로 삼았다가 실패했다.
    #   water_area가 0인 프레임이 많은 카메라에서 기준이 0에 수렴해
    #   배율이 352배까지 폭발하고 등급 분포가 역전됐다(경계 0.9% < 위험 5.2%).
    #   → 0을 뺀 중앙값을 기준으로 쓰고, 기준이 화면의 1% 미만인 카메라는
    #     수역이 사실상 안 잡히는 것이므로 제외한다.
    MIN_BASE_RATIO = 0.01
    FRAME_PX = 1280 * 720
    by_cam = defaultdict(list)
    for r in rows:
        by_cam[r["cam"]].append(r)

    kept, dropped = {}, []
    for cam, rs in by_cam.items():
        pos = sorted(x["water"] for x in rs if x["water"] > 0)
        if not pos:
            dropped.append((cam, len(rs), 0.0))
            continue
        base = pos[len(pos) // 2]
        if base < FRAME_PX * MIN_BASE_RATIO:
            dropped.append((cam, len(rs), base))
            continue
        kept[cam] = base
        for r in rs:
            r["water_rel"] = r["water"] / base

    rows = [r for r in rows if r["cam"] in kept]
    by_cam = {c: v for c, v in by_cam.items() if c in kept}
    print(f"\n■ 카메라 {len(kept)}개 사용 · {len(dropped)}개 제외 "
          f"(유효 프레임 {len(rows):,}건)")
    for cam in sorted(kept, key=lambda c: -len(by_cam[c])):
        rs, base = by_cam[cam], kept[cam]
        rain = [x["rain_1h"] for x in rs]
        print(f"    {cam:<28} {len(rs):>5}프레임  강우 {min(rain):.0f}~{max(rain):.0f}mm  "
              f"기준 {base:,.0f}px ({base/FRAME_PX*100:.1f}%)")
    for cam, n, base in dropped:
        print(f"    [제외] {cam:<24} {n:>5}프레임  기준 {base:,.0f}px")

    # ── 수위 등급 ───────────────────────────────────────────
    # 평상시 대비 배율로 4단계. 절대 수심을 알 수 없으므로 상대 등급으로 둔다.
    def grade(rel):
        if rel < 1.3:
            return 0        # 평상
        if rel < 2.0:
            return 1        # 증가
        if rel < 3.0:
            return 2        # 경계
        return 3            # 위험
    for r in rows:
        r["level"] = grade(r.get("water_rel", 1.0))

    from collections import Counter
    print(f"\n■ 수위 등급 분포 (평상시 수역면적 대비)")
    gc = Counter(r["level"] for r in rows)
    names = {0: "0 평상 (<1.3배)", 1: "1 증가 (1.3~2배)",
             2: "2 경계 (2~3배)", 3: "3 위험 (3배↑)"}
    for k in sorted(gc):
        print(f"    {names[k]:<20} {gc[k]:>6,}건 ({gc[k]/len(rows)*100:>5.1f}%)")

    # ── 강우량 ↔ 수위 관계 ──────────────────────────────────
    print(f"\n■ 1시간 강우량 구간별 평균 수역면적 배율")
    bins = [(0, 1), (1, 5), (5, 15), (15, 30), (30, 100)]
    for lo, hi in bins:
        sel = [r for r in rows if lo <= r["rain_1h"] < hi]
        if not sel:
            continue
        avg = sum(r["water_rel"] for r in sel) / len(sel)
        hi_lv = sum(1 for r in sel if r["level"] >= 2) / len(sel) * 100
        print(f"    {lo:>3}~{hi:<4}mm  n={len(sel):>5,}  평균 {avg:>5.2f}배  "
              f"경계이상 {hi_lv:>5.1f}%")

    # ── 맑음/호우 짝 ────────────────────────────────────────
    pairs = []
    for cam, rs in by_cam.items():
        dry = [r for r in rs if r["level"] == 0 and r["rain_1h"] < 1]
        wet = [r for r in rs if r["level"] >= 2]
        if dry and wet:
            pairs.append((cam, len(dry), len(wet)))
    print(f"\n■ 같은 카메라의 (평상시, 경계이상) 짝 {len(pairs)}개")
    for cam, nd, nw in pairs:
        print(f"    {cam:<28} 평상 {nd:>5,}장 · 경계이상 {nw:>5,}장")

    rows.sort(key=lambda r: (r["cam"], r["ymd"] or "", r["hms"] or "", r["frame"] or 0))
    json.dump(rows, open(os.path.join(OUT, "index.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    with open(os.path.join(OUT, "index.csv"), "w", encoding="utf-8", newline="") as f:
        cols = ["file", "cam", "ymd", "hms", "frame", "rain_10m", "rain_1h", "rain_3h",
                "water", "water_rel", "level", "pier", "driveway"]
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\n  → {os.path.join(OUT, 'index.json')}")
    print(f"  → {os.path.join(OUT, 'index.csv')}")


if __name__ == "__main__":
    main()
