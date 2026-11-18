# -*- coding: utf-8 -*-
"""
CV 데이터셋 분할 — 클립 단위로 나눈다

■ 프레임 단위로 랜덤 분할하면 안 된다
  한 클립(같은 1시간 녹화분)의 프레임들은 거의 동일한 장면이다.
  랜덤으로 섞으면 train에 있던 장면이 val에도 들어가
  성능이 가짜로 높게 나온다. 수위 예측에서 시간축을 지켰던 것과 같은 이유다.

■ 분할 기준
  1순위: 클립(site+날짜+시각) 단위로 완전히 분리
  2순위: 각 등급이 train/val 양쪽에 들어가도록 배분
         (경계·위험 등급이 전체의 11%뿐이라 한쪽에 몰리기 쉽다)

사용:
    python scripts/split_cv_dataset.py
"""
import sys, os, json, random
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(BASE, "data", "aihub_135", "dataset")


def main():
    rows = json.load(open(os.path.join(DS, "index.json"), encoding="utf-8"))
    print(f"■ 전체 {len(rows):,}프레임")

    by_clip = defaultdict(list)
    for r in rows:
        key = (r["cam"], r.get("clip") or f"{r.get('ymd')}_{r.get('hms')}")
        by_clip[key].append(r)
    print(f"  클립 {len(by_clip)}개 · 카메라 {len({k[0] for k in by_clip})}개")

    # 클립의 대표 등급 = 그 클립에서 가장 높은 등급
    # (한 클립 안에서 수위가 오르면 마지막 프레임이 위험일 수 있다)
    clip_info = []
    for key, rs in by_clip.items():
        top = max(r["level"] for r in rs)
        clip_info.append(dict(cam=key[0], clip=key[1], n=len(rs), top=top,
                              rain3=max(r["rain_3h"] for r in rs)))
    print(f"\n■ 클립 대표등급 분포")
    for k, v in sorted(Counter(c["top"] for c in clip_info).items()):
        print(f"    등급 {k}: {v:>3}클립")

    # 등급별로 층화 분할 — 위험 등급 클립이 적어 한쪽에 몰리면 평가가 무의미해진다
    random.seed(42)
    train, val = [], []
    for lv in sorted({c["top"] for c in clip_info}):
        grp = [c for c in clip_info if c["top"] == lv]
        random.shuffle(grp)
        cut = max(1, int(len(grp) * 0.25)) if len(grp) > 1 else 0
        val += grp[:cut]
        train += grp[cut:]

    tr_keys = {(c["cam"], c["clip"]) for c in train}
    va_keys = {(c["cam"], c["clip"]) for c in val}
    assert not (tr_keys & va_keys), "클립이 양쪽에 겹쳤다"

    tr_rows = [r for r in rows
               if (r["cam"], r.get("clip") or f"{r.get('ymd')}_{r.get('hms')}") in tr_keys]
    va_rows = [r for r in rows
               if (r["cam"], r.get("clip") or f"{r.get('ymd')}_{r.get('hms')}") in va_keys]

    print(f"\n■ 분할 결과 (클립 단위 · 겹침 없음)")
    for nm, rs, cl in (("train", tr_rows, train), ("val", va_rows, val)):
        gc = Counter(r["level"] for r in rs)
        print(f"  {nm:<6} 클립 {len(cl):>3}개 · 프레임 {len(rs):>6,}  "
              f"등급 " + " ".join(f"{k}:{gc.get(k,0):,}" for k in range(4)))

    hi_tr = sum(1 for r in tr_rows if r["level"] >= 2)
    hi_va = sum(1 for r in va_rows if r["level"] >= 2)
    print(f"\n  경계이상 프레임  train {hi_tr:,} / val {hi_va:,}")
    if hi_va < 30:
        print("  ⚠ val의 경계이상 표본이 적어 평가가 불안정할 수 있다")

    out = dict(
        train=[dict(file=r["file"], cam=r["cam"], level=r["level"],
                    rain_1h=r["rain_1h"], rain_3h=r["rain_3h"],
                    water_rel=round(r["water_rel"], 4)) for r in tr_rows],
        val=[dict(file=r["file"], cam=r["cam"], level=r["level"],
                  rain_1h=r["rain_1h"], rain_3h=r["rain_3h"],
                  water_rel=round(r["water_rel"], 4)) for r in va_rows])
    json.dump(out, open(os.path.join(DS, "split.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"\n  → {os.path.join(DS, 'split.json')}")


if __name__ == "__main__":
    main()
