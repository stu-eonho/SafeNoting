# -*- coding: utf-8 -*-
"""
CCTV 침수 감지 데이터셋 구축 — 같은 카메라의 '맑음 vs 비' 짝을 모은다

■ 왜 대조군이 필요한가
  침수 감지 모델에게 필요한 것은 "물이 찬 도로 사진"만이 아니다.
  **같은 카메라가 평상시에 어떻게 보이는지**를 알아야
  "지금이 비정상"이라고 판단할 수 있다.
  카메라마다 각도·조명·주변 구조가 전부 다르므로,
  다른 지점의 맑은 사진은 대조군이 되지 못한다.

■ 라벨 체계
  dry/    비가 오지 않는 평상시  (지금 쌓을 수 있다)
  rain/   비가 내리는 중          (호우 감시로 자동 수집)
  flood/  도로 침수 발생          (드물다 — 확보되면 최고 가치)

■ 카메라 고정 추적
  카메라 URL은 토큰이 붙어 매번 바뀌므로 cctvname을 식별자로 쓴다.
  같은 이름의 카메라를 dry/rain 양쪽에서 모아야 짝이 만들어진다.

사용:
    python scripts/cctv_dataset.py --track          감시 대상 카메라 등록
    python scripts/cctv_dataset.py --dry            평상시 영상 수집
    python scripts/cctv_dataset.py --stat           데이터셋 현황
"""
import sys, os, json, time, re, argparse, ssl, urllib.parse, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import get_key  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = os.path.join(BASE, "data", "cctv_snapshots")
TRACK = os.path.join(SNAP, "tracked_cams.json")
ITS_KEY = get_key("ITS_KEY")

# 침수가 잦거나 위험도가 높은 지점을 우선 추적한다.
# 지하차도는 실제로 사망 사고가 반복된 유형이라 최우선이다.
REGIONS = [
    ("청주",   36.58, 36.72, 127.40, 127.55),
    ("오송",   36.58, 36.68, 127.28, 127.40),
    ("부산초량", 35.09, 35.14, 129.01, 129.06),
    ("거제",   34.80, 35.00, 128.50, 128.75),
]
PRIORITY_KW = ("지하차도", "지하", "교", "천", "강", "터널", "저지대")


def http_json(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout, context=CTX) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def fetch_cams(s, n, w, e):
    out, seen = [], set()
    for t in ("its", "ex"):
        u = (f"https://openapi.its.go.kr:9443/cctvInfo?apiKey={ITS_KEY}&type={t}"
             f"&cctvType=1&minX={w}&maxX={e}&minY={s}&maxY={n}&getType=json")
        try:
            d = http_json(u)
        except Exception:
            continue
        for c in (d.get("response") or {}).get("data") or []:
            nm = c.get("cctvname")
            if nm and nm not in seen:
                seen.add(nm)
                out.append(c)
    return out


def save_img(path, frame):
    import cv2
    ok, buf = cv2.imencode(".jpg", frame)
    if ok:
        buf.tofile(path)          # cv2.imwrite는 한글 경로를 못 쓴다
        return True
    return False


def capture(cams, label, tag):
    """label: dry / rain / flood"""
    import cv2
    d = os.path.join(SNAP, label, tag)
    os.makedirs(d, exist_ok=True)
    meta, ok = [], 0
    for c in cams:
        cap = cv2.VideoCapture(c["cctvurl"])
        if not cap.isOpened():
            continue
        r, frame = cap.read()
        cap.release()
        if r:
            key = re.sub(r'[^가-힣A-Za-z0-9]', '', str(c.get("cctvname")))[:28]
            p = os.path.join(d, f"{key}.jpg")
            if save_img(p, frame):
                meta.append(dict(cam=c.get("cctvname"), file=os.path.basename(p),
                                 lat=c.get("coordy"), lon=c.get("coordx"),
                                 at=time.strftime("%Y-%m-%d %H:%M:%S"), label=label))
                ok += 1
        time.sleep(0.2)
    if meta:
        json.dump(meta, open(os.path.join(d, "index.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    return ok


def cmd_track():
    """추적 대상 카메라를 고정한다. 같은 카메라를 계속 봐야 짝이 생긴다."""
    all_cams = []
    for nm, s, n, w, e in REGIONS:
        cams = fetch_cams(s, n, w, e)
        # 지하차도·교량·하천 인근을 우선
        cams.sort(key=lambda c: (not any(k in str(c.get("cctvname")) for k in PRIORITY_KW),
                                 str(c.get("cctvname"))))
        pick = cams[:12]
        for c in pick:
            c["_region"] = nm
        all_cams += pick
        print(f"  {nm:<8} 전체 {len(cams):>3}개 → 추적 {len(pick)}개")
    json.dump(all_cams, open(TRACK, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  추적 대상 {len(all_cams)}개 → {TRACK}")
    nu = sum(1 for c in all_cams if "지하차도" in str(c.get("cctvname")))
    print(f"  그중 지하차도 {nu}개")


def cmd_dry():
    """평상시(비 안 오는 상태) 영상 수집 — 대조군"""
    if not os.path.exists(TRACK):
        raise SystemExit("먼저 --track 으로 대상 카메라를 등록하세요")
    cams = json.load(open(TRACK, encoding="utf-8"))
    # URL 토큰은 만료되므로 매번 새로 받아야 한다
    fresh = {}
    for nm, s, n, w, e in REGIONS:
        for c in fetch_cams(s, n, w, e):
            fresh[c.get("cctvname")] = c
    use = [fresh[c["cctvname"]] for c in cams if c["cctvname"] in fresh]
    print(f"■ 평상시 영상 수집 · 대상 {len(use)}/{len(cams)}개")
    tag = time.strftime("%Y%m%d_%H%M")
    n = capture(use, "dry", tag)
    print(f"  {n}장 → data/cctv_snapshots/dry/{tag}")


def cmd_stat():
    print("■ 데이터셋 현황")
    per_cam = {}
    for label in ("dry", "rain", "flood"):
        root = os.path.join(SNAP, label)
        imgs = []
        for dirpath, _, files in os.walk(root):
            imgs += [f for f in files if f.endswith(".jpg")]
            for f in files:
                if f == "index.json":
                    try:
                        for m in json.load(open(os.path.join(dirpath, f), encoding="utf-8")):
                            per_cam.setdefault(m["cam"], {"dry": 0, "rain": 0, "flood": 0})
                            per_cam[m["cam"]][m.get("label", label)] += 1
                    except Exception:
                        pass
        print(f"  {label:<6} {len(imgs):>5}장")

    paired = [(k, v) for k, v in per_cam.items() if v["dry"] > 0 and (v["rain"] or v["flood"])]
    print(f"\n■ 맑음·비 짝이 갖춰진 카메라 {len(paired)}개")
    for k, v in paired[:10]:
        print(f"    {str(k)[:32]:<34} dry {v['dry']} · rain {v['rain']} · flood {v['flood']}")
    if not paired:
        print("    (아직 없음 — 비 올 때 rain_watch.py 가 자동 수집한다)")
    only_dry = sum(1 for v in per_cam.values() if v["dry"] and not (v["rain"] or v["flood"]))
    print(f"\n  평상시만 있는 카메라 {only_dry}개 — 호우 시 같은 카메라를 찍어야 짝이 된다")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--stat", action="store_true")
    a = ap.parse_args()
    if a.track:
        cmd_track()
    elif a.dry:
        cmd_dry()
    else:
        cmd_stat()


if __name__ == "__main__":
    main()
