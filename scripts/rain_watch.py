# -*- coding: utf-8 -*-
"""
호우 자동 감시 · CCTV 수집 — ②CV 모듈의 학습 데이터를 스스로 모은다

■ 왜 자동화가 필요한가
  2026-08-18 실제로 겪은 일이다.
      03:51  거제시 호우주의보 발효
      08:13  특보 해제
      18:49  사람이 확인하고 CCTV 캡처 → 노면이 이미 말라 있었다
  호우는 몇 시간이면 끝난다. 사람이 문자를 보고 움직이는 방식으로는
  침수 영상을 절대 확보할 수 없다.

■ 동작
  1. N분마다 긴급재난문자 최신분 확인 (오래된 순 정렬이므로 마지막 페이지)
  2. 호우·침수 관련 문자를 최근 발령분만 골라냄
  3. 수신 지역명을 카카오 로컬 API로 좌표 변환
  4. 그 좌표 반경의 ITS CCTV 프레임을 즉시 캡처
  5. 반복

사용:
    python scripts/rain_watch.py            # 15분 주기로 계속 감시
    python scripts/rain_watch.py --once     # 1회만 확인
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
STATE = os.path.join(SNAP, "watch_state.json")

MSG_API = "https://www.safetydata.go.kr/V2/api/DSSP-IF-00247"
MSG_KEY = get_key("SAFETYDATA_MSG_KEY")
ITS_KEY = get_key("ITS_KEY")
KAKAO_KEY = get_key("KAKAO_REST_KEY")

# 호우로 판단할 조건. '해제'가 들어간 문자는 제외한다(비가 그쳤다는 뜻).
RAIN_KW = ("호우", "침수", "폭우", "홍수", "범람", "집중호우", "많은 비")
STOP_KW = ("해제", "정상운행", "복구")
FRESH_MIN = 180          # 이 시간 안에 발령된 문자만 유효 (분)


def http_json(url, headers=None, timeout=60):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def latest_messages(rows=1000, pages=5):
    """
    ⚠ 이 API는 오래된 순 정렬이다. 최신은 마지막 페이지 쪽에 있다.
      다만 마지막 한 페이지만 읽으면 최신분이 빠진다(실측: 마지막 페이지가
      08-13인데 그 앞 페이지들에 08-18이 들어 있었다). 뒤에서 몇 페이지를 합친다.
    """
    p = {"serviceKey": MSG_KEY, "pageNo": "1", "numOfRows": "1", "returnType": "json"}
    head = http_json(MSG_API + "?" + urllib.parse.urlencode(p))
    total = int(head.get("totalCount") or 0)
    last = (total + rows - 1) // rows
    out = []
    for pg in range(last, max(last - pages, 0), -1):
        p = {"serviceKey": MSG_KEY, "pageNo": str(pg), "numOfRows": str(rows),
             "returnType": "json"}
        try:
            d = http_json(MSG_API + "?" + urllib.parse.urlencode(p), timeout=120)
            out += d.get("body") or []
        except Exception:
            continue
    return out


def parse_dt(s):
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})\s+(\d{2}):(\d{2}):(\d{2})", str(s))
    if not m:
        return None
    import datetime
    return datetime.datetime(*(int(x) for x in m.groups()))


def geocode(region):
    """카카오 로컬 API로 지역명 → 좌표. AWS 관측소 좌표를 못 구해 이 경로를 쓴다."""
    q = urllib.parse.quote(region)
    url = f"https://dapi.kakao.com/v2/local/search/address.json?query={q}"
    try:
        d = http_json(url, headers={"Authorization": f"KakaoAK {KAKAO_KEY}"}, timeout=30)
        docs = d.get("documents") or []
        if docs:
            return float(docs[0]["y"]), float(docs[0]["x"])
    except Exception:
        pass
    # 주소 검색이 실패하면 키워드 검색으로 한 번 더
    try:
        url = f"https://dapi.kakao.com/v2/local/search/keyword.json?query={q}"
        d = http_json(url, headers={"Authorization": f"KakaoAK {KAKAO_KEY}"}, timeout=30)
        docs = d.get("documents") or []
        if docs:
            return float(docs[0]["y"]), float(docs[0]["x"])
    except Exception:
        pass
    return None


def cctv_around(lat, lon, half=0.12):
    out, seen = [], set()
    for t in ("its", "ex"):
        u = (f"https://openapi.its.go.kr:9443/cctvInfo?apiKey={ITS_KEY}&type={t}"
             f"&cctvType=1&minX={lon-half}&maxX={lon+half}"
             f"&minY={lat-half}&maxY={lat+half}&getType=json")
        try:
            d = http_json(u, timeout=60)
        except Exception:
            continue
        for c in (d.get("response") or {}).get("data") or []:
            url = c.get("cctvurl")
            if url and url not in seen:
                seen.add(url)
                out.append(c)
    return out


def capture(cams, tag, limit=10):
    import cv2
    d = os.path.join(SNAP, tag)
    os.makedirs(d, exist_ok=True)
    ts = time.strftime("%H%M%S")
    meta, ok = [], 0
    for i, c in enumerate(cams[:limit]):
        cap = cv2.VideoCapture(c["cctvurl"])
        if not cap.isOpened():
            continue
        r, frame = cap.read()
        cap.release()
        if r:
            p = os.path.join(d, f"{ts}_{i:02d}.jpg")
            good, buf = cv2.imencode(".jpg", frame)
            if good:
                buf.tofile(p)      # cv2.imwrite는 한글 경로를 못 쓴다
                meta.append(dict(file=p, name=c.get("cctvname"),
                                 lat=c.get("coordy"), lon=c.get("coordx"),
                                 at=time.strftime("%Y-%m-%d %H:%M:%S")))
                ok += 1
        time.sleep(0.2)
    if meta:
        with open(os.path.join(d, f"index_{ts}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    return ok


def load_state():
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE, encoding="utf-8"))
        except Exception:
            pass
    return {"seen": []}


def save_state(st):
    st["seen"] = st["seen"][-500:]
    os.makedirs(SNAP, exist_ok=True)
    json.dump(st, open(STATE, "w", encoding="utf-8"), ensure_ascii=False)


def check_once():
    import datetime
    st = load_state()
    seen = set(st["seen"])
    now = datetime.datetime.now()
    try:
        msgs = latest_messages()
    except Exception as e:
        print(f"[{now:%H:%M}] 재난문자 조회 실패: {str(e)[:60]}", flush=True)
        return 0

    hits = []
    for m in msgs:
        sn = str(m.get("SN"))
        if sn in seen:
            continue
        body = str(m.get("MSG_CN") or "")
        kind = str(m.get("DST_SE_NM") or "")
        if not (any(k in body for k in RAIN_KW) or "호우" in kind or "홍수" in kind):
            continue
        if any(k in body for k in STOP_KW):      # 해제 문자는 제외
            continue
        dt = parse_dt(m.get("CRT_DT"))
        if dt is None or (now - dt).total_seconds() > FRESH_MIN * 60:
            continue
        hits.append(m)

    if not hits:
        newest = max((parse_dt(m.get("CRT_DT")) for m in msgs if parse_dt(m.get("CRT_DT"))),
                     default=None)
        print(f"[{now:%H:%M}] 호우 문자 없음 (최신 문자 "
              f"{newest:%m-%d %H:%M})" if newest else f"[{now:%H:%M}] 호우 문자 없음",
              flush=True)
        return 0

    total = 0
    for m in hits:
        sn = str(m.get("SN"))
        seen.add(sn)
        region = str(m.get("RCPTN_RGN_NM") or "").split(",")[0].strip()
        dt = parse_dt(m.get("CRT_DT"))
        print(f"[{now:%H:%M}] ☔ {region} · {dt:%H:%M} · "
              f"{str(m.get('MSG_CN'))[:44]}", flush=True)
        loc = geocode(region)
        if not loc:
            print(f"          좌표 변환 실패: {region}", flush=True)
            continue
        cams = cctv_around(*loc)
        if not cams:
            print(f"          CCTV 없음 ({loc[0]:.3f},{loc[1]:.3f})", flush=True)
            continue
        tag = f"rain_{dt:%Y%m%d}_{re.sub(r'[^가-힣A-Za-z0-9]', '', region)[:12]}"
        n = capture(cams, tag)
        total += n
        print(f"          CCTV {len(cams)}개 중 {n}장 캡처 → {tag}", flush=True)

    st["seen"] = list(seen)
    save_state(st)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=15, help="감시 주기(분)")
    a = ap.parse_args()

    print("■ 호우 자동 감시 시작")
    print(f"  주기 {a.interval}분 · 최근 {FRESH_MIN}분 내 발령 문자만 대상")
    print(f"  저장 위치 {SNAP}\n", flush=True)

    if a.once:
        n = check_once()
        print(f"\n  캡처 {n}장")
        return
    while True:
        try:
            check_once()
        except KeyboardInterrupt:
            print("\n중단")
            return
        except Exception as e:
            print(f"  오류(계속 진행): {str(e)[:70]}", flush=True)
        time.sleep(a.interval * 60)


if __name__ == "__main__":
    main()
