# -*- coding: utf-8 -*-
"""
기상청 ASOS 시간자료 수집기 (공공데이터포털)

수위 예측 모델의 외생변수(강수·기온·풍속·습도)를 확보한다.
초단기실황 API는 "최근 1일"만 제공하므로 과거 학습 데이터는 ASOS에서 받는다.

주의: .env의 DATA_GO_KR_KEY는 Encoding 키(%2B, %3D 형태)라 재인코딩하면 안 된다.

사용:
    python scripts/collect_weather.py 131 2005 2026      # 131 = 청주
"""
import sys, os, ssl, time, json, csv, datetime, urllib.request

sys.stdout.reconfigure(encoding='utf-8')
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(BASE, "data", "timeseries")
API = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"

# 수집 필드: 시각/기온/강수/풍속/풍향/습도
FIELDS = ("tm", "ta", "rn", "ws", "wd", "hm")


def load_key():
    for line in open(os.path.join(BASE, ".env"), encoding="utf-8"):
        line = line.strip()
        if line.startswith("DATA_GO_KR_KEY=") and "DECODED" not in line:
            return line.split("=", 1)[1].split("#")[0].strip()
    raise SystemExit("DATA_GO_KR_KEY 없음")


def year_range(y):
    """연말이 미래면 어제까지로 자른다. ASOS는 미래 날짜를 요청하면 통째로 실패한다."""
    end = datetime.date(y, 12, 31)
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    if end > yesterday:
        end = yesterday
    if end < datetime.date(y, 1, 1):
        return None, None
    return f"{y}0101", end.strftime("%Y%m%d")


def call(key, stn, y, page):
    s, e = year_range(y)
    if s is None:
        return [], 0
    url = (f"{API}?serviceKey={key}&pageNo={page}&numOfRows=999&dataType=JSON"
           f"&dataCd=ASOS&dateCd=HR&startDt={s}&startHh=00"
           f"&endDt={e}&endHh=23&stnIds={stn}")
    for i in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90, context=CTX) as r:
                d = json.loads(r.read().decode("utf-8", errors="replace"))
            body = d["response"]["body"]
            items = body.get("items") or {}
            return items.get("item") or [], int(body.get("totalCount") or 0)
        except Exception:
            if i == 2:
                return [], 0
            time.sleep(2 + i * 2)
    return [], 0


def main(stn, y0, y1):
    os.makedirs(ROOT, exist_ok=True)
    key = load_key()
    path = os.path.join(ROOT, f"asos_{stn}_1H.csv")

    # 기존 파일이 있으면 병합한다 (부분 재수집 시 기존분 보존)
    rows = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("tm"):
                    rows[r["tm"]] = r
        print(f"  기존 {len(rows):,}건 로드 후 병합", flush=True)

    for y in range(y0, y1 + 1):
        first, total = call(key, stn, y, 1)
        if not first:
            print(f"  {y}: 데이터 없음", flush=True)
            continue
        got = list(first)
        for pg in range(2, (total + 998) // 999 + 1):
            items, _ = call(key, stn, y, pg)
            got += items
            time.sleep(0.25)
        for it in got:
            rows[it["tm"]] = it
        print(f"  {y}: +{len(got):>5}건 (총 {total})  누적 {len(rows):>7}", flush=True)
        time.sleep(0.3)

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(FIELDS) + "\n")
        for t in sorted(rows):
            it = rows[t]
            # 강수는 무강수 시 빈 문자열로 오므로 0으로 채운다
            vals = [str(it.get(c, "")).strip() or ("0" if c == "rn" else "")
                    for c in FIELDS]
            f.write(",".join(vals) + "\n")
    print(f"\n총 {len(rows):,}건 → {path}")


if __name__ == "__main__":
    stn = sys.argv[1] if len(sys.argv) > 1 else "131"
    y0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2005
    y1 = int(sys.argv[3]) if len(sys.argv) > 3 else 2026
    print(f"■ ASOS {stn} / {y0}~{y1} 수집 시작", flush=True)
    main(stn, y0, y1)
