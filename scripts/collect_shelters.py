# -*- coding: utf-8 -*-
"""
행정안전부 통합대피소 수집 (safetydata DSSP-IF-10941)

OSM 학교·관공서를 대피소 후보로 쓰던 것을 실제 지정 대피소로 교체한다.
전국 75,692곳에서 대상 범위만 걸러낸다.

⚠ safetydata는 공인 IP 등록 방식이다. IP가 바뀌면 resultCode 32가 돌아온다.

사용:
    python scripts/collect_shelters.py 36.60 36.70 127.42 127.55
    python scripts/collect_shelters.py            # 청주 도심 기본
"""
import sys, os, json, time, urllib.parse, urllib.request, ssl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import get_key  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data", "routing")
API = "https://www.safetydata.go.kr/V2/api/DSSP-IF-10941"
KEY = get_key("SAFETYDATA_SHELTER_KEY")


def call(page, rows=1000, tries=3):
    p = {"serviceKey": KEY, "pageNo": str(page), "numOfRows": str(rows),
         "returnType": "json"}
    url = API + "?" + urllib.parse.urlencode(p)
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90, context=CTX) as r:
                return json.loads(r.read().decode("utf-8", errors="replace"))
        except Exception:
            if i == tries - 1:
                return None
            time.sleep(3 + i * 3)
    return None


def main():
    if len(sys.argv) >= 5:
        s, n, w, e = (float(x) for x in sys.argv[1:5])
    else:
        s, n, w, e = 36.600, 36.700, 127.420, 127.550     # 청주 도심
    print(f"■ 대피소 수집 · 범위 {s}~{n} / {w}~{e}")

    first = call(1, 1)
    if not first or first.get("header", {}).get("resultCode") != "00":
        raise SystemExit(f"API 오류: {first.get('header') if first else '응답 없음'}")
    total = int(first.get("totalCount") or 0)
    pages = (total + 999) // 1000
    print(f"  전국 {total:,}곳 · {pages}페이지 조회")

    found = []
    for pg in range(1, pages + 1):
        d = call(pg)
        if not d:
            print(f"    {pg}페이지 실패", flush=True)
            continue
        for x in (d.get("body") or []):
            try:
                la, lo = float(x["LAT"]), float(x["LOT"])
            except (KeyError, ValueError, TypeError):
                continue
            if s <= la <= n and w <= lo <= e:
                found.append(dict(
                    name=x.get("REARE_NM"), lat=la, lon=lo,
                    kind=x.get("SHLT_SE_NM"), addr=x.get("RONA_DADDR"),
                    capacity=x.get("SHNT_PSBLTY_NOPE")))
        if pg % 15 == 0:
            print(f"    {pg}/{pages}  누적 {len(found)}곳", flush=True)
        time.sleep(0.15)

    print(f"\n■ 범위 내 대피소 {len(found)}곳")
    from collections import Counter
    for k, v in Counter(x["kind"] for x in found).most_common():
        print(f"    {str(k):<18} {v}곳")
    print("\n  샘플")
    for x in found[:6]:
        print(f"    {str(x['name'])[:22]:<24} {x['kind']}  {str(x['addr'])[:28]}")

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "shelters_area.json")
    json.dump(found, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  → {p}")


if __name__ == "__main__":
    main()
