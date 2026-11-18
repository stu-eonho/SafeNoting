# -*- coding: utf-8 -*-
"""
긴급재난문자 수집 (safetydata DSSP-IF-00247)

이 프로젝트의 출발점이 된 데이터다.
    "재난문자는 90자 제한 때문에 '위험이 있다'만 전하고
     '어디로 어떻게 대피하라'는 전하지 못한다"
그 주장을 실제 문안으로 증명하려면 원문이 필요하다.

⚠ 이 API는 **오래된 순으로 정렬**된다.
  첫 페이지는 2023-09-19, 마지막 페이지가 최신(2026-08)이다.
  첫 페이지만 보고 "옛날 데이터"로 판단하면 안 된다.

사용:
    python scripts/collect_disaster_msg.py            # 최근 3000건에서 호우 관련 추출
    python scripts/collect_disaster_msg.py 8000       # 더 많이
"""
import sys, os, json, time, urllib.parse, urllib.request, ssl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import get_key  # noqa: E402
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data", "disaster_msg")
API = "https://www.safetydata.go.kr/V2/api/DSSP-IF-00247"
KEY = get_key("SAFETYDATA_MSG_KEY")
ROWS = 1000

FLOOD_KW = ("호우", "침수", "폭우", "홍수", "하천", "범람", "물바다", "수위",
            "지하차도", "산사태", "대피", "저지대", "배수")


def call(page, rows=ROWS, tries=3):
    p = {"serviceKey": KEY, "pageNo": str(page), "numOfRows": str(rows),
         "returnType": "json"}
    url = API + "?" + urllib.parse.urlencode(p)
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120, context=CTX) as r:
                return json.loads(r.read().decode("utf-8", errors="replace"))
        except Exception:
            if i == tries - 1:
                return None
            time.sleep(4 + i * 4)
    return None


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    os.makedirs(OUT, exist_ok=True)

    first = call(1, 1)
    if not first or first.get("header", {}).get("resultCode") != "00":
        raise SystemExit(f"API 오류: {first.get('header') if first else '응답 없음'}")
    total = int(first.get("totalCount") or 0)
    last = (total + ROWS - 1) // ROWS
    print(f"■ 재난문자 전체 {total:,}건 · {last}페이지")
    print(f"  ※ 오래된 순 정렬 — 최신을 보려면 마지막 페이지부터 읽는다")

    got, pages = [], 0
    for pg in range(last, 0, -1):
        d = call(pg)
        if not d:
            print(f"    p{pg} 실패", flush=True)
            continue
        body = d.get("body") or []
        got += body
        pages += 1
        if pages % 3 == 0:
            print(f"    p{pg}  누적 {len(got):,}건  "
                  f"(최고 {got[-1].get('CRT_DT','')[:10]})", flush=True)
        if len(got) >= want:
            break
        time.sleep(0.2)

    got.sort(key=lambda x: str(x.get("CRT_DT") or ""))
    print(f"\n■ 수집 {len(got):,}건 · {got[0].get('CRT_DT')} ~ {got[-1].get('CRT_DT')}")

    print(f"\n■ 재난 유형 분포")
    for k, v in Counter(x.get("DST_SE_NM") for x in got).most_common(8):
        print(f"    {str(k):<16} {v:>5}건")

    flood = [x for x in got
             if any(k in str(x.get("MSG_CN", "")) for k in FLOOD_KW)
             or any(k in str(x.get("DST_SE_NM", "")) for k in ("호우", "홍수", "태풍"))]
    print(f"\n■ 호우·침수 관련 {len(flood):,}건 ({len(flood)/max(len(got),1)*100:.1f}%)")

    lens = [len(str(x.get("MSG_CN", ""))) for x in flood]
    if lens:
        lens_s = sorted(lens)
        print(f"  문자 길이: 중앙값 {lens_s[len(lens_s)//2]}자 · "
              f"최대 {max(lens)}자 · 90자 이하 {sum(1 for l in lens if l <= 90)}건 "
              f"({sum(1 for l in lens if l <= 90)/len(lens)*100:.0f}%)")

    print(f"\n■ 최근 호우 문자 5건 (Before 자료)")
    for x in flood[-5:]:
        msg = str(x.get("MSG_CN", ""))
        print(f"\n  [{x.get('CRT_DT')}] {x.get('DST_SE_NM')} / {x.get('EMRG_STEP_NM')}")
        print(f"    수신: {str(x.get('RCPTN_RGN_NM'))[:50]}")
        print(f"    본문({len(msg)}자): {msg[:100]}")

    json.dump(got, open(os.path.join(OUT, "msg_recent.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(flood, open(os.path.join(OUT, "msg_flood.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n  → data/disaster_msg/msg_recent.json ({len(got):,}건)")
    print(f"  → data/disaster_msg/msg_flood.json ({len(flood):,}건)")


if __name__ == "__main__":
    main()
