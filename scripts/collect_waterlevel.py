# -*- coding: utf-8 -*-
"""
한강홍수통제소(HRFCO) 장기 수위 시계열 수집기

HRFCO 서버는 응답이 길면 chunked 전송을 도중에 끊는다(2005~2026 확인).
따라서 ① 월 단위로 쪼개 요청하고 ② 잘린 응답도 정규식으로 건질 만큼 건지며
③ 결손이 크면 반으로 분할해 재요청한다.

사용:
    python scripts/collect_waterlevel.py 3011665 2005 2026
"""
import sys, os, re, ssl, time, csv, datetime, calendar, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import get_key  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

KEY = get_key("HRFCO_KEY")
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "timeseries")

# 잘린 JSON에서도 레코드를 건지기 위해 파서 대신 정규식을 쓴다
REC = re.compile(
    r'"wlobscd"\s*:\s*"(?P<cd>\d+)"\s*,\s*'
    r'"ymdhm"\s*:\s*"(?P<t>\d{10,12})"\s*,\s*'
    r'"wl"\s*:\s*"(?P<wl>-?[\d.]+)"'
    r'(?:\s*,\s*"fw"\s*:\s*"(?P<fw>-?[\d.]+)")?'
)


def fetch(url, tries=5, timeout=90):
    """
    잘린 응답도 받은 만큼 반환한다.

    ⚠ HRFCO는 부하가 걸리면 200 OK에 '빈 본문'을 돌려준다.
      예외가 발생하지 않으므로 try/except 재시도로는 걸러지지 않고,
      호출부는 이를 '자료 없음'으로 오인한다.
      실제로 미호강교(22년치 확보 완료된 관측소)조차 재조회 시 0건이 나왔다.
      따라서 본문에 레코드가 실제로 들어 있는지 확인하고, 없으면 재시도한다.
    """
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout, context=CTX) as r:
                txt = r.read().decode("utf-8", errors="replace")
            if '"wl"' in txt:
                return txt
            # 빈 본문 = 서버가 흘린 요청. 점점 길게 쉬었다 다시 묻는다.
        except Exception:
            pass
        if i < tries - 1:
            time.sleep(2 + i * 3)
    return ""


def parse(txt):
    """{시각: (수위, 유량)} — 중복은 자동 제거된다."""
    out = {}
    for m in REC.finditer(txt or ""):
        t = m.group("t")
        if len(t) == 10:
            t += "00"          # 1H 응답은 YYYYMMDDHH
        try:
            out[t] = (float(m.group("wl")),
                      float(m.group("fw")) if m.group("fw") else None)
        except ValueError:
            continue
    return out


def span(sta, res, s, e):
    url = f"http://api.hrfco.go.kr/{KEY}/waterlevel/list/{res}/{sta}/{s}/{e}.json"
    return parse(fetch(url))


def collect_month(sta, res, y, m, expect):
    """월 단위 수집. 결손이 크면 반으로 쪼개 보충한다."""
    last = calendar.monthrange(y, m)[1]
    got = span(sta, res, f"{y}{m:02d}010000", f"{y}{m:02d}{last}2350")

    # fetch()가 빈 본문을 걸러 재시도하므로, 여기서는 반으로 쪼개는 보충만 한다.
    if not got:
        mid = last // 2
        got = span(sta, res, f"{y}{m:02d}010000", f"{y}{m:02d}{mid:02d}2350")
        got.update(span(sta, res, f"{y}{m:02d}{mid+1:02d}0000", f"{y}{m:02d}{last}2350"))

    if 0 < len(got) < expect * 0.9:
        mid = last // 2
        for s, e in ((f"{y}{m:02d}010000", f"{y}{m:02d}{mid:02d}2350"),
                     (f"{y}{m:02d}{mid+1:02d}0000", f"{y}{m:02d}{last}2350")):
            got.update(span(sta, res, s, e))
            time.sleep(0.4)
    return got


def save(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("ymdhm,wl,fw\n")
        for t in sorted(rows):
            wl, fw = rows[t]
            f.write(f"{t},{wl},{'' if fw is None else fw}\n")


def main(sta, y0, y1, res="1H", months=None):
    """months: 수집할 월 목록. None이면 12개월 전부.
    홍수기(6~9월)만 받으면 시간이 1/3로 줄고, 급상승 표본은 대부분 보존된다."""
    os.makedirs(ROOT, exist_ok=True)
    per_day = 24 if res == "1H" else 144
    tag = "" if months is None else "_flood"
    path = os.path.join(ROOT, f"wl_{sta}_{res}{tag}.csv")
    today = datetime.date.today()
    month_list = months or list(range(1, 13))

    # 기존 파일 병합 — 중단 후 재실행해도 처음부터 다시 받지 않는다
    all_rows, empty_years = {}, []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    all_rows[r["ymdhm"]] = (float(r["wl"]),
                                            float(r["fw"]) if r.get("fw") else None)
                except (ValueError, KeyError):
                    pass
        print(f"  기존 {len(all_rows):,}건 로드 후 병합", flush=True)

    started = False          # 이 관측소의 자료가 시작된 시점을 지났는가
    for y in range(y0, y1 + 1):
        ybefore = len(all_rows)
        for mi, m in enumerate(month_list):
            if datetime.date(y, m, 1) > today:      # 미래 월은 요청하지 않는다
                continue
            expect = calendar.monthrange(y, m)[1] * per_day
            got = collect_month(sta, res, y, m, expect)
            all_rows.update(got)
            if got:
                started = True
            # 자료 시작 전이고 그 해 전체가 비었으면 남은 달은 건너뛴다.
            # 관측소 설치 이전 연도를 통째로 요청하는 낭비만 막고,
            # 자료가 한 번이라도 나온 뒤에는 절대 건너뛰지 않는다.
            elif not started and mi >= 2 and len(all_rows) == ybefore:
                break
            time.sleep(0.3)
        n = len(all_rows) - ybefore
        theoretical = sum(calendar.monthrange(y, m)[1] for m in month_list) * per_day
        print(f"  {y}: +{n:>6}건  (이론 {theoretical})  누적 {len(all_rows):>7}", flush=True)
        save(path, all_rows)        # 연도마다 중간 저장 — 중단돼도 진척이 남는다
        if n == 0:
            empty_years.append(y)

    save(path, all_rows)
    print(f"\n총 {len(all_rows):,}건 → {path}")
    if empty_years:
        print(f"데이터 없는 연도: {empty_years}")
    return len(all_rows)


if __name__ == "__main__":
    sta = sys.argv[1] if len(sys.argv) > 1 else "3011665"
    y0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2005
    y1 = int(sys.argv[3]) if len(sys.argv) > 3 else 2026
    res = sys.argv[4] if len(sys.argv) > 4 else "1H"
    # 다섯 번째 인자로 "6-9" 형태를 주면 해당 월만 수집한다
    months = None
    if len(sys.argv) > 5 and sys.argv[5] != "all":
        a, b = sys.argv[5].split("-")
        months = list(range(int(a), int(b) + 1))
    print(f"■ 관측소 {sta} / {res} / {y0}~{y1} / "
          f"{'전월' if months is None else str(months)} 수집 시작", flush=True)
    main(sta, y0, y1, res, months)
