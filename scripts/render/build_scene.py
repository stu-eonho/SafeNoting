# -*- coding: utf-8 -*-
"""
장면 명세(scene spec) 빌더 — 실데이터 → 렌더러가 먹는 단일 계약

■ 왜 중간 형식을 두는가
  렌더러가 원본 파일을 직접 읽으면, 스키마가 갈린 다섯 가지 함정을
  렌더러마다 다시 밟는다. 장면 명세를 사이에 두면 렌더러는
  "무엇을 어디에 그릴지"만 알면 되고, 원본을 아는 코드는 datasource 하나뿐이다.

■ provenance 와 readings 가 핵심이다
  · provenance : 이 장면이 어느 파일에서 나왔는지
  · readings   : 화면에 뜨는 모든 수치의 실측값
  렌더러는 숫자를 지어내지 않고 readings 만 읽는다. 이게 있으면
  모든 수치가 원본 파일로 추적되고,
  "이 수치 어디서 났나요"에 파일명으로 답할 수 있다.

■ 도시 배분 (취향이 아니라 데이터가 결정한 것)
  오송     수위 관측소·타임라인·침수흔적·대피소가 있다. 시간축 서사 담당.
           과거에는 "오송 최단경로가 침수흔적을 지나지 않는다"고 적혀 있었으나,
           그건 위험도를 폴리곤 '중심점 거리'로만 재던 버그가 만든 착시였다.
           실제 통과 판정을 넣자 최단경로는 침수구역 2곳을 1024m 지난다.
           오송에서도 우회 서사가 데이터로 성립한다(+233m).
  청주     도로망이 조밀하고 지하차도 속성이 그래프에 있다. 경로 서사 담당.
  부산초량  2020년 초량지하차도 참사 지점. 보조 사례.

사용:
    python build_scene.py list
    python build_scene.py underpass flood-detour osong-predictive app
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasource import DS, check_coverage, in_bbox  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

SLIDE_AR = 940 / 1080     # Before/After 패널 한 칸
PHONE_AR = 393 / 852


# ════════════════════════════════════════════════════════════════════
# 좌표 유틸
# ════════════════════════════════════════════════════════════════════

def bbox_of(coords, margin=0.25):
    la = [c[0] for c in coords]
    lo = [c[1] for c in coords]
    dla = (max(la) - min(la)) or 0.004
    dlo = (max(lo) - min(lo)) or 0.004
    return [min(la) - dla * margin, max(la) + dla * margin,
            min(lo) - dlo * margin, max(lo) + dlo * margin]


def fit_aspect(bbox, aspect):
    """
    bbox 를 목표 가로세로비에 맞춘다.
    이걸 안 하면 투영기가 남는 공간을 여백으로 두고 지도가 작게 뜬다.
    """
    la0, la1, lo0, lo1 = bbox
    k = math.cos(math.radians((la0 + la1) / 2))
    h, w = la1 - la0, (lo1 - lo0) * k
    if w / h < aspect:
        half = h * aspect / k / 2
        cx = (lo0 + lo1) / 2
        return [la0, la1, cx - half, cx + half]
    need = w / aspect
    cy = (la0 + la1) / 2
    return [cy - need / 2, cy + need / 2, lo0, lo1]


def _layers(ds, city, bbox):
    ctx = ds.context(city, bbox)
    return {"roads": ds.roads(city, bbox), "buildings": ctx["buildings"],
            "water": ctx["water"], "rail": ctx["rail"],
            "landmarks": ctx["landmarks"]}


def _finish(scene, warn=True):
    ok, problems = check_coverage(scene["bbox"], scene["layers"])
    scene["coverage_ok"] = ok
    scene["coverage_problems"] = problems
    if warn and not ok:
        for p in problems:
            print(f"  ⚠ {p}")
    return scene


def _model_pred(ds, hour):
    """
    scripts/predict_osong.py --json 이 저장한 모델의 3시간 뒤 예측.
    '3시간 뒤' 자리에 실측값을 넣으면 예측처럼 읽히므로, 예측은 여기서만 가져온다.
    """
    try:
        rep = ds.jload("routing", "osong_replay.json")
    except FileNotFoundError:
        return None
    key = "20230715" + hour.replace(":", "")
    row = next((r for r in rep.get("rows", []) if r["time"] == key), None)
    return row["pred_m"] if row else None


def _readings(row, ahead, st, tl, i, pred=None):
    """타임라인 한 시점의 실측 수치와 모델 예측. 화면에 뜨는 숫자는 전부 여기서 나온다."""
    wl = row["wl"]
    return {
        "time": row["time"], "station": st["name"], "station_code": st["code"],
        "wl": wl, "wl_ahead": ahead["wl"], "wl_pred": pred,
        "ahead_hours": min(3, len(tl) - 1 - i),
        "rain": row["rain"], "rain3h": row["rain3h"],
        "alert_level": st["alert_level_m"], "danger_level": st["danger_level_m"],
        "level_name": ("심각" if wl >= st["danger_level_m"]
                       else "경계" if wl >= st["alert_level_m"]
                       else "주의" if wl >= st["alert_level_m"] * 0.7 else "관심"),
    }


# ════════════════════════════════════════════════════════════════════
# 장면 ① 지하차도 회피 — 경로 서사
# ════════════════════════════════════════════════════════════════════

def scene_underpass(ds, city="cheongju", aspect=SLIDE_AR):
    """
    최단경로는 지하차도를 지나고, 안전경로는 우회한다.

    출발·도착 좌표를 지어내지 않는다. 그래프에서 지하차도를 사이에 둔
    실제 도로 노드 쌍을 찾아, 프로젝트의 실제 라우팅 규칙으로 두 경로를
    계산한다. 우회 대가가 5~60%인 사례만 채택한다 —
    +5m 면 그림에서 안 보이고, +2.6km 면 메시지가 반대가 된다.
    """
    case = ds.find_underpass_case(city, want=(5, 60))
    if not case:
        raise SystemExit(f"{city}: 지하차도 회피 사례를 찾지 못했다")
    short, safe = case["short"], case["safe"]
    detour = safe["length_m"] - short["length_m"]
    bbox = fit_aspect(bbox_of(short["coords"] + safe["coords"], 0.30), aspect)
    ups = ds.underpasses(city, bbox, names=[case["underpass"]])
    return _finish({
        "id": f"underpass-{city}",
        "kind": "map",
        "city": city,
        "theme": "dark",
        "title": "침수 시 지하차도는 '가장 빠른 길'이자 '가장 위험한 길'이다",
        "subtitle": f"청주 {case['underpass']} 지하차도  —  우회 대가 +{detour}m "
                    f"(도보 약 {max(1, round(detour / 67))}분)",
        "bbox": bbox,
        "layers": _layers(ds, city, bbox),
        "routes": [
            {"role": "risk", "coords": short["coords"],
             "length_m": short["length_m"],
             "label": f"최단경로 {short['length_m']}m · 지하차도 통과",
             "underpasses": short["underpasses"]},
            {"role": "safe", "coords": safe["coords"],
             "length_m": safe["length_m"],
             "label": f"안전경로 {safe['length_m']}m · 지하차도 회피",
             "underpasses": safe["underpasses"]},
        ],
        "underpass_segs": [u["pts"] for u in ups],
        "flood_polys": [],
        "markers": [
            {"lat": case["origin"][0], "lon": case["origin"][1], "kind": "here",
             "label": "현재 위치"},
            {"lat": case["dest"][0], "lon": case["dest"][1], "kind": "dest",
             "label": "대피 목적지"},
        ],
        "facts": [
            {"k": "최단경로", "v": f"{short['length_m']}m", "tone": "risk"},
            {"k": "안전경로", "v": f"{safe['length_m']}m", "tone": "safe"},
            {"k": "우회 대가", "v": f"+{detour}m", "tone": "neutral"},
            {"k": "차단한 지하차도", "v": case["underpass"], "tone": "risk"},
        ],
        "sources": ["도로망 OpenStreetMap",
                    "지하차도 OSM tunnel 태그 자동 식별",
                    "경로 = 본 시스템의 위험가중 최단경로 알고리즘"],
        "provenance": {"files": [f"data/routing/graph_{city}.pkl",
                                 f"data/routing/context_{city}.json"],
                       "computed": "datasource.find_underpass_case()"},
    })


# ════════════════════════════════════════════════════════════════════
# 장면 ② 침수구역 회피 + 대피소 재평가
# ════════════════════════════════════════════════════════════════════

def scene_flood_detour(ds, city="cheongju", aspect=SLIDE_AR):
    fr = ds.flood_route()
    c = max(fr["cases"], key=lambda x: x.get("detour", 0))
    bbox = fit_aspect(bbox_of(c["short"] + c["safe"], 0.35), aspect)
    polys = ds.flood_polys(bbox)
    safe, unsafe = ds.shelters(bbox)
    detour = round(c["L2"] - c["L1"])
    mk = [{"lat": c["short"][0][0], "lon": c["short"][0][1], "kind": "here",
           "label": "현재 위치"}]
    mk += [{"lat": s["lat"], "lon": s["lon"], "kind": "shelter",
            "label": s["name"], "elev": s.get("elev"),
            "flood_dist": s.get("flood_dist")} for s in safe[:8]]
    mk += [{"lat": s["lat"], "lon": s["lon"], "kind": "blocked", "label": ""}
           for s in unsafe[:14]]
    return _finish({
        "id": f"flood-detour-{city}",
        "kind": "map",
        "city": city,
        "theme": "dark",
        "title": "대피 경로 한가운데 침수 구역이 있다",
        "subtitle": f"최단경로는 침수도로 {c['n_flood']}개를 지난다 "
                    f"(최대 수심 {round(c['max_depth'], 2)}m)  —  우회 +{detour}m",
        "bbox": bbox,
        "layers": _layers(ds, city, bbox),
        "routes": [
            {"role": "risk", "coords": c["short"], "length_m": round(c["L1"]),
             "label": f"최단경로 {round(c['L1'])}m · 침수구역 통과"},
            {"role": "safe", "coords": c["safe"], "length_m": round(c["L2"]),
             "label": f"안전경로 {round(c['L2'])}m · 침수구역 회피"},
        ],
        "flood_polys": [p["pts"] for p in polys],
        "flood_meta": polys,
        "underpass_segs": [],
        "markers": mk,
        "shelters": safe[:8],
        "facts": [
            {"k": "통과 침수도로", "v": f"{c['n_flood']}개", "tone": "risk"},
            {"k": "최대 수심", "v": f"{round(c['max_depth'], 2)}m", "tone": "risk"},
            {"k": "우회 대가", "v": f"+{detour}m", "tone": "neutral"},
            {"k": "이용 가능 대피소", "v": f"{len(safe)}곳", "tone": "safe"},
            {"k": "침수 시 부적합", "v": f"{len(unsafe)}곳", "tone": "risk"},
        ],
        "sources": ["침수흔적도 행정안전부", "도로망 OpenStreetMap",
                    "통합대피소 행정안전부"],
        "provenance": {"files": ["data/routing/flood_route.json",
                                 f"data/routing/graph_{city}.pkl",
                                 f"data/routing/context_{city}.json"]},
    })


# ════════════════════════════════════════════════════════════════════
# 장면 ③ 오송 — 공간 × 시간
# ════════════════════════════════════════════════════════════════════

def scene_osong_predictive(ds, aspect=SLIDE_AR, hour="05:00"):
    loc = ds.locations()
    inc, st = loc["incident"], loc["waterlevel_station"]
    tl = ds.osong_timeline()
    row = next((r for r in tl if r["time"] == hour), tl[5])
    i = tl.index(row)
    ahead = tl[min(i + 3, len(tl) - 1)]
    pred = _model_pred(ds, hour)
    if pred is None:
        raise SystemExit("osong_replay.json 없음 — python scripts/predict_osong.py --json 을 먼저 실행")
    over = "초과" if pred >= st["danger_level_m"] else "미만"
    bbox = fit_aspect([inc["lat"] - 0.016, inc["lat"] + 0.016,
                       inc["lng"] - 0.022, inc["lng"] + 0.022], aspect)
    polys = ds.flood_marks(bbox, "chungbuk")
    return _finish({
        "id": "osong-predictive",
        "kind": "map",
        "city": "osong",
        "theme": "dark",
        "title": "침수흔적도는 '어디'를, 수위예측은 '언제'를 알려준다",
        "subtitle": f"{st['name']} {hour} {row['wl']}m → 3시간 뒤 예측 {pred}m "
                    f"(심각수위 {st['danger_level_m']}m {over}, {ahead['time']} 실측 {ahead['wl']}m)",
        "bbox": bbox,
        "layers": _layers(ds, "osong", bbox),
        "routes": [],
        "flood_polys": [p["pts"] for p in polys],
        "flood_meta": polys,
        "underpass_segs": [u["pts"] for u in ds.underpasses("osong", bbox)],
        "markers": [{"lat": inc["lat"], "lon": inc["lng"], "kind": "incident",
                     "label": "궁평2지하차도"}],
        "facts": [
            {"k": f"{hour} 수위", "v": f"{row['wl']}m", "tone": "risk"},
            {"k": "3시간 뒤 예측", "v": f"{pred}m", "tone": "risk"},
            {"k": f"{ahead['time']} 실측", "v": f"{ahead['wl']}m", "tone": "neutral"},
            {"k": "참사 발생", "v": "08:40", "tone": "risk"},
        ],
        "timeline": tl,
        "readings": _readings(row, ahead, st, tl, i, pred),
        "sources": ["침수흔적도 행정안전부",
                    f"수위 한강홍수통제소 {st['name']}({st['code']})",
                    "강수 기상청 청주 ASOS",
                    "예측 scripts/predict_osong.py"],
        "provenance": {"files": ["data/osong_2023/osong_20230715_timeline.csv",
                                 "data/osong_2023/locations.json",
                                 "data/osong_2023/fludmarks_chungbuk.json",
                                 "data/routing/osong_replay.json"]},
    })


# ════════════════════════════════════════════════════════════════════
# 장면 ④ 앱 화면 — 오송 시나리오
# ════════════════════════════════════════════════════════════════════

def scene_app(ds, hour="06:00"):
    """
    앱 화면에 뜨는 수치는 전부 readings 에 실측값으로 담는다.

    ⚠ 렌더러가 숫자를 지어내면 추적성이 깨지고, 그 순간 이 이미지는
      수치의 출처를 확인할 수 없게 된다. 청주 장면에 미호강 수위를 적어 넣는
      사고가 실제로 났다 — 그래서 수치는 여기서만 만든다.
    """
    loc = ds.locations()
    inc, st = loc["incident"], loc["waterlevel_station"]
    tl = ds.osong_timeline()
    row = next((r for r in tl if r["time"] == hour), tl[6])
    i = tl.index(row)
    ahead = tl[min(i + 3, len(tl) - 1)]
    pred = _model_pred(ds, hour)

    wide = [inc["lat"] - 0.02, inc["lat"] + 0.02,
            inc["lng"] - 0.026, inc["lng"] + 0.026]
    polys = ds.flood_marks(wide, "chungbuk")
    all_sh = ds.osong_shelters(wide)
    # 대피소 재평가 규칙과 같은 기준: 실내 시설이고 침수흔적에서 200m 이상
    shelters = [x for x in all_sh if not x["outdoor"] and not x["flood_inside"]
                and x["flood_km"] is not None and x["flood_km"] >= 0.2]
    if not shelters:
        raise SystemExit("기준을 통과한 오송 대피소를 찾지 못했다")
    # 직선거리가 가까워도 침수흔적을 피해 가는 길은 멀 수 있다.
    # 가까운 후보 몇 곳의 안전경로를 실제로 계산해, 침수구역을 지나지 않는
    # 경로 중 가장 짧은 곳을 목적지로 고른다.
    origin = (inc["lat"], inc["lng"])
    cands = []
    for sh in shelters[:8]:
        r = ds.route("osong", origin, (sh["lat"], sh["lon"]), "safe", flood_polys=polys)
        if r:
            cands.append((r.get("flood_hits", 0) > 0, r["length_m"], sh, r))
    if not cands:
        raise SystemExit("오송 경로 계산 실패")
    _, _, dest, r_safe = min(cands, key=lambda c: (c[0], c[1]))

    # flood_polys 를 short 에도 넘긴다. mode="short" 는 가중치에 쓰지 않고
    # 결과 경로가 침수구역을 밟는지 사후 판정하는 데만 쓴다.
    r_short = ds.route("osong", origin, (dest["lat"], dest["lon"]), "short",
                       avoid_underpass=False, flood_polys=polys)
    if not r_short:
        raise SystemExit("오송 경로 계산 실패")

    bbox = fit_aspect(bbox_of(r_short["coords"] + r_safe["coords"], 0.45),
                      PHONE_AR)
    near_polys = ds.flood_marks(bbox, "chungbuk")
    near_shelters = [x for x in shelters
                     if in_bbox(x["lat"], x["lon"], bbox)][:6]

    mk = [{"lat": inc["lat"], "lon": inc["lng"], "kind": "here",
           "label": "현재 위치"}]
    mk += [{"lat": s["lat"], "lon": s["lon"], "kind": "shelter",
            "label": s["name"], "dist_km": s["dist_km"], "kind_name": s["kind"]}
           for s in near_shelters]

    msgs = ds.disaster_msgs(region="청주")
    msg = next((m for m in msgs if "수위" in m["text"] or "통제" in m["text"]),
               msgs[0] if msgs else None)

    rd = _readings(row, ahead, st, tl, i, pred)
    detour = r_safe["length_m"] - r_short["length_m"]
    rd.update({
        "dest_name": dest["name"], "dest_kind": dest["kind"],
        "dest_dist_km": dest["dist_km"],
        "dest_flood_m": round(dest["flood_km"] * 1000),
        "route_m": r_safe["length_m"], "short_m": r_short["length_m"],
        "detour_m": detour,
        "walk_min": max(1, round(r_safe["length_m"] / 67)),
        "blocked_underpasses": r_short["underpasses"],
        # 최단경로가 실제로 밟는 침수구역 — 재탐색 화면의 근거다
        "short_flood_hits": r_short.get("flood_hits", 0),
        "short_flood_m": r_short.get("flood_hit_m", 0),
        "safe_flood_hits": r_safe.get("flood_hits", 0),
        "n_flood_marks": len(near_polys),
        "n_shelters": len(near_shelters),
        "n_outdoor_excluded": len([x for x in all_sh if x["outdoor"]]),
        # 정직하게 기록한다 — 이 구간에서는 두 경로가 같을 수 있다.
        # 렌더러는 이 값을 보고 '우회했다'는 문구를 띄울지 말지 결정한다.
        "routes_identical": detour == 0,
    })

    return _finish({
        "id": "app-osong",
        "kind": "app",
        "city": "osong",
        "theme": "light",
        "title": "복잡한 데이터를 지금 해야 할 행동과 경로로 바꿉니다",
        "subtitle": f"2023-07-15 오송 사례 재구성 · {hour} 기준 · 재난문자는 실제 수신 문자 예시",
        "bbox": bbox,
        "layers": _layers(ds, "osong", bbox),
        "routes": [
            {"role": "risk", "coords": r_short["coords"],
             "length_m": r_short["length_m"],
             "underpasses": r_short["underpasses"]},
            {"role": "safe", "coords": r_safe["coords"],
             "length_m": r_safe["length_m"],
             "underpasses": r_safe["underpasses"]},
        ],
        "flood_polys": [p["pts"] for p in near_polys],
        "flood_meta": near_polys,
        "underpass_segs": [u["pts"] for u in ds.underpasses("osong", bbox)],
        "markers": mk,
        "shelters": near_shelters,
        "disaster_msg": msg,
        "readings": rd,
        "facts": [
            {"k": "현재 수위", "v": f"{row['wl']}m", "tone": "risk"},
            {"k": "3시간 뒤 예측", "v": f"{pred}m", "tone": "risk"},
            {"k": "안전경로", "v": f"{r_safe['length_m']}m", "tone": "safe"},
            {"k": "회피한 침수구간", "v": f"{r_short.get('flood_hit_m', 0)}m",
             "tone": "risk"},
        ],
        "sources": [f"수위 한강홍수통제소 {st['name']}({st['code']})",
                    "강수 기상청 청주 ASOS", "침수흔적도 행정안전부",
                    "통합대피소 행정안전부", "재난문자 행정안전부",
                    "도로망 OpenStreetMap"],
        "provenance": {"files": ["data/osong_2023/osong_20230715_timeline.csv",
                                 "data/osong_2023/locations.json",
                                 "data/osong_2023/shelters_osong_10km.json",
                                 "data/osong_2023/fludmarks_chungbuk.json",
                                 "data/disaster_msg/msg_flood.json",
                                 "data/routing/graph_osong.pkl"],
                       "computed": "datasource.route()"},
    })


SCENES = {
    "underpass": lambda ds: scene_underpass(ds),
    "flood-detour": lambda ds: scene_flood_detour(ds),
    "osong-predictive": lambda ds: scene_osong_predictive(ds),
    "app": lambda ds: scene_app(ds),
}


def build(name, outdir="scenes"):
    ds = DS()
    if name not in SCENES:
        raise SystemExit(f"알 수 없는 장면: {name}. 가능: {', '.join(SCENES)}")
    print(f"■ 장면 생성: {name}")
    sc = SCENES[name](ds)
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, f"{sc['id']}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(sc, f, ensure_ascii=False)
    L = sc["layers"]
    print(f"  도로 {len(L['roads'])} · 건물 {len(L['buildings'])} · "
          f"수계 {len(L['water'])} · 랜드마크 {len(L['landmarks'])}")
    print(f"  경로 {len(sc['routes'])} · 침수폴리곤 {len(sc['flood_polys'])} · "
          f"지하차도 {len(sc['underpass_segs'])} · 마커 {len(sc['markers'])}")
    if sc.get("readings"):
        r = sc["readings"]
        print(f"  실측: {r.get('station', '')} {r.get('wl', '')}m → "
              f"{r.get('wl_ahead', '')}m · 단계 {r.get('level_name', '')}")
    print(f"  커버리지 {'OK' if sc['coverage_ok'] else '문제 있음'}")
    print(f"  → {p}  ({os.path.getsize(p) // 1024}KB)")
    return p


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "list":
        print("사용 가능한 장면:")
        for k in SCENES:
            print(f"  {k}")
        sys.exit(0)
    out = a[a.index("-o") + 1] if "-o" in a else "scenes"
    for n in a:
        if n.startswith("-") or n == out:
            continue
        build(n, out)
