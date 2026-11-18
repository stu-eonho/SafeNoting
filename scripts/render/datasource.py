# -*- coding: utf-8 -*-
"""
실데이터 로더 — 프로젝트 원본 파일의 스키마 차이를 여기서 전부 흡수한다.

■ 왜 이 파일이 따로 있는가
  이 저장소의 데이터는 시기별로 다른 스크립트가 만들어서 스키마가 갈린다.
  렌더러마다 분기를 넣으면 같은 버그를 세 번 고치게 된다.
  **원본을 아는 코드는 여기 하나뿐이고, 나머지는 정규화된 결과만 받는다.**

■ 실제로 밟은 함정 세 가지 (전부 여기서 처리한다)
  ① context_osong.json 은 roads/water/buildings/rail,
     context_cheongju·busan 은 buildings/landmarks/water_lines/water_areas
     → 키 이름이 아예 다르다. roads 가 없는 파일도 있다.
  ② 도로의 정답 소스는 context 가 아니라 graph_{city}.pkl 이다.
     context 로만 그리면 도시에 따라 도로가 통째로 비어 보인다.
  ③ graph_osong/busan 은 노드에 y/x 속성이 있는데,
     graph_cheongju 는 **노드 ID 자체가 (lat, lon) 튜플**이고 속성이 비어 있다.
  ④ 수계는 점 개수로 면/선을 가르면 안 된다. 미호강 중심선은 450점짜리 '선'이다.
     natural=water 태그와 폴리곤 폐합 여부로만 판정한다.

사용:
    from datasource import DS
    ds = DS()                      # 프로젝트 루트 자동 탐색
    roads = ds.roads("osong", bbox)
    ctx   = ds.context("osong")
"""
import json
import math
import os
import pickle


def _find_root(start=None):
    """data/ 와 scripts/ 가 함께 있는 디렉터리를 프로젝트 루트로 본다."""
    p = os.path.abspath(start or os.getcwd())
    for _ in range(8):
        if os.path.isdir(os.path.join(p, "data")) and os.path.isdir(os.path.join(p, "scripts")):
            return p
        nxt = os.path.dirname(p)
        if nxt == p:
            break
        p = nxt
    return os.path.abspath(start or os.getcwd())


def in_bbox(lat, lon, bbox):
    if not bbox:
        return True
    la0, la1, lo0, lo1 = bbox
    return la0 <= lat <= la1 and lo0 <= lon <= lo1


def any_in_bbox(pts, bbox):
    return any(in_bbox(p[0], p[1], bbox) for p in pts)


class DS:
    """프로젝트 데이터 접근자. 모든 반환값은 [lat, lon] 순서로 통일한다."""

    def __init__(self, root=None):
        self.root = _find_root(root)
        self.d = os.path.join(self.root, "data")
        self._cache = {}

    # ── 낮은 수준 ───────────────────────────────────────────────────
    def path(self, *parts):
        return os.path.join(self.d, *parts)

    def jload(self, *parts):
        p = self.path(*parts)
        if p in self._cache:
            return self._cache[p]
        if not os.path.exists(p):
            raise FileNotFoundError(f"데이터 없음: {p}")
        with open(p, encoding="utf-8") as f:
            v = json.load(f)
        self._cache[p] = v
        return v

    def exists(self, *parts):
        return os.path.exists(self.path(*parts))

    # ── 도로망 (graph pickle이 정답 소스) ────────────────────────────
    @staticmethod
    def node_latlon(G, n):
        """
        노드 → (lat, lon). 그래프마다 저장 방식이 다르다.
          osong/busan   : 노드 속성 y/x
          cheongju      : 노드 ID 자체가 (lat, lon) 튜플
        """
        a = G.nodes[n]
        if "y" in a and "x" in a:
            return (float(a["y"]), float(a["x"]))
        if isinstance(n, (tuple, list)) and len(n) == 2:
            try:
                return (float(n[0]), float(n[1]))
            except (TypeError, ValueError):
                return None
        return None

    def graph(self, city):
        key = f"__g_{city}"
        if key in self._cache:
            return self._cache[key]
        p = self.path("routing", f"graph_{city}.pkl")
        if not os.path.exists(p):
            raise FileNotFoundError(f"그래프 없음: {p}")
        with open(p, "rb") as f:
            G = pickle.load(f)
        self._cache[key] = G
        return G

    def roads(self, city, bbox=None, max_edges=20000):
        """
        [{"pts": [[lat,lon],[lat,lon]], "tags": {...}}, ...]

        엣지 하나가 선분 하나다. geometry 속성이 있으면 그걸 쓰고,
        없으면 양 끝 노드를 잇는다(osmnx 그래프의 일반적 형태).
        """
        G = self.graph(city)
        out = []
        for u, v, d in G.edges(data=True):
            a, b = self.node_latlon(G, u), self.node_latlon(G, v)
            if not a or not b:
                continue
            pts = [list(a), list(b)]
            if bbox and not any_in_bbox(pts, bbox):
                continue
            out.append({"pts": pts, "tags": {
                "highway": _s(d.get("highway")),
                "name": _s(d.get("name")),
                "underpass": _is_up(d),
                "flooded": str(d.get("flooded", "")).lower() == "true",
                "flood_risk": _f(d.get("_flood_risk", d.get("flood_risk"))),
                "flood_depth": _f(d.get("flood_depth")),
                "length": _f(d.get("length")),
            }})
            if len(out) >= max_edges:
                break
        return out

    def underpasses(self, city, bbox=None, names=None):
        """
        지하차도 엣지. 침수 시 하드 차단 대상이라 따로 뽑는다.

        식별 경로가 도시마다 다르다 — 셋을 순서대로 시도한다.
          ① 그래프 엣지의 underpass / _underpass 속성 (cheongju / osong·busan)
          ② 호출자가 넘긴 이름, graph_meta.json, city_cases.json 의 이름 목록
             — 속성이 하나도 없는 그래프에서만 쓴다. 이름 매칭은 같은 이름의
               지상 구간까지 통째로 잡기 때문이다.
        """
        rs = self.roads(city, bbox)
        hits = [r for r in rs if r["tags"]["underpass"]]
        if hits:
            return hits
        # 이름 후보를 순서대로 시도한다. graph_meta 는 오송 전용이라
        # 그것만 보고 끝내면 부산에서 0개가 나온다 — 실제로 그랬다.
        candidates = [set(names or [])]
        try:
            candidates.append(set(self.jload("routing", "graph_meta.json")
                                  .get("underpass_names", [])))
        except FileNotFoundError:
            pass
        try:
            candidates.append({c["underpass"] for c in self.city_cases(city)
                               if c.get("underpass")})
        except FileNotFoundError:
            pass
        for want in candidates:
            if not want:
                continue
            hit = [r for r in rs if r["tags"]["name"] in want]
            if hit:
                return hit
        return []

    # ── 경로 계산 (실제 그래프로 돌린다 — 좌표를 지어내지 않는다) ────
    def nearest_node(self, city, lat, lon):
        G = self.graph(city)
        best, bd = None, 1e18
        for n in G.nodes:
            c = self.node_latlon(G, n)
            if not c:
                continue
            d = (c[0] - lat) ** 2 + ((c[1] - lon) * 0.8) ** 2
            if d < bd:
                best, bd = n, d
        return best

    def route(self, city, origin, dest, mode="safe", flood_polys=None,
              avoid_underpass=True, alpha=8.0, risk_radius=250.0,
              flood_penalty=2.0e4):
        """
        실제 도로망 그래프에서 경로를 계산한다.

        mode="short" : 길이만 최소화 (기존 내비게이션이 하는 일)
        mode="safe"  : 길이 × (1 + alpha·침수근접위험), 지하차도는 하드 차단

        프로젝트의 실제 라우팅 규칙을 그대로 쓴다.
          그래서 결과 좌표가 전부 실제 도로 위에 놓인다.
        반환: {"coords": [[lat,lon],...], "length_m": float, "underpasses": [...]}
        """
        import networkx as nx
        G = self.graph(city)
        src = self.nearest_node(city, *origin)
        dst = self.nearest_node(city, *dest)
        if src is None or dst is None:
            return None

        # 이름 매칭은 지하차도 속성이 전혀 없는 그래프에서만 쓴다.
        # 속성이 있는데 이름까지 쓰면 같은 이름의 지상 도로 전체가 막힌다.
        has_up_attr = any(_is_up(dd) for _, _, dd in G.edges(data=True))
        up_names = set()
        if avoid_underpass and not has_up_attr:
            try:
                up_names = set(self.jload("routing", "graph_meta.json")
                               .get("underpass_names", []))
            except FileNotFoundError:
                pass

        fidx = FloodIndex(flood_polys) if flood_polys else None

        def _one(d):
            """
            ⚠ MultiDiGraph 에서 weight 콜백이 받는 d 는 엣지 속성이 아니라
              {key: attrs} 사전이다. 그걸 그대로 .get("length") 하면 None 이
              돌아와 모든 엣지 비용이 1이 되고, '최단경로'가 실제로는
              홉 수 최소 경로가 된다 — 실제로 그 버그를 밟았다.
            """
            if d and all(isinstance(v2, dict) for v2 in d.values()):
                return min(d.values(), key=lambda x: _f(x.get("length"), 1e9))
            return d

        def w(u, v, d):
            d = _one(d)
            L = _f(d.get("length"), 0.0)
            if L <= 0:
                a0, b0 = self.node_latlon(G, u), self.node_latlon(G, v)
                L = _hav(a0, b0) if (a0 and b0) else 1.0
            if mode == "short":
                return L
            name = _s(d.get("name"))
            is_up = _is_up(d) or bool(name and name in up_names)
            if is_up:
                return L * 1e6                        # 사실상 차단
            if not fidx:
                return L
            a, b = self.node_latlon(G, u), self.node_latlon(G, v)
            if not (a and b):
                return L
            # 관통은 '조금 비싼 길'이 아니라 '쓰지 않는 길'이다.
            if fidx.crosses(a, b):
                return L * flood_penalty
            r = fidx.proximity(a, b, risk_radius)
            return L * (1.0 + alpha * r)

        try:
            path = nx.shortest_path(G, src, dst, weight=w)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        coords = [list(self.node_latlon(G, n)) for n in path
                  if self.node_latlon(G, n)]
        length = 0.0
        ups = []
        flood_hits = 0
        flood_hit_m = 0.0
        for i in range(len(path) - 1):
            d = _one(G.get_edge_data(path[i], path[i + 1]))
            seg = _f(d.get("length"))
            a0, b0 = self.node_latlon(G, path[i]), self.node_latlon(G, path[i + 1])
            if seg <= 0:
                seg = _hav(a0, b0)
            length += seg
            nm = _s(d.get("name"))
            if _is_up(d) or (nm and nm in up_names):
                ups.append(nm or "(무명 지하차도)")
            # 결과 경로가 실제로 침수구역을 밟는지 사후 확인한다.
            # 우회가 불가능한 지형이면 여기서 0 이 아닌 값이 나오고,
            # 장면은 그 사실을 숨기지 않고 기록한다.
            if fidx and a0 and b0 and fidx.crosses(a0, b0):
                flood_hits += 1
                flood_hit_m += seg
        return {"coords": coords, "length_m": round(length),
                "underpasses": sorted(set(ups)),
                "flood_hits": flood_hits,
                "flood_hit_m": round(flood_hit_m)}

    def find_underpass_case(self, city, bbox=None, want=(5, 90), radius=700,
                            flood_polys=None, alpha=8.0, limit=12):
        """
        '최단경로는 지하차도를 지나고, 안전경로는 우회한다'가 성립하는
        실제 출발-도착 쌍을 그래프에서 찾는다.

        좌표를 지어내지 않고 실제 도로 노드만 쓴다. 여러 후보를 계산해
        우회 대가가 want 범위(%)에 들어오는 것 중 가장 작은 것을 고른다
        — 우회가 +5m 면 그림에서 안 보이고, +2.6km 면 메시지가 반대가 된다.

        반환: {"origin","dest","short","safe","underpass"} 또는 None
        """
        import networkx as nx
        G = self.graph(city)
        ups = self.underpasses(city, bbox)
        if not ups:
            return None
        seen, out = set(), []
        for up in ups[:limit * 4]:
            nm = up["tags"]["name"] or "(무명 지하차도)"
            key = (nm, round(up["pts"][0][0], 4))
            if key in seen:
                continue
            seen.add(key)
            a, b = up["pts"][0], up["pts"][1]
            o = self._far_node(G, a, b, radius)
            d = self._far_node(G, b, a, radius)
            if not o or not d:
                continue
            r1 = self.route(city, o, d, "short", avoid_underpass=False)
            if not r1 or nm not in r1["underpasses"]:
                continue
            r2 = self.route(city, o, d, "safe", flood_polys=flood_polys,
                            alpha=alpha)
            if not r2 or r2["underpasses"]:
                continue
            pct = 100.0 * (r2["length_m"] - r1["length_m"]) / max(1, r1["length_m"])
            if not (want[0] <= pct <= want[1]):
                continue
            out.append({"origin": o, "dest": d, "short": r1, "safe": r2,
                        "underpass": nm, "detour_pct": pct})
            if len(out) >= limit:
                break
        return min(out, key=lambda x: x["detour_pct"]) if out else None

    def _far_node(self, G, near, away, radius):
        """near 에서 radius 안이면서 away 로부터 가장 먼 실제 도로 노드."""
        best, bd = None, -1.0
        for n in G.nodes:
            c = self.node_latlon(G, n)
            if not c:
                continue
            if _hav(c, near) > radius:
                continue
            d = _hav(c, away)
            if d > bd:
                best, bd = c, d
        return list(best) if best else None

    # ── 맥락(건물·수계·랜드마크) ────────────────────────────────────
    def context(self, city, bbox=None):
        """
        두 가지 스키마를 하나로 정규화한다.
        반환: {"buildings":[...], "water":[...], "rail":[...], "landmarks":[...]}
              water 항목은 {"pts":..., "area": bool} 로 면/선이 이미 판정돼 있다.
        """
        fn = f"context_{city}.json"
        if not self.exists("routing", fn):
            return {"buildings": [], "water": [], "rail": [], "landmarks": []}
        c = self.jload("routing", fn)

        def clip(items):
            if not bbox:
                return items
            return [it for it in items if any_in_bbox(it["pts"], bbox)]

        buildings = [{"pts": p} for p in map(_pts, c.get("buildings", [])) if p]
        rail = [{"pts": p} for p in map(_pts, c.get("rail", [])) if p]

        water = []
        # 스키마 A — context_osong: water 하나에 면/선이 섞여 있다
        for w in c.get("water", []):
            pts = _pts(w)
            if not pts:
                continue
            tags = _tags(w)
            area = tags.get("natural") == "water" or (len(pts) > 3 and pts[0] == pts[-1])
            water.append({"pts": pts, "area": area,
                          "name": tags.get("name", ""),
                          "big": tags.get("waterway") == "river"})
        # 스키마 B — context_cheongju / busan: 면과 선이 키로 분리돼 있고
        #            원소가 dict 가 아니라 좌표 배열 그 자체다
        for w in c.get("water_areas", []):
            pts = _pts(w)
            if pts:
                water.append({"pts": pts, "area": True,
                              "name": _tags(w).get("name", ""), "big": True})
        for w in c.get("water_lines", []):
            pts = _pts(w)
            if pts:
                tags = _tags(w)
                water.append({"pts": pts, "area": False,
                              "name": tags.get("name", ""),
                              "big": tags.get("waterway", "river") == "river"})

        landmarks = []
        for m in c.get("landmarks", []):
            if "lat" in m and ("lon" in m or "lng" in m):
                landmarks.append({"lat": m["lat"], "lon": m.get("lon", m.get("lng")),
                                  "name": m.get("name", ""),
                                  "kind": m.get("kind", m.get("type", ""))})

        return {"buildings": clip(buildings), "water": clip(water),
                "rail": clip(rail),
                "landmarks": [m for m in landmarks
                              if in_bbox(m["lat"], m["lon"], bbox)] if bbox else landmarks}

    # ── 도메인 데이터 ───────────────────────────────────────────────
    def osong_timeline(self):
        """참사 당일 실측. [{"time","rain","rain3h","wl","state"}, ...]"""
        import csv
        p = self.path("osong_2023", "osong_20230715_timeline.csv")
        rows = []
        with open(p, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "time": r["시각"],
                    "rain": _f(r["강수량_mm"]),
                    "rain3h": _f(r["3시간누적_mm"]),
                    "wl": _f(r["미호천수위_m"]),
                    "state": r.get("상태", "").strip(),
                })
        return rows

    def locations(self):
        return self.jload("osong_2023", "locations.json")

    def flood_route(self):
        """{"polys":[...], "shelters_safe":[...], "shelters_unsafe":[...], "cases":[...]}"""
        return self.jload("routing", "flood_route.json")

    def flood_polys(self, bbox=None):
        """
        침수흔적 폴리곤을 정규화한다.
        원본은 {"ring": [[lat,lon],...], "depth": 0.78, "year": "2017"} 형태다.
        ring 을 벗겨내지 않고 좌표로 착각하면 문자열 비교 오류가 난다.

        반환: [{"pts": [[lat,lon],...], "depth": float, "year": str}, ...]
        """
        out = []
        for p in self.flood_route().get("polys", []):
            ring = p.get("ring") if isinstance(p, dict) else _pts(p)
            if not ring:
                continue
            if bbox and not any_in_bbox(ring, bbox):
                continue
            out.append({"pts": ring,
                        "depth": _f(p.get("depth")) if isinstance(p, dict) else 0.0,
                        "year": str(p.get("year", "")) if isinstance(p, dict) else ""})
        return out

    def flood_marks(self, bbox=None, region="chungbuk", limit=None):
        """
        원본 침수흔적도(행정안전부)를 직접 읽는다.

        flood_route.json 의 폴리곤은 청주 도심 범위만 담고 있어서,
        오송 장면에서는 0개가 나온다. 충북 전체 원본은 따로 있고
        GEOM 이 EPSG:3857 문자열이라 파싱과 좌표 변환이 필요하다.

        반환: [{"pts": [[lat,lon],...], "depth": float, "year": str, "cause": str}]
        """
        import re
        rows = self.jload("osong_2023", f"fludmarks_{region}.json")
        if isinstance(rows, dict):
            rows = rows.get("body") or rows.get("items") or next(
                (v for v in rows.values() if isinstance(v, list)), [])
        out = []
        for f in rows:
            nums = re.findall(r"(\d{6,9}\.?\d*)\s+(\d{6,9}\.?\d*)", f.get("GEOM") or "")
            if len(nums) < 3:
                continue
            ring = [_m3857_to_wgs(float(a), float(b)) for a, b in nums]
            if bbox and not any_in_bbox(ring, bbox):
                continue
            out.append({"pts": ring,
                        "depth": _f(f.get("FLDN_DOWA")),
                        "year": str(f.get("FLDN_YR", "")),
                        "cause": f.get("FLDN_CS_DTL_NM", "")})
            if limit and len(out) >= limit:
                break
        return out

    def osong_shelters(self, bbox=None, limit=None):
        """
        오송 반경 10km 통합대피소 385곳. 필드명이 행안부 원본 그대로라
        (REARE_NM / LAT / LOT) 여기서 정규화한다.
        """
        # shelters_flood_joined.json = shelters_osong_10km.json + 침수흔적 거리 필드
        rows = self.jload("osong_2023", "shelters_flood_joined.json")
        if isinstance(rows, dict):
            rows = next((v for v in rows.values() if isinstance(v, list)), [])
        out = []
        for r in rows:
            lat, lon = _f(r.get("LAT")), _f(r.get("LOT"))
            if not lat or not lon or (bbox and not in_bbox(lat, lon, bbox)):
                continue
            kind = r.get("SHLT_SE_NM", "")
            out.append({"name": r.get("REARE_NM", ""), "lat": lat, "lon": lon,
                        "kind": kind, "addr": r.get("RONA_DADDR", ""),
                        "dist_km": _f(r.get("_dist_km")),
                        "flood_km": r.get("_nearest_flood_km"),
                        "flood_inside": bool(r.get("_flood_inside")),
                        # 야외 시설은 폭우 중 무용지물이다 — 판정 근거를 보존한다
                        "outdoor": "옥외" in kind or "야외" in kind})
            if limit and len(out) >= limit:
                break
        return sorted(out, key=lambda x: x["dist_km"])

    def shelters(self, bbox=None):
        """대피소를 안전/부적합으로 나눠 반환. 판정 근거 필드를 그대로 보존한다."""
        fr = self.flood_route()

        def pick(key, safe):
            out = []
            for s in fr.get(key, []):
                if bbox and not in_bbox(s["lat"], s["lon"], bbox):
                    continue
                out.append({"name": s.get("name", ""), "lat": s["lat"], "lon": s["lon"],
                            "kind": s.get("kind", ""), "elev": s.get("elev"),
                            "flood_dist": s.get("flood_dist"),
                            "flood_depth": s.get("flood_depth"),
                            "outdoor": s.get("outdoor"), "safe": safe})
            return out
        return pick("shelters_safe", True), pick("shelters_unsafe", False)

    def city_cases(self, city=None):
        cs = self.jload("routing", "city_cases.json")
        return [c for c in cs if c["city"] == city] if city else cs

    def route_compare(self):
        return self.jload("routing", "route_compare.json")

    def shelters_ranked(self):
        return self.jload("routing", "shelters_ranked.json")

    def disaster_msgs(self, keyword=None, region=None, limit=None):
        """실제 재난문자 원문. Before 화면에 그대로 인용한다."""
        rows = self.jload("disaster_msg", "msg_flood.json")
        out = []
        for r in rows:
            if keyword and keyword not in r.get("MSG_CN", ""):
                continue
            if region and region not in r.get("RCPTN_RGN_NM", ""):
                continue
            out.append({"text": r.get("MSG_CN", ""),
                        "region": r.get("RCPTN_RGN_NM", "").strip(),
                        "at": r.get("CRT_DT", ""),
                        "step": r.get("EMRG_STEP_NM", ""),
                        "kind": r.get("DST_SE_NM", "")})
        return out[:limit] if limit else out


# ════════════════════════════════════════════════════════════════════
# 침수 폴리곤 기하 — "가까운가"가 아니라 "통과하는가"를 본다
# ════════════════════════════════════════════════════════════════════

def _poly_bbox(pts):
    ys = [q[0] for q in pts]
    xs = [q[1] for q in pts]
    return (min(ys), max(ys), min(xs), max(xs))


def _pip(y, x, pts):
    """점이 폴리곤 내부인가 (ray casting). pts = [[lat,lon],...]"""
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        yi, xi = pts[i][0], pts[i][1]
        yj, xj = pts[j][0], pts[j][1]
        if (yi > y) != (yj > y):
            xint = (xj - xi) * (y - yi) / ((yj - yi) or 1e-18) + xi
            if x < xint:
                inside = not inside
        j = i
    return inside


def _pt_seg_d(py, px, ay, ax, by, bx):
    """점-선분 거리(m). lat/lon 을 국소 평면으로 근사한다."""
    kx = 89000.0
    ky = 111000.0
    vy, vx = (by - ay) * ky, (bx - ax) * kx
    wy, wx = (py - ay) * ky, (px - ax) * kx
    L2 = vy * vy + vx * vx
    t = 0.0 if L2 <= 0 else max(0.0, min(1.0, (wy * vy + wx * vx) / L2))
    dy, dx = wy - t * vy, wx - t * vx
    return (dy * dy + dx * dx) ** 0.5


def _poly_dist(y, x, pts):
    """점에서 폴리곤 경계까지 최단거리(m). 내부면 0."""
    if _pip(y, x, pts):
        return 0.0
    best = 1e18
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        d = _pt_seg_d(y, x, a[0], a[1], b[0], b[1])
        if d < best:
            best = d
    return best


class FloodIndex:
    """
    침수 폴리곤 공간 인덱스.

    ⚠ 중심점 거리로 위험을 재면 폴리곤을 관통하는 경로가 '조금 위험한 길'로
      평가된다. 실제로 그 버그로 오송 안전경로가 침수구역을 뚫고 지나갔다.
      그래서 여기서는 (1) 실제 내부 판정, (2) 엣지를 여러 점으로 샘플링,
      (3) 경계까지의 거리로 근접 위험을 잰다.
    """

    CELL = 0.004   # 약 400m 격자

    def __init__(self, polys):
        self.polys = []
        self.grid = {}
        for poly in polys or []:
            pts = poly["pts"] if isinstance(poly, dict) else poly
            if not pts or len(pts) < 3:
                continue
            bb = _poly_bbox(pts)
            idx = len(self.polys)
            self.polys.append((pts, bb))
            y0, y1, x0, x1 = bb
            for cy in range(int(y0 / self.CELL), int(y1 / self.CELL) + 1):
                for cx in range(int(x0 / self.CELL), int(x1 / self.CELL) + 1):
                    self.grid.setdefault((cy, cx), []).append(idx)

    def __bool__(self):
        return bool(self.polys)

    def _near(self, y, x, pad_cells=1):
        cy, cx = int(y / self.CELL), int(x / self.CELL)
        out = set()
        for dy in range(-pad_cells, pad_cells + 1):
            for dx in range(-pad_cells, pad_cells + 1):
                out.update(self.grid.get((cy + dy, cx + dx), ()))
        return out

    def inside(self, y, x):
        for i in self._near(y, x, 0):
            pts, bb = self.polys[i]
            if bb[0] <= y <= bb[1] and bb[2] <= x <= bb[3] and _pip(y, x, pts):
                return True
        return False

    def dist(self, y, x, cap):
        """경계까지 최단거리(m), cap 이상이면 cap."""
        best = cap
        for i in self._near(y, x, 1):
            pts, _ = self.polys[i]
            d = _poly_dist(y, x, pts)
            if d < best:
                best = d
                if best <= 0:
                    return 0.0
        return best

    def seg_samples(self, a, b, step_m=25.0, max_n=12):
        """엣지를 따라 샘플점을 만든다. 중점 하나만 보면 관통을 놓친다."""
        L = _hav(a, b)
        n = max(2, min(max_n, int(L / step_m) + 1))
        return [((a[0] + (b[0] - a[0]) * i / (n - 1)),
                 (a[1] + (b[1] - a[1]) * i / (n - 1))) for i in range(n)]

    def crosses(self, a, b):
        """선분이 침수 폴리곤을 통과하는가."""
        for y, x in self.seg_samples(a, b):
            if self.inside(y, x):
                return True
        return False

    def proximity(self, a, b, radius):
        """0(멀다)~1(경계에 붙었다). 내부면 1."""
        best = radius
        for y, x in self.seg_samples(a, b, step_m=60.0, max_n=5):
            d = self.dist(y, x, radius)
            if d < best:
                best = d
        return max(0.0, 1.0 - best / radius)


def _is_up(d):
    """지하차도 엣지 판정. 그래프마다 속성명이 다르다 (청주 underpass, 오송·부산 _underpass)."""
    return (str(d.get("underpass", "")).lower() == "true"
            or str(d.get("_underpass", "")).lower() == "true")


def _s(v):
    """None 과 문자열 "None" 을 똑같이 빈 문자열로 만든다.
    이걸 안 하면 이름 없는 엣지가 "None" 이라는 지하차도로 둔갑한다."""
    t = "" if v is None else str(v).strip()
    return "" if t.lower() in ("none", "nan", "null") else t


def _hav(a, b):
    """두 (lat,lon) 사이 거리(m). length 속성이 없는 그래프를 위한 보완."""
    if not a or not b:
        return 1.0
    R = 6371000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def _m3857_to_wgs(x, y):
    """EPSG:3857 미터 → (lat, lon). 침수흔적도 GEOM 이 이 좌표계로 온다."""
    R = 6378137.0
    lon = x / R * 180.0 / math.pi
    lat = (2.0 * math.atan(math.exp(y / R)) - math.pi / 2.0) * 180.0 / math.pi
    return [lat, lon]


def _pts(item):
    """
    좌표 배열을 꺼낸다. 원소가 두 형태로 존재한다.
      {"pts": [[lat,lon],...], "tags": {...}}   ← context_osong
      [[lat,lon], [lat,lon], ...]               ← context_cheongju / busan
    """
    if isinstance(item, dict):
        p = item.get("pts")
        return p if isinstance(p, list) and p else None
    if isinstance(item, list) and item and isinstance(item[0], (list, tuple)) and len(item[0]) >= 2:
        return item
    return None


def _tags(item):
    return (item.get("tags") or {}) if isinstance(item, dict) else {}


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ════════════════════════════════════════════════════════════════════
# 커버리지 검사 — 조용히 빈 지도가 나오는 사고를 막는다
# ════════════════════════════════════════════════════════════════════

def check_coverage(bbox, layers, min_counts=None):
    """
    장면 bbox 안에 실제로 그릴 것이 있는지 확인한다.

    이 검사가 없으면 배경이 통째로 빈 지도가 조용히 만들어진다.
    (실제로 오송 사례 bbox가 lon 127.289~127.301 인데
     context_osong 은 127.301 부터라 도로가 한 개도 안 걸린 적이 있다)

    layers = {"roads": [...], "buildings": [...], ...}
    반환: (ok: bool, problems: [str])
    """
    min_counts = min_counts or {"roads": 20}
    problems = []
    for k, want in min_counts.items():
        got = len(layers.get(k) or [])
        if got < want:
            problems.append(
                f"{k} 레이어가 {got}개뿐이다(최소 {want} 기대). "
                f"장면 bbox lat {bbox[0]:.4f}~{bbox[1]:.4f} / "
                f"lon {bbox[2]:.4f}~{bbox[3]:.4f} 와 원본 데이터 범위가 "
                f"어긋났을 수 있다 — 다른 city 의 데이터를 쓰고 있는지 확인하라.")
    return (not problems), problems


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ds = DS()
    print(f"프로젝트 루트: {ds.root}\n")
    for city in ("osong", "cheongju", "busan_choryang"):
        try:
            r = ds.roads(city)
            c = ds.context(city)
            la = [p[0] for e in r for p in e["pts"]]
            lo = [p[1] for e in r for p in e["pts"]]
            print(f"{city:16s} 도로 {len(r):5d}  건물 {len(c['buildings']):5d}  "
                  f"수계 {len(c['water']):3d}  랜드마크 {len(c['landmarks']):3d}")
            print(f"{'':16s} bbox lat {min(la):.4f}~{max(la):.4f} "
                  f"lon {min(lo):.4f}~{max(lo):.4f}")
            print(f"{'':16s} 지하차도 엣지 {len(ds.underpasses(city))}")
        except Exception as e:
            print(f"{city:16s} 실패: {e}")
        print()
