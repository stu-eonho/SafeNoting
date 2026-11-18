# -*- coding: utf-8 -*-
"""
오송 일대 지도 맥락 수집 — 도로·하천·건물

폴리곤만 떠 있는 그림은 지도로 읽히지 않는다.
도로가 있어야 '어디로 갈 수 있는가'가 보이고, 하천이 있어야 물이 어디서 오는지 보인다.

osmnx는 이 환경에서 Overpass 연결에 실패한다. curl 직접 POST로 우회한다.

사용:
    python scripts/fetch_osong_context.py
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts"))
from flood_route import overpass

BBOX = (36.600, 36.650, 127.310, 127.385)
OUT = os.path.join(BASE, "data", "routing", "context_osong.json")


def ways(d):
    out = []
    for el in d.get("elements", []):
        g = el.get("geometry")
        if g and len(g) >= 2:
            out.append(dict(pts=[(p["lat"], p["lon"]) for p in g],
                            tags=el.get("tags", {})))
    return out


def main():
    s, n, w, e = BBOX
    box = f"{s},{w},{n},{e}"
    res = {}

    print("■ 도로망")
    d = overpass(f'[out:json][timeout:180];way({box})'
                 f'["highway"~"motorway|trunk|primary|secondary|tertiary|'
                 f'unclassified|residential|living_street"];out geom tags;')
    res["roads"] = ways(d) if d else []
    print(f"  {len(res['roads'])}개")

    print("■ 하천·수계")
    d = overpass(f'[out:json][timeout:180];('
                 f'way({box})["waterway"~"river|stream"];'
                 f'way({box})["natural"="water"];'
                 f'relation({box})["natural"="water"];);out geom tags;')
    res["water"] = ways(d) if d else []
    print(f"  {len(res['water'])}개")

    print("■ 건물")
    d = overpass(f'[out:json][timeout:180];way({box})["building"];out geom tags;')
    res["buildings"] = ways(d) if d else []
    print(f"  {len(res['buildings'])}개")

    print("■ 철도 (오송역)")
    d = overpass(f'[out:json][timeout:180];('
                 f'way({box})["railway"="rail"];'
                 f'node({box})["railway"="station"];);out geom tags;')
    res["rail"] = ways(d) if d else []
    res["rail_nodes"] = [dict(lat=el["lat"], lon=el["lon"], tags=el.get("tags", {}))
                         for el in (d or {}).get("elements", []) if el.get("type") == "node"]
    print(f"  선로 {len(res['rail'])}개 · 역 {len(res['rail_nodes'])}개")

    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\n  → {OUT}")


if __name__ == "__main__":
    main()
