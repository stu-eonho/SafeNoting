# -*- coding: utf-8 -*-
"""
API 키 로더 — 환경변수를 먼저 보고, 없으면 저장소 루트의 .env를 읽는다.

키는 코드에 직접 쓰지 않는다. .env.example을 복사해 .env를 만들고 값을 채운다.
    cp .env.example .env
"""
import os

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_PATH = os.path.join(_BASE, ".env")


def _read_dotenv():
    values = {}
    if not os.path.exists(_ENV_PATH):
        return values
    for line in open(_ENV_PATH, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.split(" #")[0].strip().strip('"').strip("'")
    return values


def get_key(name):
    """name에 해당하는 키를 돌려준다. 어디에도 없으면 안내 메시지와 함께 종료한다."""
    value = os.environ.get(name) or _read_dotenv().get(name)
    if not value:
        raise SystemExit(f"{name} 가 설정되지 않았습니다. .env.example 을 참고해 .env 에 추가하세요.")
    return value
