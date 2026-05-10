import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request


PREDICT_RATE_LIMIT = 30
PREDICT_RATE_WINDOW_SECONDS = 60
RATE_LIMIT_EXCEEDED_MESSAGE = (
    "Limite de requisições excedido. Tente novamente em instantes."
)

_request_timestamps = defaultdict(deque)


def get_client_ip(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host

    return "unknown"


def is_rate_limited(
    client_ip: str,
    limit: int | None = None,
    window_seconds: int | None = None,
    now: float | None = None,
) -> bool:
    limit = PREDICT_RATE_LIMIT if limit is None else limit
    window_seconds = (
        PREDICT_RATE_WINDOW_SECONDS if window_seconds is None else window_seconds
    )
    current_time = now if now is not None else time.time()
    timestamps = _request_timestamps[client_ip]
    window_start = current_time - window_seconds

    while timestamps and timestamps[0] <= window_start:
        timestamps.popleft()

    if len(timestamps) >= limit:
        return True

    timestamps.append(current_time)
    return False


def enforce_predict_rate_limit(request: Request) -> None:
    client_ip = get_client_ip(request)

    if is_rate_limited(client_ip):
        raise HTTPException(
            status_code=429,
            detail=RATE_LIMIT_EXCEEDED_MESSAGE,
        )


def clear_rate_limit_state() -> None:
    _request_timestamps.clear()
