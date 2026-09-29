"""Keep calls to a real LLM provider within its requests-per-minute limit."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

from app.core.logging import get_logger
from app.llm.base import LLMClient, LLMProviderError, LLMRequest, LLMResponse

WINDOW_S = 60.0
MAX_RETRY_DELAY_S = 60.0
TOO_MANY_REQUESTS = 429

logger = get_logger(__name__)


class ThrottledLLMClient(LLMClient):
    """Wraps another client and paces it.

    1. At most ``max_rpm`` calls start in any 60-second window; an extra call waits for a free slot.
    2. If the provider still answers "too many requests" (HTTP 429), wait the delay it asked for
       (or one pacing interval), capped at 60 s, and try again, at most ``max_rate_limit_retries``
       times.

    Time spent waiting is added to ``LLMResponse.wait_ms`` so traces show it. ``clock`` and
    ``sleep`` can be replaced in tests.
    """

    def __init__(
        self,
        inner: LLMClient,
        *,
        max_rpm: int,
        max_rate_limit_retries: int = 2,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_rpm < 1:
            raise ValueError("max_rpm must be at least 1")
        self._inner = inner
        self._max_rpm = max_rpm
        self._max_rate_limit_retries = max_rate_limit_retries
        self._clock = clock
        self._sleep = sleep
        self._started: deque[float] = deque()

    @property
    def model_name(self) -> str:
        """The wrapped client's model."""
        return self._inner.model_name

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Wait for a free slot, call the wrapped client, and retry politely when rate limited."""
        waited_s = 0.0
        retries = 0
        while True:
            waited_s += self._wait_for_slot()
            try:
                response = self._inner.complete(request)
            except LLMProviderError as exc:
                if exc.status_code != TOO_MANY_REQUESTS or retries >= self._max_rate_limit_retries:
                    raise
                retries += 1
                delay = min(exc.retry_after_s or WINDOW_S / self._max_rpm, MAX_RETRY_DELAY_S)
                logger.warning("llm_rate_limited", retry_in_s=round(delay, 3), retry=retries)
                self._sleep(delay)
                waited_s += delay
                continue
            return response.model_copy(update={"wait_ms": response.wait_ms + waited_s * 1000})

    def _wait_for_slot(self) -> float:
        """Block until fewer than ``max_rpm`` calls started in the last minute; return the wait."""
        now = self._clock()
        self._forget_before(now - WINDOW_S)
        delay = 0.0
        if len(self._started) >= self._max_rpm:
            delay = self._started[0] + WINDOW_S - now
            self._sleep(delay)
            now = self._clock()
            self._forget_before(now - WINDOW_S)
        self._started.append(now)
        return delay

    def _forget_before(self, cutoff: float) -> None:
        while self._started and self._started[0] <= cutoff:
            self._started.popleft()
