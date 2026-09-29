"""Tests for the requests-per-minute throttle, with a fake clock (no real waiting)."""

from __future__ import annotations

import pytest

from app.llm.base import ChatMessage, LLMClient, LLMProviderError, LLMRequest, LLMResponse, Role
from app.llm.throttle import ThrottledLLMClient

REQUEST = LLMRequest(messages=[ChatMessage(role=Role.USER, content="hi")])


class FakeClock:
    """A clock that only moves when the code under test sleeps (or the test advances it)."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class ScriptedClient(LLMClient):
    """Returns (or raises) pre-set outcomes in order."""

    def __init__(self, *outcomes: LLMResponse | Exception) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "scripted"

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


OK = LLMResponse(content="{}", model="scripted")


def _rate_limited(retry_after_s: float | None = None) -> LLMProviderError:
    return LLMProviderError("RateLimitError", status_code=429, retry_after_s=retry_after_s)


def _throttled(inner: LLMClient, clock: FakeClock, max_rpm: int = 2) -> ThrottledLLMClient:
    return ThrottledLLMClient(inner, max_rpm=max_rpm, clock=clock, sleep=clock.sleep)


def test_calls_under_the_limit_do_not_wait() -> None:
    clock = FakeClock()
    client = _throttled(ScriptedClient(OK), clock)

    responses = [client.complete(REQUEST) for _ in range(2)]

    assert clock.sleeps == []
    assert [r.wait_ms for r in responses] == [0.0, 0.0]
    assert client.model_name == "scripted"


def test_extra_call_waits_until_the_oldest_leaves_the_window() -> None:
    clock = FakeClock()
    client = _throttled(ScriptedClient(OK), clock)

    client.complete(REQUEST)
    clock.now += 10
    client.complete(REQUEST)
    third = client.complete(REQUEST)

    assert clock.sleeps == [50.0]
    assert third.wait_ms == 50_000.0


def test_window_slides_so_old_calls_stop_counting() -> None:
    clock = FakeClock()
    client = _throttled(ScriptedClient(OK), clock)

    client.complete(REQUEST)
    clock.now += 30
    client.complete(REQUEST)
    clock.now += 31
    client.complete(REQUEST)

    assert clock.sleeps == []


def test_rate_limit_error_waits_the_requested_delay_and_retries() -> None:
    clock = FakeClock()
    inner = ScriptedClient(_rate_limited(retry_after_s=7.5), OK)

    response = _throttled(inner, clock).complete(REQUEST)

    assert inner.calls == 2
    assert clock.sleeps == [7.5]
    assert response.wait_ms == 7_500.0


def test_rate_limit_without_a_delay_waits_one_pacing_interval() -> None:
    clock = FakeClock()
    inner = ScriptedClient(_rate_limited(), OK)

    _throttled(inner, clock, max_rpm=5).complete(REQUEST)

    assert clock.sleeps == [12.0]


def test_requested_delay_is_capped() -> None:
    clock = FakeClock()

    _throttled(ScriptedClient(_rate_limited(retry_after_s=500), OK), clock).complete(REQUEST)

    assert clock.sleeps == [60.0]


def test_gives_up_after_the_allowed_rate_limit_retries() -> None:
    clock = FakeClock()
    inner = ScriptedClient(_rate_limited(retry_after_s=1))

    with pytest.raises(LLMProviderError):
        ThrottledLLMClient(
            inner, max_rpm=100, max_rate_limit_retries=2, clock=clock, sleep=clock.sleep
        ).complete(REQUEST)

    assert inner.calls == 3


def test_other_provider_errors_are_not_retried() -> None:
    clock = FakeClock()
    inner = ScriptedClient(LLMProviderError("AuthenticationError", status_code=401), OK)

    with pytest.raises(LLMProviderError, match="AuthenticationError"):
        _throttled(inner, clock).complete(REQUEST)

    assert inner.calls == 1


def test_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ThrottledLLMClient(ScriptedClient(OK), max_rpm=0)
