import asyncio
import time
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest

from debilgi.language.russian_reference import (
    WiktionaryClient,
    WiktionaryUnavailable,
    _retry_after_seconds,
)


class Response:
    def __init__(self, status=200, payload=None, headers=None):
        self.status = status
        self.payload = payload
        self.headers = headers or {}

    async def __aenter__(self):
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        assert self.status == 200

    async def json(self, **kwargs):
        return self.payload


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        return next(self.responses)


def client_with(*responses):
    client = WiktionaryClient(user_agent="TransportTests/1.0")
    session = Session(responses)
    client._session = session
    return client, session


async def test_policy_denial_stops_all_concurrent_words_and_searches():
    client, session = client_with(Response(403))
    results = await asyncio.gather(
        client.lookup("ручка"),
        client.lookup("словарь"),
        client._search_candidate("слово", ("слово",)),
        return_exceptions=True,
    )
    assert all(isinstance(item, WiktionaryUnavailable) for item in results)
    assert all(item.status == 403 for item in results)
    assert session.calls == 1
    assert client._retry_at == float("inf")


@pytest.mark.parametrize(
    "response",
    [
        Response(429, headers={"Retry-After": "120"}),
        Response(503, headers={"Retry-After": "120"}),
        Response(payload={"error": {"code": "maxlag"}}),
        Response(payload={"error": {"code": "ratelimited"}}),
        Response(payload={}),
        Response(payload=[]),
    ],
)
async def test_operational_errors_do_not_trigger_missing_word_fallback(response):
    client, session = client_with(response)
    with pytest.raises(WiktionaryUnavailable):
        await client.lookup("слово")
    with pytest.raises(WiktionaryUnavailable):
        await client.lookup("ручка")
    assert session.calls == 1


async def test_retry_after_expires_and_successful_dictionary_response_is_parsed():
    client, session = client_with(
        Response(429),
        Response(
            payload={
                "parse": {
                    "wikitext": (
                        "= {{-ru-}} =\n== {{з|(существительное)}} ==\n"
                        "==== Значение ====\n# инструмент для письма\n"
                    )
                }
            }
        ),
    )
    with pytest.raises(WiktionaryUnavailable):
        await client.lookup("ручка")
    client._retry_at = client._next_request_at = 0
    result = await client.lookup("ручка")
    assert result.meanings == ("инструмент для письма",)
    assert session.calls == 2
    assert client._failure is None


async def test_only_missingtitle_is_a_missing_page():
    client, session = client_with(Response(payload={"error": {"code": "missingtitle"}}))
    assert await client._lookup_page("неслово") is None
    assert session.calls == 1


def test_retry_after_supports_seconds_dates_and_invalid_headers():
    assert _retry_after_seconds("120") == 120
    assert _retry_after_seconds("garbage") == 60
    assert _retry_after_seconds("-2") == 60
    future = format_datetime(datetime.now(UTC) + timedelta(seconds=120))
    assert 118 <= _retry_after_seconds(future) <= 120


async def test_successful_requests_are_serialized_and_spaced_across_workers():
    starts = []

    class TimedSession(Session):
        def get(self, *args, **kwargs):
            starts.append(time.monotonic())
            return super().get(*args, **kwargs)

    client = WiktionaryClient(user_agent="TransportTests/1.0")
    client._session = TimedSession([
        Response(payload={"error": {"code": "missingtitle"}}),
        Response(payload={"error": {"code": "missingtitle"}}),
    ])
    assert await asyncio.gather(
        client._lookup_page("ручка"), client._lookup_page("словарь")
    ) == [None, None]
    assert starts[1] - starts[0] >= 1.0
