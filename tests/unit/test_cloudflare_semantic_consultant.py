from __future__ import annotations

from typing import Any

from debilgi.language.semantic_consultant import (
    CloudflareWorkersAIConsultant,
    ConsultationRequest,
)


class FakeResponse:
    def __init__(self, status: int, payload: object) -> None:
        self.status = status
        self.payload = payload

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def json(self, *, content_type: object = None) -> object:
        del content_type
        return self.payload


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.closed = False
        self.calls: list[dict[str, Any]] = []

    def post(self, endpoint: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"endpoint": endpoint, **kwargs})
        return self.response

    async def close(self) -> None:
        self.closed = True


def request() -> ConsultationRequest:
    return ConsultationRequest(
        prompt="Что значит слово?",
        context_messages=(),
        known_terms=("слово",),
        unknown_terms=(),
        local_relations=(),
        local_confidence=0.4,
        reason_codes=("low_confidence",),
    )


async def test_cloudflare_consultant_accepts_structured_response() -> None:
    session = FakeSession(
        FakeResponse(
            200,
            {
                "success": True,
                "result": {
                    "response": {
                        "interpretation": "Запрошено значение слова.",
                        "intent": "definition_question",
                        "facts": ["Слово является единицей языка."],
                        "confidence": 0.82,
                    }
                },
            },
        )
    )
    consultant = CloudflareWorkersAIConsultant(
        "account-id",
        "secret-token",
        model="@cf/meta/llama-3.1-8b-instruct-fast",
        instruction="Верни структурный разбор.",
    )
    consultant._session = session  # type: ignore[assignment]

    result = await consultant.consult(request())

    assert result is not None
    assert result.provider == "cloudflare-workers-ai"
    assert result.response_text is None
    assert result.intent == "definition_question"
    call = session.calls[0]
    assert call["endpoint"].endswith("/@cf/meta/llama-3.1-8b-instruct-fast")
    payload = call["json"]
    assert isinstance(payload, dict)
    assert payload["response_format"]["type"] == "json_schema"


async def test_cloudflare_consultant_marks_quota_error_retryable() -> None:
    session = FakeSession(
        FakeResponse(
            429,
            {"success": False, "errors": [{"message": "Daily allocation exceeded"}]},
        )
    )
    consultant = CloudflareWorkersAIConsultant(
        "account-id",
        "secret-token",
        model="@cf/meta/llama-3.1-8b-instruct-fast",
        instruction="Верни структурный разбор.",
    )
    consultant._session = session  # type: ignore[assignment]

    result = await consultant.consult(request())

    assert result is None
    assert consultant.last_failure_retryable

    repeated = await consultant.consult(request())

    assert repeated is None
    assert len(session.calls) == 1
