"""Validate the pinned vLLM choice request without loading weights or calling a model."""

import json

import httpx

from b77.teacher import Teacher


def test_choice_contract_and_query_redaction():
    teacher = Teacher()

    def handler(request):
        payload = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert len(payload["guided_choice"]) == 77 and payload["temperature"] == 0.7
        assert "jane@example.com" not in request.content.decode()
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": teacher.labels[0]}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        )

    teacher.client.close()
    teacher.client = httpx.Client(
        base_url="http://local.test/v1", transport=httpx.MockTransport(handler)
    )
    label, usage = teacher.label("Email jane@example.com about my card")
    teacher.client.close()
    assert label == teacher.labels[0] and usage["prompt_tokens"] == 10
