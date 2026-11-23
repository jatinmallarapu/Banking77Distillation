"""Smoke-test the actual container health and classification response."""

import time

import httpx

for _ in range(30):
    try:
        health = httpx.get("http://127.0.0.1:8000/health", timeout=3)
        health.raise_for_status()
        assert health.json()["system"] == "tfidf"
        response = httpx.post(
            "http://127.0.0.1:8000/classify", json={"text": "Where is my new card?"}, timeout=3
        )
        response.raise_for_status()
        assert 0 <= response.json()["confidence"] <= 1
        print("Container baseline health and classification verified")
        break
    except httpx.HTTPError:
        time.sleep(1)
else:
    raise RuntimeError("Container readiness deadline exceeded")
