"""Reuse the inference-toolkit load pattern with this API's bounded classification contract."""

from locust import HttpUser, between, task


class IntentUser(HttpUser):
    wait_time = between(0.2, 1)

    @task
    def classify(self):
        with self.client.post(
            "/classify", json={"text": "Where is my new debit card?"}, catch_response=True
        ) as response:
            if (
                response.status_code == 200
                and {"label", "confidence", "needs_human_review", "latency_ms"}
                <= response.json().keys()
            ):
                response.success()
            else:
                response.failure("Classification contract failed")
