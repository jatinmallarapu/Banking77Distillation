"""Validate the pinned vLLM choice request without loading weights or calling a model."""

import json

import httpx

from b77.teacher import Teacher


def test_local_teacher_omits_empty_authorization(monkeypatch):
    monkeypatch.delenv("TEACHER_API_KEY", raising=False)
    teacher = Teacher()
    try:
        assert "authorization" not in teacher.client.headers
    finally:
        teacher.client.close()


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


def test_parallel_filter_commits_in_order_and_resumes(tmp_path, monkeypatch):
    from b77 import teacher as module
    from b77.data import read_rows, write_rows

    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "data/prepared"
    folder.mkdir(parents=True)
    (tmp_path / "data/labels.json").write_text(json.dumps(["card_arrival"]))
    train = [{"id": f"train-{i}", "text": f"source {i}", "label": 0} for i in range(3)]
    write_rows(folder / "train.jsonl", train)
    write_rows(folder / "validation.jsonl", [{"id": "val", "text": "validation query", "label": 0}])
    write_rows(folder / "test.jsonl", [{"id": "test", "text": "test query", "label": 0}])
    calls = []

    class FakeTeacher:
        config = {"revision": "fixture"}
        labels = ["card_arrival"]

        def __init__(self):
            self.client = httpx.Client()

        def paraphrases(self, text, intent=None):
            calls.append(text)
            return [text + " paraphrase", "shared paraphrase", "validation query"]

        def label(self, text, temperature, seed, examples=None):
            assert examples is not None
            assert all(example["text"] + " paraphrase" != text for example in examples)
            return "card_arrival", {}

    monkeypatch.setattr(module, "Teacher", FakeTeacher)
    result = module.distill(workers=2)
    rows = read_rows(tmp_path / "data/distilled.jsonl")
    assert result["completed_sources"] == [row["id"] for row in train]
    assert result["proposed"] == 9 and result["unique_kept"] == 4
    assert [row["source_id"] for row in rows] == ["train-0", "train-0", "train-1", "train-2"]
    assert module.distill(workers=2) == result
    assert len(calls) == 3
    # A torn derived file must be repaired from the joint rows-and-counters checkpoint.
    (tmp_path / "data/distilled.jsonl").write_text("")
    (tmp_path / "runs/distill-progress.json").write_text("{}")
    assert module.distill(workers=2) == result
    assert len(calls) == 3
    assert read_rows(tmp_path / "data/distilled.jsonl") == rows
