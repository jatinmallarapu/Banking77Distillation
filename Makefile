.PHONY: data baseline gate test lint serve up
data:
	b77 prepare
baseline: data
	b77 train-baseline
gate:
	b77 evaluate-baseline --split gate
	b77 gate
test:
	python -m pytest -q
lint:
	python -m ruff check .
	python -m mypy src/b77
serve:
	uvicorn b77.serve:app --host 127.0.0.1 --port 8000 --no-access-log
up:
	docker compose up --build
