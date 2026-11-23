# CPU baseline image: reproducible serving without GPU or private credentials.
FROM python:3.11-slim
WORKDIR /app
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
COPY pyproject.toml ./
COPY src src
RUN pip install --no-cache-dir .
COPY configs configs
COPY data/labels.json data/labels.json
COPY evals evals
RUN b77 prepare && b77 train-baseline
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn","b77.serve:app","--host","0.0.0.0","--port","8000","--no-access-log"]
