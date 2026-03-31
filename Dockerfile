FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY poc/ poc/
COPY data/ data/

RUN pip install --no-cache-dir ".[gemini]"

ENV LLM_PROVIDER=gemini

EXPOSE 8080

CMD ["uvicorn", "poc.app:app", "--host", "0.0.0.0", "--port", "8080"]
