FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/vector_store

EXPOSE 10000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000}"]
