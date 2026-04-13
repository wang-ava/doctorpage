FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY doctor_web ./doctor_web

CMD ["sh", "-c", "uvicorn doctor_web.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
