# Trading-signals Slack bot (Bolt Socket Mode). No secrets baked in: tokens and API
# keys come from GCP Secret Manager at runtime via the service account (or env vars).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SLACK_BOT_DB=/tmp/slack_bot.db

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN useradd --create-home --uid 10001 bot \
    && mkdir -p /app/data && chown -R bot:bot /app
USER bot

# Socket Mode: outbound WebSocket only, nothing listens on a port.
CMD ["python", "slack_bot.py"]
