FROM mcr.microsoft.com/playwright/python:v1.48.0-focal
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
COPY . .
ENV PLAYWRIGHT_STATE_DIR=/app/playwright_state
RUN playwright install --with-deps
EXPOSE 8000
