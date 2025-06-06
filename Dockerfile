FROM python:3.10-alpine3.20

# Install system dependencies including FFmpeg
RUN apk add --no-cache \
    build-base \
    python3-dev \
    libffi-dev \
    musl-dev \
    gcc \
    py3-pip \
    cargo \
    ffmpeg

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make sure to expose the port that Railway will use
EXPOSE 8000

CMD ["python", "main.py"]