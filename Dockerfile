FROM python:3.10-alpine3.20

# Install system dependencies
RUN apk add --no-cache \
    build-base \
    python3-dev \
    libffi-dev \
    musl-dev \
    gcc \
    py3-pip \
    cargo

WORKDIR /app
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
