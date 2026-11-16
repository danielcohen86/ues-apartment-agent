FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Create non-root user for security
RUN groupadd -r agent && useradd -r -g agent -d /app agent

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY sql/ ./sql/
COPY main.py .

RUN mkdir -p /app/data /app/logs && chown -R agent:agent /app

USER agent

CMD ["python", "main.py"]
