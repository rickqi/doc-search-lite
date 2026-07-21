FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install project
COPY . .
RUN pip install --no-cache-dir -e ".[dev]"

# Entrypoint: init sample data + build index + start server
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/docker-entrypoint.sh"]
