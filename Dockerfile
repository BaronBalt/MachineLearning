# Use official Python image
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libpq-dev python3-dev \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "psycopg[c]"

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy application code
COPY . .

# Expose Flask port
EXPOSE 5000

ENV ML_DB_URL="postgresql://mluser:mlpass@localhost:5432/mlregistry"

ENV APP_ENV="production"

# Run Flask app
CMD ["python", "app.py"]
