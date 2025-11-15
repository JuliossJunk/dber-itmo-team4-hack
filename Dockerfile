FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.3
RUN pip install "poetry==$POETRY_VERSION"

ENV POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app
COPY pyproject.toml poetry.lock ./
COPY app ./app/
COPY .env .env

RUN poetry install --no-interaction --no-ansi

EXPOSE 8501

CMD ["streamlit", "run", "app/server.py", "--server.address=0.0.0.0"]
