FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./

# Install poetry
RUN pip install poetry

# RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml poetry.lock ./

COPY . .

RUN poetry install --only main

ENTRYPOINT ["poetry", "run", "easydel"]