FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml poetry.lock ./

# Install poetry
RUN pip install poetry

# RUN pip install --no-cache-dir -r requirements.txt

RUN poetry install --only main

COPY . .

ENTRYPOINT ["poetry", "run", "easydel"]