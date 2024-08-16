FROM python:3.12-slim

RUN apt-get update && apt-get install -y curl \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && mv /root/.local/bin/poetry /usr/local/bin/poetry \
    && poetry config virtualenvs.create false \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -u 1000 -m appuser
WORKDIR /home/appuser/src

COPY pyproject.toml poetry.lock README.md ./

COPY waterquality/inference.py ./

RUN poetry install --no-dev

RUN chown -R appuser:appuser /home/appuser

USER appuser

EXPOSE 8000

CMD ["python", "inference.py"]
