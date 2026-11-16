# One image for both services (the API and the Streamlit page); docker-compose.yml picks
# the command. Python 3.14 because pyproject.toml requires it.
FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

# Dependencies first: this layer is rebuilt only when pyproject.toml or uv.lock change.
# README.md is there because pyproject.toml names it as the package readme.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

# Not root. /data holds the checkpoint database (a volume in compose); /app/logs holds
# the log file.
RUN useradd --create-home app \
    && mkdir -p /data /app/logs \
    && chown -R app:app /data /app/logs
USER app

ENV PATH="/app/.venv/bin:$PATH" \
    RESEARCH_DB=/data/checkpoints.db

EXPOSE 8000 8501
CMD ["uvicorn", "research_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]
