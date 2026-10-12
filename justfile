set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

# Formato, lint, tipos e testes. Obrigatório antes de dar uma tarefa como pronta.
check:
    uv run ruff format --check .
    uv run ruff check .
    uv run mypy src
    uv run pytest -q

test *args:
    uv run pytest {{args}}

lock:
    uv lock
    uv sync
