# ---------------------------------------------------------------------------
# PatchOps - Dockerfile
# ---------------------------------------------------------------------------
# python:3.10-slim is a stripped-down Debian base with just enough to run
# Python - far smaller than the full python:3.10 image, which matters both
# for free-tier build/storage limits and for cold-start time on platforms
# like Render that spin containers down when idle.
FROM python:3.10-slim

# Prevents Python from writing .pyc files and buffering stdout/stderr -
# without this, your `logger.info(...)` calls might not show up in the
# host's log viewer until the buffer flushes, which makes debugging painful.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only requirements.txt first, before the rest of the source code.
# Docker caches layers - if requirements.txt hasn't changed, this `pip
# install` layer is reused on the next build instead of re-running, which
# makes iterating on main.py/index.html much faster to rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code.
COPY main.py auth.py incident_store.py ./
COPY templates ./templates

# Documents which port the app listens on. This does NOT actually publish
# the port - that happens via `docker run -p` or the hosting platform's
# config - but it's good practice and some platforms (Render) read it.
EXPOSE 8000

# Runs the app directly with uvicorn (not `python main.py`) so we don't
# rely on the `if __name__ == "__main__"` block, and so we can pass
# --host/--port explicitly here as the canonical production entrypoint.
#
# IMPORTANT: We use the shell form (not exec form with hardcoded port)
# so that ${PORT:-8000} expands at container start time - this lets AWS
# App Runner / Render / Fly.io inject their own PORT env var and have the
# container actually bind to it, while still defaulting to 8000 locally.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}