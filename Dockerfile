FROM python:3.12-slim

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY registry.py gateway.py cli.py server.py app.py ./

# Streamlit UI port; the FastAPI server (server.py) can be run on 8000 if desired.
EXPOSE 8501 8000

ENV OLLAMA_HOST=http://ollama:11434

# Default: launch the Streamlit UI, reachable from the host.
CMD ["streamlit", "run", "app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true"]
