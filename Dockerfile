# Python 3.11 pinned here so the deploy environment can never drift to an
# unsupported version (avoids missing-wheel build failures).
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure data + model artifact exist in the image (idempotent if already committed).
RUN python -m data.generate_data && python -m ml.train

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# Render/Railway inject $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
