FROM python:3.8-slim

WORKDIR /app

COPY requirements-docker.txt ./
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY .env ./.env
COPY . .

# Make sure the environment is activated:
RUN echo "Make sure flask are installed:"
RUN python -c "import flask"

CMD gunicorn line-movie-api:app --bind 0.0.0.0:${PORT:-8000}