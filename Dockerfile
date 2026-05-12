# Dockerfile para Critical Graph RAG Chatbot

FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# Instalar dependencias del sistema (si es necesario)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de requisitos
COPY requirements-prod.txt ./

# Instalar dependencias Python
RUN pip install --upgrade pip && \
    pip install -r requirements-prod.txt

# Copiar fuentes
COPY src ./src
COPY config ./config
COPY server.py .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Exponer puerto
EXPOSE 8000

# Comando para iniciar
CMD ["python", "server.py"]
