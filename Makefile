# Critical Graph RAG - Makefile para desarrollo

.PHONY: help install dev-install server test test-local lint format clean

# Variables
PYTHON := python3
PIP := pip3
VENV := venv

help:
	@echo "Critical Graph RAG - Comandos disponibles:"
	@echo ""
	@echo "Instalación:"
	@echo "  make install           - Instalar dependencias (producción)"
	@echo "  make dev-install       - Instalar dependencias + dev tools"
	@echo ""
	@echo "Desarrollo:"
	@echo "  make server            - Iniciar servidor FastAPI (http://localhost:8000)"
	@echo "  make test              - Correr tests del chatbot (servidor debe estar corriendo)"
	@echo "  make test-local        - Tests sin servidor en background"
	@echo ""
	@echo "Calidad:"
	@echo "  make lint              - Verificar estilo con ruff"
	@echo "  make format            - Formatear código con ruff"
	@echo ""
	@echo "Utilidades:"
	@echo "  make clean             - Limpiar archivos temporales"
	@echo "  make venv              - Crear virtual environment"

## Instalación

install:
	$(PIP) install -e .

dev-install: install
	$(PIP) install pytest pytest-asyncio ruff mypy

## Desarrollo

server:
	@echo "🚀 Iniciando servidor en http://localhost:8000"
	@echo "   Docs: http://localhost:8000/docs"
	@echo "   Press Ctrl+C para detener"
	$(PYTHON) server.py

test:
	@echo "🧪 Corriendo tests del chatbot..."
	$(PYTHON) test_chatbot.py

test-local:
	@echo "🧪 Corriendo tests del agente (componente aislado)..."
	$(PYTHON) -m pytest tests/ -v

## Calidad

lint:
	@echo "🔍 Verificando estilo..."
	ruff check src/ server.py

format:
	@echo "✨ Formateando código..."
	ruff format src/ server.py

## Utilidades

venv:
	$(PYTHON) -m venv $(VENV)
	@echo "✓ Virtual environment creado"
	@echo "  Activar con: source $(VENV)/bin/activate"

clean:
	@echo "🧹 Limpiando..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache .ruff_cache
	@echo "✓ Limpieza completada"
