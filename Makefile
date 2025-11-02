# =============================
# SFCR Pipeline - Developer Makefile
# =============================

PYTHON := python3

# load .env if present
ifneq (,$(wildcard .env))
	include .env
	export $(shell sed 's/=.*//' .env)
endif

export SFCR_DATA
export SFCR_OUTPUT

# Default target when you run "make"
.DEFAULT_GOAL := help

# ---  Help  --------------------------------------------------------
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@echo "  install             Install all dependencies in editable mode"
	@echo "  test                Run unit tests with pytest"
	@echo "  ingest              Run ingestion on sample PDFs"
	@echo "  validate            Validate produced *.ingest.json files"
	@echo "  schema              Regenerate ingestion JSON Schema"
	@echo "  extract             Extract values from ingested PDF"
	@echo "  extract-ollama      Extract using Ollama provider with mistral model"
	@echo "  extract-dir-ollama  Extract from directory using Ollama and mistral model"
	@echo "  eval                Evaluate extraction results against gold CSV file"
	@echo "  gold                Generate gold standard data"
	@echo "  ui                  Launch the user interface"
	@echo "  db-init             Initialize the database"
	@echo "  db-load             Load data into the database"
	@echo "  clean               Remove build artifacts and temporary files"

# ---  Setup  -------------------------------------------------------
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .[dev]

test:
	pytest -q

# ---  Pipeline  ----------------------------------------------------
ingest:
	$(PYTHON) scripts/cli.py ingest

validate:
	$(PYTHON) scripts/cli.py validate-dir

schema:
	$(PYTHON) scripts/export_schema.py
	git diff --exit-code schema/ingestion.schema.json || \
	(echo "Schema changed — bump schema_version and update examples." && exit 1)

extract:
	$(PYTHON) scripts/cli.py extract $(PDF)

extract-dir:
	$(PYTHON) scripts/cli.py extract-dir

extract-ollama:
	$(PYTHON) scripts/cli.py extract --provider ollama --model mistral

extract-dir-ollama:
	$(PYTHON) scripts/cli.py extract-dir --provider ollama --model mistral

eval:
	$(PYTHON) scripts/cli.py eval

gold:
	$(PYTHON) scripts/cli.py gold

summarize:
	$(PYTHON) scripts/cli.py summarize --provider mock --model mock

summarize-ollama:
	$(PYTHON) scripts/cli.py summarize --provider ollama --model mistral

ui:
	$(PYTHON) scripts/cli.py ui

db-init:
	$(PYTHON) scripts/cli.py db-init

db-load:
	$(PYTHON) scripts/cli.py db-load

# ---  Maintenance  -------------------------------------------------
clean:
	rm -rf __pycache__ .pytest_cache artifacts/ingest/*.json artifacts/*.json

# These targets do not refer to files - always run them
.PHONY: help install test ingest validate schema extract extract-dir extract-ollama extract-dir-ollama eval gold summarize ui db-init db-load clean
