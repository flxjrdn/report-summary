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

# ---  Model / Provider config ----------------------------------
# MODEL controls which model to use; override on the command line: make extract MODEL=mistral
MODEL ?= mock
# Infer provider from MODEL (override with PROVIDER=... if needed)
ifeq ($(MODEL),mock)
  PROVIDER ?= mock
else
  PROVIDER ?= ollama
endif

# Default target when you run "make"
.DEFAULT_GOAL := help

# ---  Help  --------------------------------------------------------
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@echo "  install             Install all dependencies in editable mode"
	@echo "  test                Run unit tests with pytest"
	@echo "  ingest              Run ingestion on sample PDF"
	@echo "  ingest-dir          Run ingestion on all PDFs in folder"
	@echo "  extract             Extract values (MODEL=$(MODEL), PROVIDER=$(PROVIDER))"
	@echo "  extract-dir         Extract all ingested docs (MODEL=$(MODEL), PROVIDER=$(PROVIDER))"
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

ingest-dir:
	$(PYTHON) scripts/cli.py ingest-dir

extract:
	$(PYTHON) scripts/cli.py extract $(PDF) --provider $(PROVIDER) --model $(MODEL)

extract-dir:
	$(PYTHON) scripts/cli.py extract-dir --provider $(PROVIDER) --model $(MODEL)

eval:
	$(PYTHON) scripts/cli.py eval

gold:
	$(PYTHON) scripts/cli.py gold

summarize:
	$(PYTHON) scripts/cli.py summarize --provider $(PROVIDER) --model $(MODEL)


summarize-dir:
	$(PYTHON) scripts/cli.py summarize-dir --provider $(PROVIDER) --model $(MODEL)

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
.PHONY: help install test ingest ingest-dir extract extract-dir eval gold summarize summarize-dir ui db-init db-load clean
