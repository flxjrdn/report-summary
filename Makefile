# =============================
# SFCR Pipeline - Developer Makefile
# =============================

# Default directories
DATA_DIR := data/samples
OUT_DIR  := artifacts/ingest
PYTHON   := python3

# Default target when you run "make"
.DEFAULT_GOAL := help

# These targets do not refer to files - always run them
.PHONY: help install test ingest validate schema clean

# ---  Help  --------------------------------------------------------
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@echo "  install     Install dependencies"
	@echo "  test        Run pytest unit tests"
	@echo "  ingest      Run ingestion on sample PDFs"
	@echo "  validate    Validate produced *.ingest.json files"
	@echo "  schema      Regenerate ingestion JSON Schema"
	@echo "  clean       Remove build artifacts"
	@echo ""
	@echo "Example: make ingest"

# ---  Setup  -------------------------------------------------------
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .[dev]

test:
	pytest -q

# ---  Pipeline  ----------------------------------------------------
ingest:
	$(PYTHON) scripts/cli.py ingest $(DATA_DIR) --outdir $(OUT_DIR)

validate:
	for f in $(OUT_DIR)/*.ingest.json; do \
	$(PYTHON) scripts/cli.py validate $$f; \
	done

schema:
	$(PYTHON) scripts/export_schema.py
	git diff --exit-code schema/ingestion.schema.json || \
	(echo "Schema changed — bump schema_version and update examples." && exit 1)

# ---  Maintenance  -------------------------------------------------
clean:
	rm -rf __pycache__ .pytest_cache $(OUT_DIR)/*.json artifacts/*.json
