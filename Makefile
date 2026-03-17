.PHONY: install install-dev test lint format notebooks docs figures clean

install:
	uv sync

install-dev:
	uv sync
	uv run pre-commit install

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

notebooks:
	uv run jupytext --to ipynb notebooks/*.py

docs: notebooks
	@mkdir -p docs
	@tmpdir=$$(mktemp -d) && \
	for nb in notebooks/*.ipynb; do \
		if [ -f "$$nb" ]; then \
			name=$$(basename "$$nb" .ipynb); \
			echo "Executing $$name..."; \
			uv run jupyter nbconvert "$$nb" \
				--to notebook --execute \
				--ExecutePreprocessor.timeout=600 \
				--output-dir="$$tmpdir" || true; \
		fi; \
	done && \
	for nb in "$$tmpdir"/*.ipynb; do \
		if [ -f "$$nb" ]; then \
			name=$$(basename "$$nb" .ipynb); \
			echo "Rendering $$name to markdown..."; \
			uv run jupyter nbconvert "$$nb" \
				--to markdown \
				--output-dir=docs \
				--NbConvertApp.output_files_dir="$${name}_files" || true; \
		fi; \
	done && \
	rm -rf "$$tmpdir"
	@echo "Rendered notebooks to docs/"

figures: notebooks
	@mkdir -p figures
	@tmpdir=$$(mktemp -d) && \
	for nb in notebooks/*.ipynb; do \
		if [ -f "$$nb" ]; then \
			echo "Executing $$(basename $$nb)..."; \
			uv run jupyter nbconvert "$$nb" \
				--to notebook --execute \
				--ExecutePreprocessor.timeout=600 \
				--output-dir="$$tmpdir" || true; \
		fi; \
	done && \
	for nb in "$$tmpdir"/*.ipynb; do \
		if [ -f "$$nb" ]; then \
			echo "Extracting figures from $$(basename $$nb)..."; \
			uv run jupyter nbconvert "$$nb" \
				--to html \
				--ExtractOutputPreprocessor.enabled=True \
				--output-dir="$$tmpdir" 2>/dev/null || true; \
		fi; \
	done && \
	find "$$tmpdir" -type d -name "*_files" | while read dir; do \
		cp $$dir/* figures/ 2>/dev/null || true; \
	done && \
	rm -rf "$$tmpdir"
	@echo "Figures extracted to figures/"

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -f figures/*.png figures/*.pdf
