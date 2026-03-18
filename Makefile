.PHONY: help install install-dev test lint format notebooks examples figures clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install core + dev dependencies
	uv sync

install-dev: ## Install dependencies and set up pre-commit hooks
	uv sync
	uv run pre-commit install

test: ## Run test suite
	uv run pytest tests/ -v

lint: ## Check code style (ruff check + format check)
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format: ## Auto-fix code style
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

notebooks: ## Convert .py examples to .ipynb via jupytext
	uv run jupytext --to ipynb examples/*.py

examples: notebooks ## Execute notebooks and render to examples/rendered/ (NB=name to run one)
	@mkdir -p examples/rendered
	@if [ -n "$(NB)" ]; then \
		nb_files=""; \
		for name in $(NB); do \
			if [ ! -f "examples/$$name.ipynb" ]; then \
				echo "Error: examples/$$name.ipynb not found" >&2; exit 1; \
			fi; \
			nb_files="$$nb_files examples/$$name.ipynb"; \
		done; \
	else \
		nb_files=$$(ls examples/*.ipynb 2>/dev/null); \
	fi && \
	tmpdir=$$(mktemp -d) && \
	for nb in $$nb_files; do \
		name=$$(basename "$$nb" .ipynb); \
		echo "Executing $$name..."; \
		uv run jupyter nbconvert "$$nb" \
			--to notebook --execute \
			--ExecutePreprocessor.timeout=600 \
			--output-dir="$$tmpdir" || true; \
	done && \
	for nb in "$$tmpdir"/*.ipynb; do \
		if [ -f "$$nb" ]; then \
			name=$$(basename "$$nb" .ipynb); \
			echo "Rendering $$name to markdown..."; \
			uv run jupyter nbconvert "$$nb" \
				--to markdown \
				--output-dir=examples/rendered \
				--NbConvertApp.output_files_dir="$${name}_files" || true; \
		fi; \
	done && \
	rm -rf "$$tmpdir"
	@echo "Rendered notebooks to examples/rendered/"

figures: notebooks ## Extract figures to extracted_figures/
	@mkdir -p extracted_figures
	@tmpdir=$$(mktemp -d) && \
	for nb in examples/*.ipynb; do \
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
		cp $$dir/* extracted_figures/ 2>/dev/null || true; \
	done && \
	rm -rf "$$tmpdir"
	@echo "Figures extracted to extracted_figures/"

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -f extracted_figures/*.png extracted_figures/*.pdf
