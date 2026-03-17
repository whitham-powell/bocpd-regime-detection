.PHONY: install install-dev test lint format notebooks figures clean

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
	uv run jupyter nbconvert --execute --inplace notebooks/*.ipynb

figures: notebooks
	@mkdir -p figures
	@for nb in notebooks/*.ipynb; do \
		if [ -f "$$nb" ]; then \
			echo "Extracting figures from $$nb..."; \
			uv run jupyter nbconvert "$$nb" \
				--to html \
				--ExecutePreprocessor.timeout=600 \
				--ExtractOutputPreprocessor.enabled=True 2>/dev/null || true; \
		fi \
	done
	@find . -type d -name "*_files" | while read dir; do \
		cp $$dir/* figures/ 2>/dev/null || true; \
		rm -r $$dir; \
	done
	@find notebooks -name '*.html' -delete 2>/dev/null || true
	@echo "Figures extracted to figures/"

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -f figures/*.png figures/*.pdf
