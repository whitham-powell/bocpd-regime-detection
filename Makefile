.PHONY: install install-dev test lint format notebooks figures clean

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

notebooks:
	jupytext --to ipynb notebooks/*.py
	jupyter nbconvert --execute --inplace notebooks/*.ipynb

figures: notebooks

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -f notebooks/*.ipynb
	rm -f figures/*.png figures/*.pdf
