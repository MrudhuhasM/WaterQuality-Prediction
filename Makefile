PYTHON := poetry run python
FLAKE8 := python -m flake8
BLACK := python -m black
ISORT := python -m isort
MYPY := python -m mypy
BANDIT := python -m bandit

install:
	poetry install

format:
	$(ISORT) waterquality
	$(BLACK) waterquality

lint:
	$(FLAKE8) waterquality

typecheck:
	$(MYPY) waterquality

security:
	$(BANDIT) -r waterquality

test:
	$(PYTHON) -m pytest


clean:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .coverage
	rm -rf .tox
	rm -rf .eggs
	rm -rf build
	rm -rf dist
	rm -rf waterquality.egg-info

.PHONY: install format lint typecheck test clean

all: install format lint typecheck test
