style:
	black --line-length 89 --target-version py38 colablink/
	isort colablink/

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:
	python setup.py sdist bdist_wheel

upload:
	twine upload dist/*

test:
	pytest tests/

.PHONY: style install install-dev clean build upload test
