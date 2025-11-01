.PHONY: test build clean

test:
	pytest --cov=skext --cov-report=term-missing

clean:
	rm -rf dist/ build/ *.egg-info

build: clean test
	poetry build
