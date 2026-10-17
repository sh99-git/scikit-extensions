.PHONY: test build clean

test:
	uv run pytest --cov=skext --cov-report=term-missing

clean:
	rm -rf dist/ build/ *.egg-info

build: clean test
	uv build
