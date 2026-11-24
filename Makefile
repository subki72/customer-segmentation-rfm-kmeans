.PHONY: install lint format test demo run docker-build clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m ruff check src tests main.py
	$(PYTHON) -m ruff format --check src tests main.py
	$(PYTHON) -m flake8 --max-line-length=120 src tests main.py

format:
	$(PYTHON) -m ruff check --fix src tests main.py
	$(PYTHON) -m ruff format src tests main.py

test:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=70

demo:
	$(PYTHON) main.py --demo

run:
	$(PYTHON) -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t customer-segmentation-rfm-kmeans:latest .

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', 'outputs', 'htmlcov']]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
