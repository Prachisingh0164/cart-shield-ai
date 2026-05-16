.PHONY: install data train test dashboard clean

install:
	pip install -r requirements.txt

data:
	python scripts/generate_data.py

train:
	python src/models/train.py

segment:
	python src/models/segmentation.py data/raw/transactions.csv

test:
	pytest tests/ -v --tb=short

dashboard:
	streamlit run dashboard/app.py

pipeline: data train
	@echo "✅ Full pipeline complete — run 'make dashboard' to launch"

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	rm -f models/*.pkl models/*.json models/*.png

all: install pipeline dashboard
