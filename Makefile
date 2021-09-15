venv.nosync: venv.nosync/bin/activate

venv.nosync/bin/activate: requirements.txt
	python3 -m venv venv.nosync
	. venv.nosync/bin/activate; pip install --upgrade pip; pip install -r requirements.txt; pip install pytest; pip install pylint
	touch venv.nosync/bin/activate

test: venv.nosync
	. venv.nosync/bin/activate; pylint src ; pytest tests

clean:
	rm -rf venv.nosync
	find . -depth -name "*.pyc" -type f -delete
