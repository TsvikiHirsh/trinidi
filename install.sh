conda create --name trinidi python==3.9
conda activate trinidi

pip install -r requirements.txt
pip install -e .


pip install ipython pyqt5
pip install pre-commit autoflake isort black pylint
pip install jupyter py2jn

pip install -r docs/docs_requirements.txt

conda install sphinx


# other commands
cd docs && make clean && make html && open -a "Safari" ../build/sphinx/html/index.html && cd ..
