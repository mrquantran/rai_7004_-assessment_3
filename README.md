# Machine Learning Pipeline Development Tool

## Project Overview
This application simplifies machine learning pipeline experimentation and development, supporting both Jupyter Notebook and command-line interfaces.

## Features
- Supports classification and regression tasks
- Implements multiple algorithms:
  - Classification: Logistic Regression, Decision Tree, Random Forest
  - Regression: Linear Regression, Decision Tree, Random Forest
- Automated pipeline creation and evaluation
- Cross-validation support
- Model persistence

## Installation
```bash
git clone <repository_url>
cd ml-pipeline-tool
pip install -e .
```

## Usage: Command Line
```bash
ml-pipeline dataset/Housing.csv price regression \
    --problem_type regression \
    --test_size 0.2 \
    --random_state 42 \
    --num_folds 5
```

## Usage: Jupyter Notebook
```python
from ml_pipeline_tool.pipeline_builder import MachineLearningPipeline

# Create pipeline
pipeline = MachineLearningPipeline(
    problem_type='classification',
    algorithm='random_forest'
)

# Train and evaluate
results = pipeline.train_and_evaluate(X, y)
```

## Running Tests
```bash
python -m pytest tests/
```