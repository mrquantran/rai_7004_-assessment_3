# Machine Learning Pipeline Development Tool

## Project Overview

This application simplifies machine learning pipeline experimentation and development, supporting both Jupyter Notebook and command-line interfaces. It automates the process of building, training, and evaluating machine learning models, making it easier to focus on analyzing results and improving model performance.

## Features

- Supports both **classification** and **regression** tasks
- Implements multiple algorithms:
  - **Classification**:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - SVM
    - Neural Network
    - KNN
  - **Regression**:
    - Linear Regression
    - Decision Tree
    - Random Forest
    - SVM
    - Neural Network
    - KNN
- Automated pipeline creation, including data preprocessing and feature engineering
- Cross-validation support for robust model evaluation
- Model persistence for saving and loading trained models
- Customizable parameters for model tuning

## Prerequisites

- Python 3.6 or higher
- pip package manager

## Installation

```bash
# Clone the repository
git clone <https://github.com/mrquantran/rai_7004_-assessment_3.git>

# Navigate to the project directory
cd ml-pipeline-tool

# Install the package and dependencies
pip install -e .
```

*Note: It is recommended to use a virtual environment to avoid dependency conflicts.*

## Usage

### Command Line Interface

The tool can be used directly from the command line.

#### Command Syntax

```bash
ml-pipeline <dataset_path> <target_variable> <algorithm> [options]
```

#### Positional Arguments

- `<dataset_path>`: Path to the CSV dataset file.
- `<target_variable>`: Name of the target variable in the dataset.
- `<algorithm>`: Algorithm to use (e.g., `random_forest`, `decision_tree`, `logistic_regression`, `linear_regression`). (checking the pipline_builder.py)

#### Optional Arguments

- `--problem_type`: Type of machine learning problem (`classification` or `regression`). Default is `classification`.
- `--test_size`: Fraction of data to use for testing (e.g., `0.2`). Default is `0.2`.
- `--random_state`: Seed for random number generator for reproducibility. Default is `42`.
- `--num_folds`: Number of folds for cross-validation. Default is `5`.

#### Example Usage

```bash
ml-pipeline dataset/Housing.csv price random_forest \
    --problem_type regression \
    --test_size 0.2 \
    --random_state 42 \
    --num_folds 5
```

#### Output

The tool will output evaluation metrics such as accuracy for classification or mean squared error for regression, along with cross-validation scores. The trained model will be saved for future use.

### Jupyter Notebook Interface

The tool can also be used within a Jupyter Notebook for more interactive experimentation.

#### Import the Package

```python
from ml_pipeline_tool.pipeline_builder import MachineLearningPipeline
```

#### Initialize the Pipeline

```python
pipeline = MachineLearningPipeline(
    problem_type='classification',
    algorithm='random_forest',
    test_size=0.2,
    random_state=42,
    num_folds=5
)
```

#### Load and Prepare Data

```python
import pandas as pd

# Load dataset
data = pd.read_csv('dataset/data.csv')

# Split features and target variable
X = data.drop('target_variable', axis=1)
y = data['target_variable']
```

#### Train and Evaluate the Model

```python
results = pipeline.train_and_evaluate(X, y)
print(results)
```

#### Save the Model

```python
pipeline.save_model('saved_models/model.pkl')
```

#### Load a Saved Model

```python
pipeline.load_model('saved_models/model.pkl')
```

## Running Tests

To run the unit tests and ensure everything is working correctly:

```bash
python -m pytest tests/
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes with clear messages.
4. Submit a pull request for review.