"""
Developer Instructions for Testing

How to run tests:
1. From project root directory, run:
    pytest tests/
    pytest tests/test_model_trainer.py -v  # For verbose output
    pytest tests/test_model_trainer.py -k "test_name"  # For specific test

How to interpret results:
- Green dots (.) indicate passing tests
- F indicates failed tests
- E indicates errors
- Test summary shows total passed/failed/errors

Common troubleshooting:
1. Check data fixtures are loading correctly
2. Verify expected vs actual outputs in assertions
3. Use pytest.mark.skip for tests needing fixes
4. Add -s flag to see print statements
"""
import pytest
import pandas as pd
from ml_pipeline_tool.pipeline_builder import MachineLearningPipeline
from ml_pipeline_tool.preprocessor import AdvancedPreprocessor
from ml_pipeline_tool.main import validate_input_file

ALGORITHMS = {
    "classification": [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "svm",
        "neural_network",
        "knn",
    ],
    "regression": [
        "linear_regression",
        "decision_tree",
        "random_forest",
        "svm",
        "neural_network",
        "knn",
    ],
}

# Test fixtures
@pytest.fixture
def sample_data():
    """Create sample dataset for testing"""
    X = pd.DataFrame(
        {
            "numeric1": [1, 2, 3, 4, 5],
            "numeric2": [2.1, 3.2, 4.3, 5.4, 6.5],
            "category1": ["A", "B", "A", "C", "B"],
        }
    )
    y = pd.Series([0, 1, 0, 1, 1])
    return X, y

def test_pipeline_initialization():
    """Test basic pipeline initialization"""
    pipeline = MachineLearningPipeline(
        problem_type="classification", algorithm="random_forest"
    )
    assert pipeline is not None
    assert pipeline.problem_type == "classification"
    assert pipeline.algorithm_name == "random_forest"

def test_all_classification_algorithms_iris():
    """Test all classification algorithms on the Iris dataset"""
    file_path = 'dataset/Iris.csv'
    X = validate_input_file(file_path)
    assert X is not None

    y = X['Species']
    X = X.drop(columns=['Species'])

    assert not X.empty

    for algo_name in ALGORITHMS['classification']:
        try:
            pipeline = MachineLearningPipeline(
                problem_type="classification", 
                algorithm=algo_name, 
                random_state=42
            )
            results = pipeline.train_and_evaluate(X, y, test_size=0.2)

            # Verify results structure
            assert isinstance(results, dict)
            assert "accuracy" in results
            assert "cross_val_mean" in results
            assert "cross_val_std" in results
            
        except Exception as e:
            pytest.fail(f"Algorithm {algo_name} failed: {str(e)}")


def test_preprocessor_integration():
    """Test preprocessor integration with pipeline"""
    preprocessor = AdvancedPreprocessor(
        numerical_strategy="standard", categorical_strategy="onehot"
    )
    pipeline = MachineLearningPipeline(
        problem_type="classification",
        algorithm="random_forest",
        preprocessor=preprocessor,
    )
    assert pipeline.pipeline.named_steps["preprocessor"] == preprocessor

def test_all_regression_algorithms_housing():
    """Test all regression algorithms on the Housing dataset"""
    file_path = 'dataset/Housing.csv'
    X = validate_input_file(file_path)
    assert X is not None

    y = X['price']
    X = X.drop(columns=['price'])

    assert not X.empty

    for algo_name in ALGORITHMS['regression']:
        try:
            pipeline = MachineLearningPipeline(
                problem_type="regression", 
                algorithm=algo_name, 
                random_state=42
            )
            results = pipeline.train_and_evaluate(X, y, test_size=0.2)

            assert isinstance(results, dict)
            assert "mse" in results
            assert "cross_val_mean" in results
            assert "cross_val_std" in results
            
        except Exception as e:
            pytest.fail(f"Algorithm {algo_name} failed: {str(e)}")

def test_invalid_configurations():
    """
    Test error handling for invalid pipeline configurations.
    """
    # Invalid problem type
    with pytest.raises(ValueError, match="Invalid problem type"):
        MachineLearningPipeline(problem_type="invalid_type")
    
    # Invalid algorithm for problem type
    with pytest.raises(ValueError, match="Invalid algorithm"):
        MachineLearningPipeline(
            problem_type="classification", 
            algorithm="linear_regression"
        )
