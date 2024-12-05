import argparse
import sys
import pandas as pd
import joblib
from typing import Optional
from .pipeline_builder import MachineLearningPipeline
from .advanced_logging import MLPipelineLogger
from .preprocessor import AdvancedPreprocessor

def validate_input_file(file_path: str) -> pd.DataFrame:
    """
    Validate and load the input CSV file.

    Args:
        file_path (str): Path to the input CSV file

    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("Input CSV file is empty")
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        raise ValueError(f"Error reading CSV file: {e}") from e

def run_pipeline(
    csv_path: str,
    target_column: str,
    algorithm: str,
    problem_type: str = 'classification',
    test_size: float = 0.2,
    random_state: int = 42,
    num_folds: int = 5,
    preprocessing_strategy: str = 'standard',
    log_file: Optional[str] = None
) -> dict:
    """
    Enhanced pipeline runner with advanced logging and preprocessing.
    """
    # Initialize logger
    logger = MLPipelineLogger(log_file or f'{algorithm}_pipeline.log')

    try:
        # Log pipeline start
        start_time = logger.log_pipeline_start({
            'csv_path': csv_path,
            'target_column': target_column,
            'algorithm': algorithm,
            'problem_type': problem_type
        })

        # Load and validate data
        df = pd.read_csv(csv_path)

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Use advanced preprocessor
        preprocessor = AdvancedPreprocessor(
            numerical_strategy=preprocessing_strategy
        )

        # Check if using algorithms
        algorithm_class = MachineLearningPipeline.ALGORITHMS[problem_type][algorithm]

        # Create pipeline
        pipeline = MachineLearningPipeline(
            problem_type=problem_type,
            algorithm=algorithm,
            random_state=random_state,
            preprocessor=preprocessor,
            algorithm_class=algorithm_class
        )

        # Train and evaluate
        results = pipeline.train_and_evaluate(
            X,
            y,
            test_size=test_size,
            num_folds=num_folds
        )

        # Save model
        model_path = f"{algorithm}_pipeline.pkl"
        joblib.dump(pipeline.pipeline, model_path)
        results['model_path'] = model_path

        # Log pipeline end
        logger.log_pipeline_end(start_time, results)

        return results

    except Exception as e:
        logger.log_error(f"Pipeline Error: {e}", e)
        raise e

    finally:
        logger.save_performance_log()

def main():
    """
    Command-line interface for the ML Pipeline tool.
    """
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline Tool")
    parser.add_argument("csv_path", help="Path to input CSV file")
    parser.add_argument("target_column", help="Name of the target column")
    parser.add_argument("algorithm", help="Machine learning algorithm to use")
    parser.add_argument(
        "--problem_type",
        default="classification",
        choices=["classification", "regression"],
        help="Type of machine learning problem",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Proportion of test data"
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_folds", type=int, default=5, help="Number of cross-validation folds"
    )

    args = parser.parse_args()

    try:
        results = run_pipeline(
            args.csv_path,
            args.target_column,
            args.algorithm,
            args.problem_type,
            args.test_size,
            args.random_state,
            args.num_folds,
        )
        print("Pipeline Results:")
        for key, value in results.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
