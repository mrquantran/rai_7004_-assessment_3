import argparse  # For parsing command-line arguments
import joblib    # For saving and loading models
import sys       # For system-specific parameters and functions
from mypackage.module import something  # Use absolute import for the module

def run_pipeline(
    csv_path: str,
    target_column: str,
    algorithm: str,
    problem_type: str,
    test_size: float,
    random_state: int,
    num_folds: int
) -> dict:
    """
    Run the machine learning pipeline.

    Args:
        csv_path (str): Path to the input CSV file.
        target_column (str): Name of the target column in the dataset.
        algorithm (str): Machine learning algorithm to use.
        problem_type (str): Type of machine learning problem ('classification' or 'regression').
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        num_folds (int): Number of folds for cross-validation.

    Returns:
        dict: A dictionary containing the results of the pipeline execution, including the model path.
    """
    try:
        # Train and evaluate the model using the provided parameters
        results = pipeline.train_and_evaluate(
            X,
            y,
            test_size=test_size,
            num_folds=num_folds
        )

        # Save the trained model to a file
        model_path = f"{algorithm}_pipeline.pkl"
        joblib.dump(pipeline.pipeline, model_path)  # Save the pipeline model
        results['model_path'] = model_path  # Add model path to results

        # Log the end of the pipeline execution
        logger.log_pipeline_end(start_time, results)

        return results  # Return the results of the pipeline

    except Exception as e:
        # Log any errors that occur during the pipeline execution
        logger.log_error(f"Pipeline Error: {e}", e)
        raise e  # Re-raise the exception for further handling

    finally:
        # Save performance logs regardless of success or failure
        logger.save_performance_log()

def main() -> None:
    """
    Command-line interface for the Machine Learning Pipeline tool.

    This function sets up the argument parser, processes command-line arguments,
    and runs the machine learning pipeline with the specified parameters.
    """
    # Set up argument parser for command-line inputs
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline Tool")
    parser.add_argument("csv_path", help="Path to input CSV file")  # Input CSV file path
    parser.add_argument("target_column", help="Name of the target column")  # Target column name
    parser.add_argument("algorithm", help="Machine learning algorithm to use")  # Algorithm choice
    parser.add_argument(
        "--problem_type",
        default="classification",
        choices=["classification", "regression"],
        help="Type of machine learning problem",  # Problem type (classification or regression)
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Proportion of test data"  # Test data proportion
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random seed for reproducibility"  # Random seed
    )
    parser.add_argument(
        "--num_folds", type=int, default=5, help="Number of cross-validation folds"  # Cross-validation folds
    )

    args = parser.parse_args()  # Parse the command-line arguments

    try:
        # Run the pipeline with the provided arguments
        results = run_pipeline(
            args.csv_path,
            args.target_column,
            args.algorithm,
            args.problem_type,
            args.test_size,
            args.random_state,
            args.num_folds,
        )
        # Print the results of the pipeline
        print("Pipeline Results:")
        for key, value in results.items():
            print(f"{key}: {value}")
    except Exception as e:
        # Print any errors that occur during the pipeline execution
        print(f"Error: {e}")
        sys.exit(1)  # Exit the program with an error status

if __name__ == "__main__":
    main()  # Execute the main function when the script is run
