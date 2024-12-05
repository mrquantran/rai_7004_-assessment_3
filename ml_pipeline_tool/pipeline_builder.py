import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


from ml_pipeline_tool.preprocessor import AdvancedPreprocessor


class MachineLearningPipeline:
    """
    A flexible machine learning pipeline builder for classification and regression.
    """

    ALGORITHMS = {
        "classification": {
            "logistic_regression": LogisticRegression,
            "decision_tree": DecisionTreeClassifier,
            "random_forest": RandomForestClassifier,
            "svm": SVC,
            "neural_network": MLPClassifier,
            "knn": KNeighborsClassifier,
        },
        "regression": {
            "linear_regression": LinearRegression,
            "decision_tree": DecisionTreeRegressor,
            "random_forest": RandomForestRegressor,
            "svm": SVR,
            "neural_network": MLPRegressor,
            "knn": KNeighborsRegressor,
        },
    }

    METRICS = {
        "classification": {"accuracy": accuracy_score, "auc": roc_auc_score},
        "regression": {"mae": mean_absolute_error, "mse": mean_squared_error},
    }

    def __init__(
        self,
        problem_type: str = "classification",
        algorithm: str = "random_forest",
        random_state: int = 42,
        preprocessor: AdvancedPreprocessor = None,
        algorithm_class = None
    ):
        """
        Initialize the machine learning pipeline.

        Args:
            problem_type (str): Type of problem (classification or regression)
            algorithm (str): Specific algorithm to use
            random_state (int): Random seed for reproducibility
            preprocessor (AdvancedPreprocessor): Preprocessor instance
            algorithm_class: Algorithm class to use (overrides default if provided)
        """
        self.problem_type = problem_type
        self.algorithm_name = algorithm
        self.random_state = random_state

        # Validate inputs
        if problem_type not in self.ALGORITHMS:
            raise ValueError(f"Invalid problem type: {problem_type}")

        if algorithm_class is None:
            if algorithm not in self.ALGORITHMS[problem_type]:
                raise ValueError(f"Invalid algorithm for {problem_type}: {algorithm}")
            algorithm_class = self.ALGORITHMS[problem_type][algorithm]

        print('Using algorithm class:', algorithm_class)

        # Set default preprocessor if none provided
        if preprocessor is None:
            preprocessor = AdvancedPreprocessor()

        # Create pipeline
        self.pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", algorithm_class()),
            ]
        )

    def train_and_evaluate(
        self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, num_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train the pipeline and evaluate its performance.

        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target variable
            test_size (float): Proportion of test data
            num_folds (int): Number of cross-validation folds

        Returns:
            Dict[str, Any]: Performance metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Fit pipeline
        self.pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = self.pipeline.predict(X_test)

        # Cross-validation scores
        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=num_folds)

        # Compute metrics
        metrics = {}
        for metric_name, metric_func in self.METRICS[self.problem_type].items():
            if self.problem_type == "classification" and metric_name == "auc":
                y_pred_proba = self.pipeline.predict_proba(X_test)

                if len(set(y_test)) > 2: # Multiclass
                    metrics[metric_name] = metric_func(y_test, y_pred_proba, multi_class="ovr")
                else:
                    metrics[metric_name] = metric_func(y_test, y_pred_proba[:, 1])
            else:
                metrics[metric_name] = metric_func(y_test, y_pred)

        # Additional information
        metrics["cross_val_mean"] = np.mean(cv_scores)
        metrics["cross_val_std"] = np.std(cv_scores)

        return metrics
