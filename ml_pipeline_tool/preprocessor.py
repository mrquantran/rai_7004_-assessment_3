import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


class AdvancedPreprocessor(BaseEstimator, TransformerMixin):
    """
    Advanced preprocessor with multiple feature engineering and imputation strategies.
    """

    def __init__(
        self,
        numerical_strategy: str = "standard",
        categorical_strategy: str = "onehot",
        impute_numerical: str = "median",
        impute_categorical: str = "most_frequent",
        handle_outliers: bool = True,
    ):
        """
        Initialize the advanced preprocessor.

        Args:
            numerical_strategy (str): Scaling strategy for numerical features
            categorical_strategy (str): Encoding strategy for categorical features
            impute_numerical (str): Imputation strategy for numerical features
            impute_categorical (str): Imputation strategy for categorical features
            handle_outliers (bool): Whether to handle outliers
        """
        self.numerical_columns_ = None
        self.categorical_columns_ = None
        self.preprocessor_ = None
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.impute_numerical = impute_numerical
        self.impute_categorical = impute_categorical
        self.handle_outliers = handle_outliers

        # Mapping of scaling strategies
        self.scaling_map = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }

        # Mapping of categorical encoding strategies
        self.encoding_map = {
            "onehot": OneHotEncoder(handle_unknown="ignore"),
            "ordinal": OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            ),
        }

        # Mapping of imputation strategies
        self.impute_map_numerical = {
            "mean": SimpleImputer(strategy="mean"),
            "median": SimpleImputer(strategy="median"),
            "constant": SimpleImputer(strategy="constant", fill_value=0),
        }

        self.impute_map_categorical = {
            "most_frequent": SimpleImputer(strategy="most_frequent"),
            "constant": SimpleImputer(strategy="constant", fill_value="missing"),
        }

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the preprocessor to the data.

        Args:
            X (pd.DataFrame): Input features
            y (array-like, optional): Target variable

        Returns:
            self: Fitted preprocessor
        """
        # Identify column types
        self.numerical_columns_ = X.select_dtypes(include=["int64", "float64"]).columns
        self.categorical_columns_ = X.select_dtypes(
            include=["object", "category"]
        ).columns

        # Create column transformer
        preprocessors = []

        # Numerical preprocessing pipeline
        if len(self.numerical_columns_) > 0:
            num_transformer = Pipeline(
                [
                    ("imputer", self.impute_map_numerical[self.impute_numerical]),
                    ("scaler", self.scaling_map[self.numerical_strategy]),
                ]
            )
            preprocessors.append(("num", num_transformer, self.numerical_columns_))

        # Categorical preprocessing pipeline
        if len(self.categorical_columns_) > 0:
            cat_transformer = Pipeline(
                [
                    ("imputer", self.impute_map_categorical[self.impute_categorical]),
                    ("encoder", self.encoding_map[self.categorical_strategy]),
                ]
            )
            preprocessors.append(("cat", cat_transformer, self.categorical_columns_))

        # Create full preprocessor
        self.preprocessor_ = ColumnTransformer(
            transformers=preprocessors, remainder="drop"
        )

        # Fit the preprocessor
        self.preprocessor_.fit(X)

        return self

    def transform(self, X: pd.DataFrame):
        """
        Transform the input data.

        Args:
            X (pd.DataFrame): Input features

        Returns:
            np.ndarray: Transformed features
        """
        return self.preprocessor_.transform(X)

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names after preprocessing.

        Returns:
            np.ndarray: Feature names
        """
        return self.preprocessor_.get_feature_names_out()
