import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RareCategoryEncoder(BaseEstimator, TransformerMixin):
    """Replace infrequent categorical values with a fallback label."""

    def __init__(self, min_frequency=0.05, replacement="other", columns=None):
        self.min_frequency = min_frequency
        self.replacement = replacement
        self.columns = columns

    def fit(self, X, y=None):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        if self.columns is None:
            self.columns_ = [
                col for col in X.columns
                if not pd.api.types.is_numeric_dtype(X[col])
                and not pd.api.types.is_bool_dtype(X[col])
            ]
        else:
            self.columns_ = list(self.columns)
            missing = [col for col in self.columns_ if col not in X.columns]
            if missing:
                raise ValueError(f"Column '{missing[0]}' is not present in X.")

        self.rare_categories_ = {}
        for col in self.columns_:
            values = X[col].astype("object").fillna("__missing__")
            counts = values.value_counts(normalize=True)
            threshold = self.min_frequency
            rare = counts[counts < threshold].index
            self.rare_categories_[col] = set(rare)
        return self

    def transform(self, X):
        if not hasattr(self, "rare_categories_"):
            raise ValueError("Encoder has not been fitted yet.")

        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        transformed = X.copy()

        for col in self.columns_:
            if col not in transformed.columns:
                raise ValueError(f"Column '{col}' is not present in X.")
            values = transformed[col].astype("object").fillna("__missing__")
            transformed[col] = values.map(
                lambda value: self.replacement if value in self.rare_categories_[col] else value
            )
            transformed[col] = transformed[col].replace("__missing__", self.replacement)
        return transformed

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = list(self.columns_)
        return np.asarray(input_features, dtype=object)


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical columns using the mean target value."""

    def __init__(self, columns=None, handle_unknown="mean"):
        self.columns = columns
        self.handle_unknown = handle_unknown

    def fit(self, X, y):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y = pd.Series(np.asarray(y)).reset_index(drop=True)

        if self.columns is None:
            self.columns_ = [
                col for col in X.columns
                if not pd.api.types.is_numeric_dtype(X[col])
                and not pd.api.types.is_bool_dtype(X[col])
            ]
        else:
            self.columns_ = list(self.columns)
            missing = [col for col in self.columns_ if col not in X.columns]
            if missing:
                raise ValueError(f"Column '{missing[0]}' is not present in X.")

        self.global_mean_ = float(y.mean()) if len(y) else 0.0
        self.mapping_ = {}

        for col in self.columns_:
            mapping = pd.Series(y.values, index=X[col].astype("object").fillna("__missing__")).groupby(level=0).mean()
            self.mapping_[col] = mapping.to_dict()
        return self

    def transform(self, X):
        if not hasattr(self, "mapping_"):
            raise ValueError("Encoder has not been fitted yet.")

        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        transformed = X.copy()

        for col in self.columns_:
            if col not in transformed.columns:
                raise ValueError(f"Column '{col}' is not present in X.")
            values = transformed[col].astype("object").fillna("__missing__")
            transformed[col] = values.map(self.mapping_[col]).fillna(self.global_mean_)
        return transformed

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = list(self.columns_)
        return np.asarray(input_features, dtype=object)


class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract calendar features from datetime columns."""

    VALID_FEATURES = {
        "month",
        "day_of_month",
        "day_of_week",
        "hour",
        "minute",
        "is_weekend",
        "quarter",
        "year",
    }

    def __init__(self, features=None, columns=None):
        if features is None:
            features = ["month", "day_of_week", "hour", "is_weekend"]
        self.features = list(features)
        self.columns = columns

    def fit(self, X, y=None):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        invalid = [feature for feature in self.features if feature not in self.VALID_FEATURES]
        if invalid:
            raise ValueError(f"Unsupported time features: {invalid}")

        if self.columns is None:
            candidate_columns = []
            for col in X.columns:
                try:
                    pd.to_datetime(X[col])
                    candidate_columns.append(col)
                except (TypeError, ValueError):
                    pass
            self.columns_ = candidate_columns
        else:
            self.columns_ = list(self.columns)
            missing = [col for col in self.columns_ if col not in X.columns]
            if missing:
                raise ValueError(f"Column '{missing[0]}' is not present in X.")

        return self

    def transform(self, X):
        if not hasattr(self, "columns_"):
            raise ValueError("Transformer has not been fitted yet.")

        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        output = pd.DataFrame(index=X.index)

        for col in self.columns_:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' is not present in X.")
            series = pd.to_datetime(X[col], errors="coerce")

            for feature in self.features:
                if feature == "month":
                    values = series.dt.month
                elif feature == "day_of_month":
                    values = series.dt.day
                elif feature == "day_of_week":
                    values = series.dt.dayofweek
                elif feature == "hour":
                    values = series.dt.hour
                elif feature == "minute":
                    values = series.dt.minute
                elif feature == "is_weekend":
                    values = series.dt.dayofweek.isin([5, 6])
                elif feature == "quarter":
                    values = series.dt.quarter
                elif feature == "year":
                    values = series.dt.year
                else:
                    continue

                name = feature if len(self.columns_) == 1 else f"{col}__{feature}"
                output[name] = values

        return output

    def get_feature_names_out(self, input_features=None):
        if not hasattr(self, "columns_"):
            raise ValueError("Transformer has not been fitted yet.")

        names = []
        for col in self.columns_:
            for feature in self.features:
                names.append(feature if len(self.columns_) == 1 else f"{col}__{feature}")
        return np.asarray(names, dtype=object)
