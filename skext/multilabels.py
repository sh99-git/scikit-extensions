import pandas as pd
import numpy as np
from scipy.sparse import issparse, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that wraps sklearn's MultiLabelBinarizer to integrate seamlessly into scikit-learn pipelines.

    This transformer converts columns containing label iterables or delimiter-separated
    strings into a binary indicator matrix. String values are split using ``delimiter``.
    It is particularly useful for multi-label classification tasks where each sample
    can belong to multiple classes.

    Parameters
    ----------
    sparse_output : bool, default=False
        If True, the output will be a sparse matrix in Compressed Sparse Row (CSR) format. Otherwise, a dense NumPy array is returned.
    delimiter : str, default=","
        Delimiter used to split string values in input columns.

    Attributes
    ----------
    mlbs_ : dict
        A dictionary mapping column names to fitted MultiLabelBinarizer instances.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Input feature names seen during fitting, when all column names are strings.
    """

    def __init__(self, sparse_output=False, delimiter=","):
        """
        Initializes the transformer with the specified parameters.

        Parameters
        ----------
        sparse_output : bool, default=False
            If True, the output will be a sparse matrix in Compressed Sparse Row (CSR) format. Otherwise, a dense NumPy array is returned.
        delimiter : str, default=","
            Delimiter used to split string values in input columns.
        """
        self.sparse_output = sparse_output
        self.delimiter = delimiter

    def _prepare_column(self, values):
        return values.map(
            lambda value: [label.strip() for label in value.split(self.delimiter) if label.strip()]
            if isinstance(value, str)
            else value
        )

    def fit(self, X, y=None):
        """
        Fits the transformer to the provided label sets.

        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_features)
            Each column contains label iterables or delimiter-separated strings,
            where each cell contains the labels for one sample.

        y : None
            Ignored. This parameter exists for compatibility with scikit-learn pipelines.

        Returns
        -------
        self : object
            The fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        self.input_columns_ = list(X.columns)
        self.n_features_in_ = X.shape[1]
        if all(isinstance(column, str) for column in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        self.mlbs_ = {
            col: MultiLabelBinarizer(sparse_output=self.sparse_output).fit(
                self._prepare_column(X[col])
            )
            for col in self.input_columns_
        }
        return self

    def transform(self, X):
        """
        Transforms the provided label sets into a binary indicator matrix.

        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_features)
            Each column contains label iterables or delimiter-separated strings,
            where each cell contains the labels for one sample.

        Returns
        -------
        transformed : pandas DataFrame or sparse matrix of shape (n_samples, n_classes)
            A binary matrix indicating the presence of each class label in each sample.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if not hasattr(self, "mlbs_"):
            raise ValueError("Transformer has not been fitted yet.")
        if list(X.columns) != self.input_columns_:
            raise ValueError("Input columns must match the columns seen during fit.")
        transformed_parts = []
        for col in self.input_columns_:
            mlb = self.mlbs_[col]
            part = mlb.transform(self._prepare_column(X[col]))
            if not self.sparse_output:
                part = pd.DataFrame(
                    part,
                    columns=[f"{col}__{cls}" for cls in mlb.classes_],
                    index=X.index
                )
            transformed_parts.append(part)
        return hstack(transformed_parts, format="csr") if self.sparse_output else pd.concat(transformed_parts, axis=1)

    def inverse_transform(self, X):
        """
        Converts a binary indicator matrix back into label sets.

        Parameters
        ----------
        X : pandas DataFrame or sparse matrix of shape (n_samples, n_classes)
            A binary matrix indicating the presence of each class label in each sample.

        Returns
        -------
        inverse_transformed : pandas DataFrame of shape (n_samples, n_features)
            A DataFrame where each column contains the original label sets for each sample.
        """
        if not hasattr(self, "mlbs_"):
            raise ValueError("Transformer has not been fitted yet.")
        if self.sparse_output and not issparse(X):
            raise ValueError("Expected sparse input for inverse_transform but got dense.")
        if not self.sparse_output:
            X = np.asarray(X)
        result = {}
        if self.sparse_output:
            current_col = 0
            for col in self.input_columns_:
                mlb = self.mlbs_[col]
                n_classes = len(mlb.classes_)
                col_slice = X[:, current_col:current_col + n_classes]
                result[col] = [list(labels) for labels in mlb.inverse_transform(col_slice)]
                current_col += n_classes
        else:
            current_col = 0
            for col in self.input_columns_:
                mlb = self.mlbs_[col]
                n_classes = len(mlb.classes_)
                result[col] = [
                    list(labels)
                    for labels in mlb.inverse_transform(X[:, current_col:current_col + n_classes])
                ]
                current_col += n_classes
        return pd.DataFrame(result)

    def get_feature_names_out(self, input_features=None):
        """
        Returns the output feature names for transformation.

        Parameters
        ----------
        input_features : list of str, default=None
            Input feature names. If None, uses the feature names seen during fitting.

        Returns
        -------
        feature_names_out : list of str
            The output feature names.
        """
        if not hasattr(self, "mlbs_"):
            raise ValueError("Transformer has not been fitted yet.")
        if input_features is None:
            input_features = getattr(
                self,
                "feature_names_in_",
                np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object),
            )
        elif hasattr(self, "feature_names_in_") and not np.array_equal(
            input_features, self.feature_names_in_
        ):
            raise ValueError("input_features must match feature_names_in_ when feature names are available.")
        out = []
        for col, fitted_col in zip(input_features, self.input_columns_):
            mlb = self.mlbs_[fitted_col]
            out.extend([f"{col}__{cls}" for cls in mlb.classes_])
        return np.asarray(out, dtype=object)
