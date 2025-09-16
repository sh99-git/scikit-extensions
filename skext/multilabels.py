import pandas as pd
import numpy as np
from scipy.sparse import issparse, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that wraps sklearn's MultiLabelBinarizer to integrate seamlessly into scikit-learn pipelines.

    This transformer converts a collection of label sets (iterables of iterables) into a binary indicator matrix.
    It is particularly useful for multi-label classification tasks where each sample can belong to multiple classes.

    Parameters
    ----------
    sparse_output : bool, default=False
        If True, the output will be a sparse matrix in Compressed Sparse Row (CSR) format. Otherwise, a dense NumPy array is returned.

    Attributes
    ----------
    mlbs : dict
        A dictionary mapping column names to fitted MultiLabelBinarizer instances.
    feature_names_in_ : list
        List of input feature names (column names) during fitting.
    """

    def __init__(self, sparse_output=False):
        """
        Initializes the transformer with the specified parameters.

        Parameters
        ----------
        sparse_output : bool, default=False
            If True, the output will be a sparse matrix in Compressed Sparse Row (CSR) format. Otherwise, a dense NumPy array is returned.
        """
        self.sparse_output = sparse_output
        self.mlbs = {}
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        """
        Fits the transformer to the provided label sets.

        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_features)
            A collection of label sets, where each set contains the labels for a single sample.

        y : None
            Ignored. This parameter exists for compatibility with scikit-learn pipelines.

        Returns
        -------
        self : object
            The fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        self.feature_names_in_ = X.columns.tolist()
        self.mlbs = {
            col: MultiLabelBinarizer(sparse_output=self.sparse_output).fit(X[col])
            for col in self.feature_names_in_
        }
        return self

    def transform(self, X):
        """
        Transforms the provided label sets into a binary indicator matrix.

        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_features)
            A collection of label sets to transform.

        Returns
        -------
        transformed : pandas DataFrame or sparse matrix of shape (n_samples, n_classes)
            A binary matrix indicating the presence of each class label in each sample.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        transformed_parts = []
        for col in self.feature_names_in_:
            mlb = self.mlbs[col]
            part = mlb.transform(X[col])
            if not self.sparse_output:
                part = pd.DataFrame(
                    part,
                    columns=[f"{col}__{cls}" for cls in mlb.classes_],
                    index=X.index
                )
            transformed_parts.append(part)
        return hstack(transformed_parts) if self.sparse_output else pd.concat(transformed_parts, axis=1)

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
        if self.sparse_output and not issparse(X):
            raise ValueError("Expected sparse input for inverse_transform but got dense.")
        if not self.sparse_output and isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.get_feature_names_out())
        result = {}
        if self.sparse_output:
            current_col = 0
            for col in self.feature_names_in_:
                mlb = self.mlbs[col]
                n_classes = len(mlb.classes_)
                col_slice = X[:, current_col:current_col + n_classes]
                result[col] = mlb.inverse_transform(col_slice)
                current_col += n_classes
        else:
            for col in self.feature_names_in_:
                mlb = self.mlbs[col]
                col_data = X[[f"{col}__{cls}" for cls in mlb.classes_]]
                result[col] = mlb.inverse_transform(col_data.values)
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
        if input_features is None:
            input_features = self.feature_names_in_
        out = []
        for col in input_features:
            mlb = self.mlbs.get(col)
            if mlb:
                out.extend([f"{col}__{cls}" for cls in mlb.classes_])
        return out
