import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MultiCollinearityReducer(BaseEstimator, TransformerMixin):
    """
    Transformer to remove multicollinearity by dropping highly correlated features.
    
    Parameters
    ----------
    correlation_threshold : float, default=0.8
        Features with correlation coefficient above this value are considered correlated
    strategy : str, default='first'
        Strategy to select features from correlated groups:
        - 'first': Keep first encountered feature (original behavior)
        - 'last': Keep last encountered feature
        - 'variance': Keep feature with highest variance
        - 'missing_ratio': Keep feature with least missing values
        - 'random': Randomly select feature from correlated group
    random_state : int, optional
        Random state for 'random' strategy
    """
    
    VALID_STRATEGIES = {'first', 'last', 'variance', 'missing_ratio', 'random'}
    
    def __init__(self, correlation_threshold=0.8, strategy='first', random_state=None):
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {self.VALID_STRATEGIES}")
            
        self.correlation_threshold = correlation_threshold
        self.strategy = strategy
        self.random_state = random_state
    
    def _select_by_variance(self, X, features):
        """Select feature with highest variance from group"""
        variances = X[features].var()
        return variances.idxmax()
    
    def _select_by_missing_ratio(self, X, features):
        """Select feature with least missing values from group"""
        missing_ratios = X[features].isnull().mean()
        return missing_ratios.idxmin()
    
    def _select_feature_from_group(self, X, features, corr_matrix):
        """Select a feature from correlated group based on strategy"""
        if len(features) == 1:
            return features[0]
            
        if self.strategy == 'first':
            return features[0]
        elif self.strategy == 'last':
            return features[-1]
        elif self.strategy == 'variance':
            return self._select_by_variance(X, features)
        elif self.strategy == 'missing_ratio':
            return self._select_by_missing_ratio(X, features)
        elif self.strategy == 'random':
            rng = np.random.RandomState(self.random_state)
            return rng.choice(features)
    
    def fit(self, X, y=None):
        """
        Identify features to keep based on correlation analysis.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input features
        y : array-like, optional (ignored)
            Target variable - not used, kept for scikit-learn compatibility
        """
        # Convert to DataFrame if needed
        X_ = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.n_features_in_ = X_.shape[1]
        if isinstance(X, pd.DataFrame):
            self.input_columns_ = list(X.columns)
            if all(isinstance(column, str) for column in X.columns):
                self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            elif hasattr(self, "feature_names_in_"):
                del self.feature_names_in_
        else:
            self.input_columns_ = [f"x{i}" for i in range(X_.shape[1])]
            X_.columns = self.input_columns_
            if hasattr(self, "feature_names_in_"):
                del self.feature_names_in_
        
        # Calculate correlation matrix
        corr_matrix = X_.corr().abs()
        self.feature_correlations_ = corr_matrix.copy()
        
        # Initialize list of features to drop
        to_drop = set()
        processed_groups = []
        
        # For each feature
        for i in range(len(corr_matrix.columns)):
            feat = corr_matrix.columns[i]
            if feat in to_drop:
                continue
                
            # Find correlated features
            correlated = corr_matrix.index[
                corr_matrix.iloc[:, i] > self.correlation_threshold
            ].tolist()
            
            # Remove self-correlation
            correlated.remove(feat)
            
            if correlated:
                # Create group of correlated features including current feature
                group = [feat] + correlated
                
                # Check if we've already processed this group
                if not any(any(f in g for f in group) for g in processed_groups):
                    # Select feature to keep
                    keep_feature = self._select_feature_from_group(X_, group, corr_matrix)
                    
                    # Add others to drop list
                    to_drop.update(f for f in group if f != keep_feature)
                    
                    # Remember this group
                    processed_groups.append(group)
        
        self.selected_features_ = [f for f in X_.columns if f not in to_drop]
        return self
    
    def transform(self, X):
        """
        Remove correlated features from X.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input features
        
        Returns
        -------
        pandas.DataFrame
            Transformed data with selected features only
        """
        if not hasattr(self, "selected_features_"):
            raise ValueError("Transformer has not been fitted yet.")

        X_ = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.input_columns_)
        if X_.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit.")
        if isinstance(X, pd.DataFrame) and list(X.columns) != self.input_columns_:
            raise ValueError("Input columns must match the columns seen during fit.")
        return X_[self.selected_features_]
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names for output features."""
        if not hasattr(self, "selected_features_"):
            raise ValueError("Transformer has not been fitted yet.")
        if input_features is not None and hasattr(self, "feature_names_in_"):
            if not np.array_equal(input_features, self.feature_names_in_):
                raise ValueError("input_features must match feature_names_in_ when feature names are available.")
        return np.asarray(self.selected_features_, dtype=object)


MulticollinearityRemover = MultiCollinearityReducer
