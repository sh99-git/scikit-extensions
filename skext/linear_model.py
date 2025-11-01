import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
import statsmodels.api as sm

class OLSRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper for statsmodels OLS regression.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    missing : str, default='raise'
        How to handle missing values. Options: 'raise', 'drop'
    """
    
    def __init__(self, fit_intercept=True, missing='raise'):
        self.fit_intercept = fit_intercept
        self.missing = missing
        self.model_ = None
        self.results_ = None
        self.feature_names_ = None
    
    def fit(self, X, y):
        """
        Fit OLS regression model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns
        -------
        self : object
            Returns self.
        """
        # Check input data
        X, y = check_X_y(X, y, y_numeric=True)
        
        # Convert to DataFrame to preserve feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = [f'x{i}' for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.feature_names_)
        
        # Add constant if needed
        if self.fit_intercept:
            X = sm.add_constant(X)
            self.feature_names_ = ['const'] + self.feature_names_
        
        # Create and fit model
        self.model_ = sm.OLS(y, X, missing=self.missing)
        self.results_ = self.model_.fit()
        
        return self
    
    def predict(self, X):
        """
        Make predictions using fitted model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
        
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values
        """
        # Check if model is fitted
        if self.results_ is None:
            raise ValueError("Model not fitted. Call 'fit' first.")
        
        # Check and prepare input
        X = check_array(X)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_[1:] if self.fit_intercept else self.feature_names_)
        
        # Add constant if needed
        if self.fit_intercept:
            X = sm.add_constant(X)
        
        return self.results_.predict(X)
    
    def score(self, X, y):
        """Return R-squared score."""
        return self.results_.rsquared
    
    @property
    def coef_(self):
        """Return model coefficients (excluding intercept)."""
        if self.results_ is None:
            raise ValueError("Model not fitted. Call 'fit' first.")
        return self.results_.params[1:] if self.fit_intercept else self.results_.params
    
    @property
    def intercept_(self):
        """Return intercept value."""
        if self.results_ is None:
            raise ValueError("Model not fitted. Call 'fit' first.")
        return self.results_.params[0] if self.fit_intercept else 0.0
    
    def summary(self):
        """Return statsmodels summary object."""
        if self.results_ is None:
            raise ValueError("Model not fitted. Call 'fit' first.")
        return self.results_.summary()
