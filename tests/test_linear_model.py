import pytest
import numpy as np
import pandas as pd
from skext.linear_model import OLSRegressor

def test_ols_basic():
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = 3 * X[:, 0] + 2 * X[:, 1] + 1 + np.random.randn(100) * 0.1
    
    # Test with numpy arrays
    model = OLSRegressor()
    model.fit(X, y)
    assert model.coef_.shape == (2,)
    assert isinstance(model.intercept_, float)
    assert model.predict(X).shape == (100,)
    
    # Test with pandas DataFrame
    X_df = pd.DataFrame(X, columns=['feat1', 'feat2'])
    model.fit(X_df, y)
    assert model.feature_names_ == ['const', 'feat1', 'feat2']
    assert model.predict(X_df).shape == (100,)
    assert model.score(X_df, y) > 0.99
    assert model.summary() is not None

def test_statsmodels_ols_no_intercept():
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1
    
    model = OLSRegressor(fit_intercept=False)
    model.fit(X, y)
    assert model.intercept_ == 0.0

def test_statsmodels_ols_error_handling():
    model = OLSRegressor()
    
    # Test not fitted error
    with pytest.raises(ValueError, match="Model not fitted"):
        model.predict(np.random.randn(10, 2))
    
    with pytest.raises(ValueError, match="Model not fitted"):
        _ = model.coef_
