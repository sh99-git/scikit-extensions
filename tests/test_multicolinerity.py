import pytest
import numpy as np
import pandas as pd
from skext.multicolinerity import MulticollinearityRemover

def test_multicollinearity_basic():
    # Create data with known correlations
    np.random.seed(42)
    X = pd.DataFrame({
        'a': np.random.randn(100),
        'b': np.random.randn(100),
        'c': np.random.randn(100),
    })
    # Make 'c' highly correlated with 'a'
    X['c'] = X['a'] * 0.9 + np.random.randn(100) * 0.1
    
    remover = MulticollinearityRemover(correlation_threshold=0.8)
    remover.fit(X)
    X_transformed = remover.transform(X)
    
    assert len(X_transformed.columns) < len(X.columns)
    assert 'a' in X_transformed.columns or 'c' in X_transformed.columns
    assert 'b' in X_transformed.columns

def test_multicollinearity_strategies():
    np.random.seed(42)
    X = pd.DataFrame({
        'a': np.random.randn(100),
        'b': np.random.randn(100),
    })
    X['c'] = X['a'] * 0.9 + np.random.randn(100) * 0.1
    
    # Test different strategies
    strategies = ['first', 'last', 'variance', 'missing_ratio', 'random']
    for strategy in strategies:
        remover = MulticollinearityRemover(correlation_threshold=0.8, strategy=strategy)
        remover.fit(X)
        assert len(remover.selected_features_) == 2

def test_multicollinearity_error_handling():
    remover = MulticollinearityRemover()
    
    # Test invalid strategy
    with pytest.raises(ValueError, match="strategy must be one of"):
        MulticollinearityRemover(strategy='invalid')
    
    # Test transform before fit
    with pytest.raises(ValueError, match="Transformer has not been fitted"):
        remover.transform(pd.DataFrame())
