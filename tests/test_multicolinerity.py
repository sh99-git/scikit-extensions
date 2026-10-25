import pytest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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


def test_multicollinearity_pipeline_compatibility():
    X = pd.DataFrame({
        'a': [1.0, 2.0, 3.0, 4.0],
        'b': [2.0, 4.0, 6.0, 8.0],
        'c': [4.0, 1.0, 3.0, 2.0],
    })
    pipeline = Pipeline([
        ('reduce', MulticollinearityRemover(correlation_threshold=0.8)),
        ('scale', StandardScaler()),
    ])

    transformed = pipeline.fit_transform(X)

    assert transformed.shape == (4, 2)
    assert pipeline.get_feature_names_out().tolist() == ['a', 'c']


def test_multicollinearity_column_transformer_compatibility():
    X = pd.DataFrame({
        'a': [1.0, 2.0, 3.0, 4.0],
        'b': [2.0, 4.0, 6.0, 8.0],
        'c': [4.0, 1.0, 3.0, 2.0],
    })
    transformer = ColumnTransformer([
        ('reduce', MulticollinearityRemover(correlation_threshold=0.8), ['a', 'b', 'c']),
    ])

    transformed = transformer.fit_transform(X)

    assert transformed.shape == (4, 2)
    assert transformer.get_feature_names_out().tolist() == ['reduce__a', 'reduce__c']


def test_multicollinearity_set_output_pandas():
    X = pd.DataFrame({
        'a': [1.0, 2.0, 3.0, 4.0],
        'b': [2.0, 4.0, 6.0, 8.0],
        'c': [4.0, 1.0, 3.0, 2.0],
    })
    transformer = MulticollinearityRemover(correlation_threshold=0.8).set_output(
        transform='pandas'
    )

    transformed = transformer.fit_transform(X)

    assert isinstance(transformed, pd.DataFrame)
    assert transformed.columns.tolist() == ['a', 'c']
