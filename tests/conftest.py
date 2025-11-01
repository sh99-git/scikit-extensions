import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1
    return X, y

@pytest.fixture
def sample_multilabel_data():
    return pd.DataFrame({
        'tags': [['a', 'b'], ['b', 'c'], ['a'], []],
        'categories': [['x'], ['x', 'y'], ['z'], ['x', 'z']]
    })
