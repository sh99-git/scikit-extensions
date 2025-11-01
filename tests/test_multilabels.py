import pytest
import pandas as pd
import numpy as np
from scipy.sparse import issparse
from skext.multilabels import MultiLabelBinarizerTransformer

def test_multilabel_basic():
    # Create sample data
    X = pd.DataFrame({
        'tags': [['a', 'b'], ['b', 'c'], ['a'], []],
        'categories': [['x'], ['x', 'y'], ['z'], ['x', 'z']]
    })
    
    transformer = MultiLabelBinarizerTransformer()
    transformer.fit(X)
    transformed = transformer.transform(X)
    
    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape[0] == 4
    assert all(col in transformer.get_feature_names_out() 
              for col in ['tags__a', 'tags__b', 'tags__c',
                         'categories__x', 'categories__y', 'categories__z'])

def test_multilabel_sparse():
    X = pd.DataFrame({
        'tags': [['a', 'b'], ['b', 'c'], ['a'], []]
    })
    
    transformer = MultiLabelBinarizerTransformer(sparse_output=True)
    transformer.fit(X)
    transformed = transformer.transform(X)
    
    assert issparse(transformed)
    assert transformed.shape[0] == 4

def test_multilabel_inverse_transform():
    X = pd.DataFrame({
        'tags': [['a', 'b'], ['b', 'c'], ['a'], []],
    })
    
    transformer = MultiLabelBinarizerTransformer()
    transformed = transformer.fit_transform(X)
    inverse = transformer.inverse_transform(transformed)
    
    assert isinstance(inverse, pd.DataFrame)
    assert (inverse['tags'].tolist() == X['tags'].tolist())

def test_multilabel_error_handling():
    transformer = MultiLabelBinarizerTransformer()
    
    # Test non-DataFrame input
    with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
        transformer.fit([['a', 'b'], ['c']])
