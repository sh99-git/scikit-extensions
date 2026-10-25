import pytest
import pandas as pd
import numpy as np
from scipy.sparse import issparse
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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


def test_multilabel_delimited_strings():
    X = pd.DataFrame({'tags': ['a, b', 'b,c', '', 'a,,c']})

    transformed = MultiLabelBinarizerTransformer().fit_transform(X)

    assert transformed.columns.tolist() == ['tags__a', 'tags__b', 'tags__c']
    assert transformed.values.tolist() == [
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 0],
        [1, 0, 1],
    ]


def test_multilabel_custom_delimiter():
    X = pd.DataFrame({'tags': ['red|blue', 'blue|green']})

    transformed = MultiLabelBinarizerTransformer(delimiter='|').fit_transform(X)

    assert transformed.columns.tolist() == [
        'tags__blue', 'tags__green', 'tags__red'
    ]
    assert transformed.values.tolist() == [[1, 0, 1], [1, 1, 0]]


def test_multilabel_fit_transform_mixed_cell_formats():
    training_data = pd.DataFrame({'tags': ['a, b', ['b', 'c']]})
    test_data = pd.DataFrame({'tags': [['c', 'a'], 'b, c']})
    transformer = MultiLabelBinarizerTransformer().fit(training_data)

    transformed = transformer.transform(test_data)

    assert transformed.columns.tolist() == ['tags__a', 'tags__b', 'tags__c']
    assert transformed.values.tolist() == [[1, 0, 1], [0, 1, 1]]


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


def test_multilabel_column_transformer_compatibility():
    X = pd.DataFrame({
        'tags': [['a', 'b'], ['b'], ['c']],
        'score': [1.0, 2.0, 3.0],
    })
    transformer = ColumnTransformer([
        ('tags', MultiLabelBinarizerTransformer(), ['tags']),
        ('score', 'passthrough', ['score']),
    ])

    transformed = transformer.fit_transform(X)

    assert transformed.shape == (3, 4)
    assert transformer.get_feature_names_out().tolist() == [
        'tags__tags__a', 'tags__tags__b', 'tags__tags__c', 'score__score'
    ]


def test_multilabel_pipeline_compatibility():
    X = pd.DataFrame({'tags': [['a'], ['b', 'c']]})
    pipeline = Pipeline([('encode', MultiLabelBinarizerTransformer())])

    transformed = pipeline.fit_transform(X)

    assert transformed.shape == (2, 3)
    assert pipeline.get_feature_names_out().tolist() == [
        'tags__a', 'tags__b', 'tags__c'
    ]


def test_multilabel_set_output_pandas():
    X = pd.DataFrame({'tags': [['a'], ['b', 'c']]})
    transformer = MultiLabelBinarizerTransformer().set_output(transform='pandas')

    transformed = transformer.fit_transform(X)

    assert isinstance(transformed, pd.DataFrame)
    assert transformed.columns.tolist() == ['tags__a', 'tags__b', 'tags__c']
