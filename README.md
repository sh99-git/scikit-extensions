# Scikit-extensions

[![PyPI version](https://badge.fury.io/py/scikit-extensions.svg)](https://badge.fury.io/py/scikit-extensions)
[![Python](https://img.shields.io/pypi/pyversions/scikit-extensions.svg)](https://pypi.org/project/scikit-extensions/)
[![Tests](https://github.com/sh99/scikit-extensions/workflows/Tests/badge.svg)](https://github.com/sh99/scikit-extensions/actions)
[![Coverage](https://codecov.io/gh/sh99/scikit-extensions/branch/main/graph/badge.svg)](https://codecov.io/gh/sh99/scikit-extensions)

A collection of Scikit-learn-compatible transformers for feature engineering, preprocessing, and pipeline utilities.

## Installation

```bash
pip install scikit-extensions
```

Or with Poetry:
```bash
poetry add scikit-extensions
```

## Features

### 1. StatsModelsOLS

A scikit-learn compatible wrapper for statsmodels OLS regression that provides additional statistical insights.

```python
from skext.linear_model import OLSRegressor

# Create and fit model
model = OLSRegressor(fit_intercept=True)
model.fit(X, y)

# Get predictions
y_pred = model.predict(X_test)

# Access statsmodels summary
print(model.summary())
```

### 2. MulticollinearityRemover

Removes highly correlated features using various strategies.

```python
from skext.multicolinerity import MulticollinearityRemover

# Initialize transformer
remover = MulticollinearityRemover(
    correlation_threshold=0.8,
    strategy='variance'  # Options: 'first', 'last', 'variance', 'missing_ratio', 'random'
)

# Fit and transform
X_transformed = remover.fit_transform(X)

# Get selected features
selected_features = remover.selected_features_
```

### 3. MultiLabelBinarizerTransformer

Transform multi-label columns in pandas DataFrames to binary indicator matrices.

```python
from skext.multilabels import MultiLabelBinarizerTransformer

# Sample data
data = pd.DataFrame({
    'tags': [['python', 'ml'], ['python'], ['ml', 'deep-learning']],
    'categories': [['tech'], ['tech', 'tutorial'], ['tutorial']]
})

# Initialize and transform
mlb = MultiLabelBinarizerTransformer(sparse_output=False)
transformed = mlb.fit_transform(data)
```

## Development

1. Clone the repository:
```bash
git clone https://github.com/sh99/scikit-extensions.git
cd scikit-extensions
```

2. Install dependencies:
```bash
poetry install
```

### Running Tests

```bash
make test  # Run tests with coverage
make build  # Run tests and build package
```

## License

MIT License. See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests (`make test`)
5. Submit a pull request

## Requirements

- Python ≥ 3.10
- scikit-learn ≥ 1.7.2
- pandas ≥ 2.3.2
- statsmodels ≥ 0.14.5
