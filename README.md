# Mini Data Science Library

A streamlined Python library for automated machine learning workflows that simplifies the process of data preprocessing, model selection, and evaluation.

## 🌟 Features

- **Automated Data Processing**: Intelligent type detection, missing value handling, and feature engineering
- **Comprehensive Model Selection**: 30+ models from scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch, and more
- **Multi-task Support**: Specialized pipelines for regression, classification, and time series forecasting
- **Insightful Visualization**: Performance metrics, feature importance, and model comparisons
- **Easy to Use**: Simple API with sensible defaults and customizable options

## 📦 Installation

Using UV (recommended):

```bash
# Install UV if you don't have it
curl -sSf https://astral.sh/uv/install.sh | bash

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Install package in development mode
uv pip install -e .
```

Using pip:

```bash
pip install -e .
```

## 🚀 Quick Start

### Basic Usage

```python
from mini_ds_lib import AutoML
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize AutoML for your task
auto_ml = AutoML(task_type='regression')  # or 'classification' or 'forecasting'

# Run the workflow
results = auto_ml.run(
    df=df,
    x_cols=['feature1', 'feature2', 'categorical_feature'],
    y_col='target_variable'
)

# Visualize results
auto_ml.plot_model_comparison()
auto_ml.plot_feature_importance()

# Make predictions on new data
new_data = pd.read_csv('new_data.csv')
predictions = AutoML.predict(
    model=results['best_model'],
    preprocessor=results['preprocessor'],
    df=new_data,
    x_cols=['feature1', 'feature2', 'categorical_feature']
)
```

### Regression Example

```python
from mini_ds_lib import AutoML
from sklearn.datasets import load_diabetes

# Load dataset
X, y = load_diabetes(return_X_y=True)
feature_names = load_diabetes().feature_names
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Initialize and run
auto_ml = AutoML(task_type='regression')
results = auto_ml.run(df=df, x_cols=feature_names, y_col='target')

# Create ensemble of top models
ensemble = auto_ml.create_ensemble(top_n=3)

# Make predictions
new_data = df.sample(5)
predictions = AutoML.predict(
    results['best_model'], 
    results['preprocessor'], 
    new_data, 
    feature_names
)
```

### Classification Example

```python
from mini_ds_lib import AutoML
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Initialize and run
auto_ml = AutoML(task_type='classification')
results = auto_ml.run(df=df, x_cols=iris.feature_names, y_col='target')

# Visualize confusion matrix
from mini_ds_lib.utils import plot_confusion_matrix
plot_confusion_matrix(auto_ml.y_test, auto_ml.best_model.predict(auto_ml.X_test))
```

### Time Series Forecasting Example

```python
from mini_ds_lib import AutoML
import pandas as pd
import numpy as np

# Create time series data
dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
df = pd.DataFrame({
    'date': dates,
    'value': np.random.normal(0, 1, len(dates)).cumsum() + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
})

# Extract date features
df['dayofweek'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Initialize and run
auto_ml = AutoML(task_type='forecasting')
results = auto_ml.run(
    df=df,
    x_cols=['date', 'dayofweek', 'month', 'day'],
    y_col='value'
)

# Forecast future values
future_dates = pd.date_range(start='2021-01-01', periods=30, freq='D')
future_df = pd.DataFrame({
    'date': future_dates,
    'dayofweek': future_dates.dayofweek,
    'month': future_dates.month,
    'day': future_dates.day
})

forecast = AutoML.predict(
    results['best_model'], 
    results['preprocessor'], 
    future_df, 
    ['date', 'dayofweek', 'month', 'day']
)
```

## 📈 Available Models

### Regression
- Linear Regression
- Ridge
- Lasso
- ElasticNet
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- Hist Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- K-Nearest Neighbors
- SVR
- Neural Networks (PyTorch)
- And more...

### Classification
- Logistic Regression
- Ridge Classifier
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- K-Nearest Neighbors
- SVM
- Naive Bayes
- Linear/Quadratic Discriminant Analysis
- Neural Networks (PyTorch)
- And more...

### Forecasting
- All regression models adapted for time series
- Prophet
- NeuralProphet
- LSTM (PyTorch)
- And more...

## 🛠️ Configuration Options

The `AutoML` class accepts various parameters to customize the workflow:

```python
AutoML(
    task_type='regression',       # 'regression', 'classification', or 'forecasting'
    test_size=0.2,                # Proportion of data to use for testing
    random_state=42,              # Random seed for reproducibility
    scaling_method='standard',    # 'standard', 'minmax', 'robust', or None
    handle_outliers_method=None,  # 'clip', 'remove', or None
    cv_folds=5,                   # Number of cross-validation folds
    n_jobs=-1                     # Number of jobs for parallel processing
)
```

The `run()` method also accepts options:

```python
auto_ml.run(
    df=df,
    x_cols=['feature1', 'feature2'],
    y_col='target',
    include_slow_models=False,    # Whether to include computationally expensive models
    include_prophet=True,         # Whether to include Prophet models (forecasting only)
    include_neural=True           # Whether to include neural network models
)
```

## 🌐 Project Structure

```
mini_ds_lib/
├── core/                  # Core functionality
│   ├── auto_ml.py         # Main AutoML class
│   └── config.py          # Configuration settings
├── data/                  # Data processing
│   ├── data_type_handler.py  # Type detection and conversion
│   └── preprocessor.py    # Data preprocessing
├── models/                # Model selection
│   ├── model_factory.py   # Model creation
│   ├── model_selection.py # Model evaluation
│   └── model_registry/    # Available models
│       ├── regression_models.py
│       ├── classification_models.py
│       └── forecasting_models.py
└── utils/                 # Utilities
    ├── evaluation.py      # Metrics and evaluation
    ├── visualization.py   # Plotting utilities
    └── validation.py      # Input validation
```

## 📊 Data Processing Features

- **Type Detection**: Automatically identifies numeric, categorical, binary, datetime, and text columns
- **Type Conversion**: Converts columns to appropriate types for modeling
- **Missing Value Handling**: Fills missing values using appropriate strategies based on column type
- **Outlier Handling**: Options to clip or remove outliers
- **Feature Scaling**: Standardization, min-max scaling, or robust scaling
- **Categorical Encoding**: Automatic one-hot encoding for low-cardinality and label encoding for high-cardinality features
- **Date Feature Extraction**: Extracts year, month, day, and other components from datetime columns

## 🔍 Evaluation Metrics

### Regression
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Explained Variance

### Classification
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Log Loss

### Forecasting
- MSE, RMSE, MAE
- R² Score
- Mean Absolute Percentage Error (MAPE)

## 📝 License

MIT

## 🤝 Contributions

Contributions are welcome! Please feel free to submit a Pull Request.
