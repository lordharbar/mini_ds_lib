"""Configuration settings for the mini_ds_lib package."""

# Valid task types
VALID_TASK_TYPES = ["regression", "classification", "forecasting"]

# Default parameters
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_CV_FOLDS = 5

# Metrics configuration
REGRESSION_METRICS = ["mse", "rmse", "r2", "mae", "explained_variance"]
CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1", "auc", "log_loss"]
FORECASTING_METRICS = ["mse", "rmse", "r2", "mae", "mape"]

# Visualization settings
PLOT_FIGSIZE = (12, 8)
PLOT_COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", 
               "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"]