"""
Registry of regression models from scikit-learn and other libraries.
"""

from typing import Dict, Any, Optional, List, Tuple

# Scikit-learn models
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    SGDRegressor,
    BayesianRidge,
    ARDRegression,
    HuberRegressor,
    RANSACRegressor,
    TheilSenRegressor,
    PoissonRegressor,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
    BaggingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# XGBoost
import xgboost as xgb

# LightGBM
import lightgbm as lgb

# CatBoost
import catboost as cb

# PyTorch (for neural network models)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_regression_models(
    include_slow: bool = False,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Get a dictionary of regression models.
    
    Args:
        include_slow: Whether to include computationally expensive models
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of model name to model instance
    """
    # Fast models (default)
    models = {
        # Linear models
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(random_state=random_state),
        "Lasso": Lasso(random_state=random_state),
        "ElasticNet": ElasticNet(random_state=random_state),
        
        # Tree-based models
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=random_state),
        
        # Ensemble methods
        "AdaBoost": AdaBoostRegressor(random_state=random_state),
        
        # Boosting libraries
        "XGBoost": xgb.XGBRegressor(objective="reg:squarederror", random_state=random_state),
        "LightGBM": lgb.LGBMRegressor(random_state=random_state),
        "CatBoost": cb.CatBoostRegressor(random_state=random_state, verbose=0),
        
        # KNN
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    }
    
    # Add slower models if requested
    if include_slow:
        slow_models = {
            # More computationally expensive models
            "SVR": SVR(),
            "SGD Regressor": SGDRegressor(random_state=random_state),
            "Bayesian Ridge": BayesianRidge(),
            "ARD Regression": ARDRegression(),
            "Huber Regressor": HuberRegressor(),
            "RANSAC": RANSACRegressor(random_state=random_state),
            "TheilSen": TheilSenRegressor(random_state=random_state),
            "Poisson Regressor": PoissonRegressor(),
            "Bagging Regressor": BaggingRegressor(random_state=random_state),
            "Multi-layer Perceptron": MLPRegressor(
                hidden_layer_sizes=(100, 50), 
                max_iter=500,
                random_state=random_state
            ),
            "Gaussian Process": GaussianProcessRegressor(
                kernel=ConstantKernel() * RBF(), 
                random_state=random_state
            ),
        }
        models.update(slow_models)
    
    return models


# PyTorch Neural Network Model
class PyTorchRegressor:
    """A scikit-learn compatible wrapper for PyTorch regression models."""
    
    class Net(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: List[int] = [100, 50]):
            super().__init__()
            
            layers = []
            prev_dim = input_dim
            
            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(h_dim))
                prev_dim = h_dim
                
            layers.append(nn.Linear(prev_dim, 1))
            
            self.model = nn.Sequential(*layers)
            
        def forward(self, x):
            return self.model(x).squeeze()
    
    def __init__(
        self, 
        hidden_dims: List[int] = [100, 50],
        lr: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        self.input_dim = None
        
        # Set random seed
        if TORCH_AVAILABLE:
            torch.manual_seed(random_state)
    
    def fit(self, X, y):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install it to use this model.")
            
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Initialize model
        self.input_dim = X.shape[1]
        self.model = self.Net(self.input_dim, self.hidden_dims)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in dataloader:
                # Forward pass
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return self
    
    def predict(self, X):
        if not TORCH_AVAILABLE or self.model is None:
            raise RuntimeError("Model not fitted or PyTorch not available")
            
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Prediction
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).numpy()
            
        return predictions