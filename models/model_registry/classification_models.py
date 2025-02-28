"""
Registry of classification models from scikit-learn and other libraries.
"""

from typing import Dict, Any, Optional, List, Tuple

# Scikit-learn models
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
    PassiveAggressiveClassifier,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

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


def get_classification_models(
    include_slow: bool = False,
    random_state: int = 42,
    n_classes: int = 2
) -> Dict[str, Any]:
    """
    Get a dictionary of classification models.
    
    Args:
        include_slow: Whether to include computationally expensive models
        random_state: Random seed for reproducibility
        n_classes: Number of target classes (affects model configuration)
        
    Returns:
        Dictionary of model name to model instance
    """
    # Objective for XGBoost
    xgb_objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    
    # Default models (fast)
    models = {
        # Linear models
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Ridge Classifier": RidgeClassifier(random_state=random_state),
        
        # Tree-based models
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=random_state),
        
        # Ensemble methods
        "AdaBoost": AdaBoostClassifier(random_state=random_state),
        
        # Boosting libraries
        "XGBoost": xgb.XGBClassifier(objective=xgb_objective, random_state=random_state),
        "LightGBM": lgb.LGBMClassifier(random_state=random_state),
        "CatBoost": cb.CatBoostClassifier(random_state=random_state, verbose=0),
        
        # Naive Bayes
        "Gaussian Naive Bayes": GaussianNB(),
        "Bernoulli Naive Bayes": BernoulliNB(),
        
        # KNN
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        
        # Discriminant Analysis
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    }
    
    # Add slower models if requested
    if include_slow:
        slow_models = {
            # More computationally expensive models
            "SVC": SVC(probability=True, random_state=random_state),
            "Linear SVC": LinearSVC(random_state=random_state, max_iter=10000),
            "SGD Classifier": SGDClassifier(max_iter=1000, random_state=random_state),
            "Passive Aggressive Classifier": PassiveAggressiveClassifier(random_state=random_state),
            "Bagging Classifier": BaggingClassifier(random_state=random_state),
            "Multi-layer Perceptron": MLPClassifier(
                hidden_layer_sizes=(100, 50), 
                max_iter=500,
                random_state=random_state
            ),
            "Gaussian Process Classifier": GaussianProcessClassifier(random_state=random_state),
            "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
            "Multinomial Naive Bayes": MultinomialNB(),
        }
        models.update(slow_models)
        
        # Add voting classifier only if we include multiple models
        base_estimators = [
            ('lr', models["Logistic Regression"]),
            ('rf', models["Random Forest"]),
            ('gb', models["Gradient Boosting"])
        ]
        models["Voting Classifier"] = VotingClassifier(
            estimators=base_estimators,
            voting='soft',
            n_jobs=-1
        )
    
    return models


# PyTorch Neural Network Model for Classification
class PyTorchClassifier:
    """A scikit-learn compatible wrapper for PyTorch classification models."""
    
    class Net(nn.Module):
        def __init__(self, input_dim: int, n_classes: int, hidden_dims: List[int] = [100, 50]):
            super().__init__()
            
            layers = []
            prev_dim = input_dim
            
            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(h_dim))
                prev_dim = h_dim
                
            layers.append(nn.Linear(prev_dim, n_classes))
            
            if n_classes == 1 or n_classes == 2:
                layers.append(nn.Sigmoid())  # Binary classification
            else:
                layers.append(nn.LogSoftmax(dim=1))  # Multi-class
                
            self.model = nn.Sequential(*layers)
            self.n_classes = n_classes
            
        def forward(self, x):
            logits = self.model(x)
            if self.n_classes == 2:
                return logits.squeeze()
            return logits
    
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
        self.n_classes = None
        self.classes_ = None
        
        # Set random seed
        if TORCH_AVAILABLE:
            torch.manual_seed(random_state)
    
    def fit(self, X, y):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install it to use this model.")
            
        # Store classes for later
        self.classes_ = sorted(set(y))
        self.n_classes = len(self.classes_)
        
        # Map classes to indices
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = torch.tensor([class_to_idx[cls] for cls in y], dtype=torch.long)
            
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(X_tensor, y_idx)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Initialize model
        self.input_dim = X.shape[1]
        self.model = self.Net(self.input_dim, self.n_classes, self.hidden_dims)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
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
            if self.n_classes == 2:
                # Binary classification
                predictions = (self.model(X_tensor) > 0.5).numpy().astype(int)
                return [self.classes_[p] for p in predictions]
            else:
                # Multi-class
                probs = self.predict_proba(X)
                predictions = probs.argmax(axis=1)
                return [self.classes_[p] for p in predictions]
    
    def predict_proba(self, X):
        if not TORCH_AVAILABLE or self.model is None:
            raise RuntimeError("Model not fitted or PyTorch not available")
            
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Prediction
        self.model.eval()
        with torch.no_grad():
            if self.n_classes == 2:
                # Binary classification
                probs = self.model(X_tensor).numpy()
                return np.vstack([1 - probs, probs]).T
            else:
                # Multi-class
                logits = self.model(X_tensor)
                probs = torch.exp(logits).numpy()
                return probs