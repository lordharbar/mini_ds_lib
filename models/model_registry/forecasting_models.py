"""
Registry of forecasting models, including traditional time series models,
regression models adapted for time series, and specialized libraries.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
import warnings

# Import regression models as baseline
from mini_ds_lib.models.model_registry.regression_models import get_regression_models

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# NeuralProphet
try:
    from neuralprophet import NeuralProphet
    NEURALPROPHET_AVAILABLE = True
except ImportError:
    NEURALPROPHET_AVAILABLE = False

# PyTorch for custom time series models
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ProphetWrapper(BaseEstimator, RegressorMixin):
    """Wrapper for Facebook Prophet to fit scikit-learn API."""
    
    def __init__(
        self,
        growth: str = 'linear',
        changepoints: Optional[List[str]] = None,
        n_changepoints: int = 25,
        yearly_seasonality: Union[str, bool, int] = 'auto',
        weekly_seasonality: Union[str, bool, int] = 'auto',
        daily_seasonality: Union[str, bool, int] = 'auto',
        seasonality_mode: str = 'additive',
        uncertainty_samples: int = 1000,
        random_state: int = 42,
    ):
        self.growth = growth
        self.changepoints = changepoints
        self.n_changepoints = n_changepoints
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.uncertainty_samples = uncertainty_samples
        self.random_state = random_state
        self.model = None
        self.future = None
        
    def fit(self, X, y):
        """
        Fit Prophet model.
        
        Args:
            X: DataFrame with a 'ds' column for dates
            y: Target values
            
        Returns:
            self
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not available. Please install it to use this model.")
            
        # Create Prophet model
        self.model = Prophet(
            growth=self.growth,
            changepoints=self.changepoints,
            n_changepoints=self.n_changepoints,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            uncertainty_samples=self.uncertainty_samples,
        )
        
        # Extract date column
        date_col = None
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                date_col = col
                break
        
        if date_col is None:
            raise ValueError("X must contain a datetime column")
            
        # Prepare DataFrame for Prophet
        df = pd.DataFrame({
            'ds': X[date_col],
            'y': y,
        })
        
        # Fit the model
        self.model.fit(df)
        
        return self
        
    def predict(self, X):
        """
        Make predictions with Prophet.
        
        Args:
            X: DataFrame with a 'ds' column for dates
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise RuntimeError("Model not fitted")
            
        # Extract date column
        date_col = None
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                date_col = col
                break
        
        if date_col is None:
            raise ValueError("X must contain a datetime column")
            
        # Create future dataframe
        future = pd.DataFrame({'ds': X[date_col]})
        
        # Make predictions
        forecast = self.model.predict(future)
        
        return forecast['yhat'].values


# PyTorch LSTM Time Series Model
class PyTorchLSTMForecaster(BaseEstimator, RegressorMixin):
    """A scikit-learn compatible PyTorch LSTM model for time series forecasting."""
    
    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            
            # LSTM layer
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, num_layers, 
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
            
            # Fully connected layer
            self.fc = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            # Initialize hidden state with zeros
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            
            # Forward propagate LSTM
            out, _ = self.lstm(x, (h0, c0))  # out: batch_size, seq_length, hidden_dim
            
            # Decode the hidden state of the last time step
            out = self.fc(out[:, -1, :])
            return out
    
    def __init__(
        self, 
        seq_length: int = 10,
        hidden_dim: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        self.input_dim = None
        
    def create_sequences(self, X, y):
        """Create input sequences for LSTM."""
        Xs, ys = [], []
        for i in range(len(X) - self.seq_length):
            Xs.append(X[i:(i + self.seq_length)])
            ys.append(y[i + self.seq_length])
        return np.array(Xs), np.array(ys)
        
    def fit(self, X, y):
        """
        Fit LSTM model for time series forecasting.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            self
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install it to use this model.")
            
        # Set random seeds for reproducibility
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Make sure we have enough data for the sequence length
        if len(X) <= self.seq_length:
            raise ValueError(f"Not enough data points. Needed: > {self.seq_length}, got: {len(X)}")
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X, y)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.FloatTensor(y_seq)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Get input dimensionality
        self.input_dim = X.shape[1]
        
        # Initialize model
        self.model = self.LSTMModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=1,
            dropout=self.dropout
        )
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in dataloader:
                # Forward pass
                outputs = self.model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the LSTM model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if not TORCH_AVAILABLE or self.model is None:
            raise RuntimeError("Model not fitted or PyTorch not available")
            
        # For the first seq_length entries, we can't make predictions
        # because we don't have enough history
        predictions = np.full(X.shape[0], np.nan)
        
        if len(X) <= self.seq_length:
            warnings.warn(f"Not enough data points for prediction. Need > {self.seq_length} points.")
            return predictions
        
        # Create sequences without target
        X_seq = []
        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i:(i + self.seq_length)])
        X_seq = np.array(X_seq)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq)
        
        # Prediction
        self.model.eval()
        with torch.no_grad():
            pred_tensor = self.model(X_tensor).squeeze().numpy()
            
        # Fill in predictions (shifted by seq_length due to sequence requirement)
        predictions[self.seq_length:] = pred_tensor
            
        return predictions


class NeuralProphetWrapper(BaseEstimator, RegressorMixin):
    """Wrapper for NeuralProphet to fit scikit-learn API."""
    
    def __init__(
        self,
        growth: str = 'linear',
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 64,
        random_state: int = 42,
    ):
        self.growth = growth
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        
    def fit(self, X, y):
        """
        Fit NeuralProphet model.
        
        Args:
            X: DataFrame with a datetime column
            y: Target values
            
        Returns:
            self
        """
        if not NEURALPROPHET_AVAILABLE:
            raise ImportError("NeuralProphet is not available. Please install it to use this model.")
            
        # Extract date column
        date_col = None
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                date_col = col
                break
        
        if date_col is None:
            raise ValueError("X must contain a datetime column")
            
        # Prepare DataFrame for NeuralProphet
        df = pd.DataFrame({
            'ds': X[date_col],
            'y': y,
        })
        
        # Create NeuralProphet model
        self.model = NeuralProphet(
            growth=self.growth,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            batch_size=self.batch_size,
            n_forecasts=1,
            n_lags=0,
        )
        
        # Fit the model
        self.model.fit(df, freq='auto')
        
        return self
        
    def predict(self, X):
        """
        Make predictions with NeuralProphet.
        
        Args:
            X: DataFrame with a datetime column
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise RuntimeError("Model not fitted")
            
        # Extract date column
        date_col = None
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                date_col = col
                break
        
        if date_col is None:
            raise ValueError("X must contain a datetime column")
            
        # Create future dataframe
        future = pd.DataFrame({'ds': X[date_col]})
        
        # Make predictions
        forecast = self.model.predict(future)
        
        return forecast['yhat1'].values


def get_forecasting_models(
    include_slow: bool = False,
    include_prophet: bool = True,
    include_neural: bool = True,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Get a dictionary of forecasting models.
    
    Args:
        include_slow: Whether to include computationally expensive models
        include_prophet: Whether to include Prophet models
        include_neural: Whether to include neural network models
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of model name to model instance
    """
    # Start with regression models adapted for time series
    base_models = get_regression_models(include_slow, random_state)
    
    # Rename models to indicate they're for time series
    models = {f"TS-{name}": model for name, model in base_models.items()}
    
    # Add Prophet if available and requested
    if include_prophet and PROPHET_AVAILABLE:
        models["Prophet"] = ProphetWrapper(
            growth='linear',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            random_state=random_state
        )
        
        if include_slow:
            models["Prophet-Changepoints"] = ProphetWrapper(
                growth='linear',
                n_changepoints=50,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                random_state=random_state
            )
    
    # Add NeuralProphet if available and requested
    if include_prophet and include_neural and NEURALPROPHET_AVAILABLE:
        models["NeuralProphet"] = NeuralProphetWrapper(
            growth='linear',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            epochs=100,
            random_state=random_state
        )
    
    # Add PyTorch LSTM model if available and neural models are requested
    if include_neural and TORCH_AVAILABLE:
        models["LSTM"] = PyTorchLSTMForecaster(
            seq_length=10,
            hidden_dim=50,
            num_layers=2,
            random_state=random_state
        )
        
        if include_slow:
            models["Deep-LSTM"] = PyTorchLSTMForecaster(
                seq_length=20,
                hidden_dim=100,
                num_layers=3,
                epochs=200,
                random_state=random_state
            )
    
    return models