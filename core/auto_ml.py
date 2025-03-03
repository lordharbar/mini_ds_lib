"""
Main AutoML class for orchestrating the entire workflow.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from mini_ds_lib.core.config import VALID_TASK_TYPES, DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE
from mini_ds_lib.data.preprocessor import DataPreprocessor
from mini_ds_lib.models.model_selection import ModelSelector


class AutoML:
    """Main class for automatic machine learning workflow"""
    
    def __init__(
        self,
        task_type: str = 'regression',
        test_size: float = DEFAULT_TEST_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE,
        scaling_method: str = 'standard',
        handle_outliers_method: Optional[str] = None,
        cv_folds: int = 5,
        n_jobs: int = -1,
    ):
        """
        Initialize AutoML with task type and parameters
        
        Args:
            task_type: Type of ML task ('regression', 'classification', or 'forecasting')
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            scaling_method: Method for scaling numeric features ('standard', 'minmax', 'robust', or None)
            handle_outliers_method: Method to handle outliers ('clip', 'remove', or None)
            cv_folds: Number of cross-validation folds
            n_jobs: Number of jobs for parallel processing (-1 for all cores)
        """
        if task_type not in VALID_TASK_TYPES:
            raise ValueError(f"Task type must be one of {VALID_TASK_TYPES}")
            
        self.task_type = task_type
        self.test_size = test_size
        self.random_state = random_state
        self.scaling_method = scaling_method
        self.handle_outliers_method = handle_outliers_method
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        
        # Initialize components
        self.preprocessor = DataPreprocessor(
            task_type=task_type, 
            scaling_method=scaling_method
        )
        
        self.model_selector = ModelSelector(
            task_type=task_type,
            cv_folds=cv_folds,
            n_jobs=n_jobs,
            random_state=random_state
        )
        
        # Store run results
        self.results = {}
        self.feature_importances = None
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.processed_data = None
        
    def run(
        self,
        df: pd.DataFrame,
        x_cols: List[str],
        y_col: str,
        include_slow_models: bool = False,
        include_prophet: bool = True,
        include_neural: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete AutoML workflow
        
        Args:
            df: Input dataframe
            x_cols: List of predictor column names
            y_col: Target column name
            include_slow_models: Whether to include computationally expensive models
            include_prophet: Whether to include Prophet models (forecasting only)
            include_neural: Whether to include neural network models
            
        Returns:
            Dictionary with results
        """
        print(f"Starting AutoML workflow for {self.task_type}...")
        print(f"Input data shape: {df.shape}")
        print(f"Features: {x_cols}")
        print(f"Target: {y_col}")
        
        # Preprocess data
        print("\nPreprocessing data...")
        df_processed = self.preprocessor.preprocess_data(
            df=df,
            x_cols=x_cols,
            y_col=y_col,
            handle_outliers_method=self.handle_outliers_method
        )
        self.processed_data = df_processed
        
        # Update x_cols after preprocessing (some columns might have been transformed)
        x_cols_updated = [col for col in df_processed.columns if col != y_col]
        print(f"Processed data shape: {df_processed.shape}")
        print(f"Updated features: {len(x_cols_updated)} columns")
        
        # Split data
        X = df_processed[x_cols_updated]
        y = df_processed[y_col]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        
        # Load models
        print("\nLoading models...")
        n_classes = None
        if self.task_type == 'classification':
            n_classes = len(np.unique(y))
            print(f"Number of target classes: {n_classes}")
            
        self.model_selector.load_models(
            include_slow=include_slow_models,
            n_classes=n_classes,
            include_prophet=include_prophet,
            include_neural=include_neural
        )
        
        # Train and evaluate models
        print("\nTraining and evaluating models...")
        self.model_selector.train_and_evaluate(
            X_train=self.X_train,
            X_test=self.X_test,
            y_train=self.y_train,
            y_test=self.y_test
        )
        
        # Display results
        self.model_selector.display_results()
        
        # Get feature importance if available
        try:
            self.feature_importances = self.model_selector.get_feature_importance()
            if self.feature_importances is not None:
                print("\nFeature Importances:")
                print(self.feature_importances.to_string(index=False))
        except:
            pass
        
        # Store best model
        self.best_model = self.model_selector.best_model
        
        # Create results dictionary
        self.results = {
            'preprocessor': self.preprocessor,
            'best_model': self.best_model,
            'model_selector': self.model_selector,
            'feature_importances': self.feature_importances,
            'features': x_cols_updated,
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
        }
        
        return self.results
    
    def plot_model_comparison(self, top_n: int = 5, metrics: Optional[List[str]] = None) -> None:
        """
        Plot model comparison
        
        Args:
            top_n: Number of top models to plot
            metrics: List of metrics to plot (None for primary metric only)
        """
        if not hasattr(self.model_selector, 'results') or not self.model_selector.results:
            print("No models have been evaluated yet.")
            return
            
        self.model_selector.plot_results(top_n=top_n, metrics=metrics)
    
    def create_ensemble(self, top_n: int = 3, ensemble_type: str = 'voting') -> Any:
        """
        Create an ensemble of top-performing models
        
        Args:
            top_n: Number of top models to include in the ensemble
            ensemble_type: Type of ensemble ('voting', 'stacking', or 'bagging')
            
        Returns:
            Ensemble model
        """
        if not hasattr(self.model_selector, 'results') or not self.model_selector.results:
            print("No models have been evaluated yet.")
            return None
            
        return self.model_selector.create_ensemble(top_n=top_n, ensemble_type=ensemble_type)
    
    def plot_feature_importance(self, top_n: Optional[int] = 10) -> None:
        """
        Plot feature importance
        
        Args:
            top_n: Number of top features to plot (None for all)
        """
        if self.feature_importances is None:
            print("Feature importance not available for the best model.")
            return
            
        imp_df = self.feature_importances.copy()
        
        # Limit to top N if specified
        if top_n is not None and len(imp_df) > top_n:
            imp_df = imp_df.head(top_n)
            
        # Create horizontal bar chart
        plt.figure(figsize=(10, max(6, len(imp_df) * 0.4)))
        
        # Plot feature importance
        plt.barh(
            y=imp_df['Feature'],
            width=imp_df['Importance'],
            color='#4E79A7'
        )
        
        # Add labels and title
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Show plot
        plt.show()
        
    @staticmethod
    def predict(
        model: Any,
        preprocessor: DataPreprocessor,
        df: pd.DataFrame,
        x_cols: List[str],
        y_col: Optional[str] = None
    ) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            model: Trained model
            preprocessor: Trained preprocessor
            df: Input dataframe
            x_cols: List of predictor column names
            y_col: Target column name (optional, only needed for preprocessing)
            
        Returns:
            Array of predictions
        """
        # Transform new data using the preprocessor
        if y_col and y_col in df.columns:
            # Use the target column if available
            df_processed = preprocessor.transform_new_data(df, x_cols, y_col)
        else:
            # Create a dummy target if needed
            dummy_df = df.copy()
            dummy_y_col = "_dummy_y_"
            dummy_df[dummy_y_col] = 0
            df_processed = preprocessor.transform_new_data(dummy_df, x_cols, dummy_y_col)
            if dummy_y_col in df_processed.columns:
                df_processed = df_processed.drop(dummy_y_col, axis=1)
            
        # Get feature columns after preprocessing
        x_cols_processed = [col for col in df_processed.columns if (y_col is None or col != y_col)]
        
        # Make predictions
        return model.predict(df_processed[x_cols_processed])
    
    def explain_predictions(
        self,
        df: pd.DataFrame,
        x_cols: List[str],
        n_samples: int = 10,
        top_features: int = 5
    ) -> None:
        """
        Explain predictions using SHAP values or feature contributions
        
        Args:
            df: Input dataframe
            x_cols: List of predictor column names
            n_samples: Number of samples to explain
            top_features: Number of top features to show
        """
        if self.best_model is None:
            print("No best model available. Run the workflow first.")
            return
            
        try:
            import shap
            
            # Preprocess the data
            df_processed = self.preprocessor.transform_new_data(df, x_cols)
            x_cols_processed = [col for col in df_processed.columns if col not in [self.results.get('target_col')]]
            
            # Sample a subset of data for explanation
            if len(df_processed) > n_samples:
                df_sample = df_processed.sample(n_samples, random_state=self.random_state)
            else:
                df_sample = df_processed
                
            # Create a SHAP explainer
            try:
                explainer = shap.Explainer(self.best_model, df_sample[x_cols_processed])
                shap_values = explainer(df_sample[x_cols_processed])
                
                # Plot SHAP summary
                shap.plots.beeswarm(shap_values)
                
                # Show feature importance
                shap.plots.bar(shap_values)
            except Exception as e:
                print(f"SHAP explanation error: {str(e)}")
                
                # Fall back to feature importance
                if self.feature_importances is not None:
                    self.plot_feature_importance(top_n=top_features)
                    
        except ImportError:
            print("SHAP library not available. Install with 'pip install shap'.")
            
            # Fall back to feature importance
            if self.feature_importances is not None:
                self.plot_feature_importance(top_n=top_features)
    
    def save_model(self, filename: str) -> None:
        """
        Save the best model and preprocessor to a file
        
        Args:
            filename: Path to save the model
        """
        import joblib
        
        if self.best_model is None:
            print("No best model available. Run the workflow first.")
            return
            
        # Create model bundle
        model_bundle = {
            'task_type': self.task_type,
            'best_model': self.best_model,
            'preprocessor': self.preprocessor,
            'features': self.results.get('features', []),
            'model_info': {
                'task_type': self.task_type,
                'model_type': type(self.best_model).__name__,
                'feature_importance': self.feature_importances,
            }
        }
        
        # Save to file
        joblib.dump(model_bundle, filename)
        print(f"Model saved to {filename}")
    
    @classmethod
    def load_model(cls, filename: str) -> Dict[str, Any]:
        """
        Load a model bundle from a file
        
        Args:
            filename: Path to the saved model
            
        Returns:
            Dictionary with model bundle
        """
        import joblib
        
        # Load from file
        model_bundle = joblib.load(filename)
        print(f"Model loaded from {filename}")
        
        return model_bundle