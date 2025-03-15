import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import xgboost as xgb

class CareerPredictor:
    def __init__(self, model_type="rf"):
        """Initialize the Career Predictor model.
        
        Args:
            model_type (str): Type of model to use ('rf' for Random Forest, 'xgb' for XGBoost)
        """
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.target_classes = None
    
    def _create_preprocessor(self, X):
        """Create a preprocessing pipeline for numerical and categorical features."""
        # Identify numerical and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        # Create preprocessing steps for different column types
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        # Create the preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        self.feature_names = list(numeric_features) + list(categorical_features)
        return preprocessor
    
    def fit(self, X, y):
        """Fit the model on the provided data."""
        # Create preprocessor
        self.preprocessor = self._create_preprocessor(X)
        
        # Create model pipeline
        if self.model_type == "rf":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "xgb":
            model = xgb.XGBClassifier(random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Create and fit the pipeline
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', model)
        ])
        
        self.model.fit(X, y)
        self.target_classes = list(y.unique())
        
        return self
    
    def predict(self, X):
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet. Call fit() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability estimates for each class."""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet. Call fit() first.")
        
        return self.model.predict_proba(X)
    
    def save_model(self, filepath):
        """Save the trained model to a file."""
        if self.model is None:
            raise ValueError("No trained model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'target_classes': self.target_classes,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model from a file."""
        model_data = joblib.load(filepath)
        
        career_predictor = cls(model_type=model_data['model_type'])
        career_predictor.model = model_data['model']
        career_predictor.feature_names = model_data['feature_names']
        career_predictor.target_classes = model_data['target_classes']
        
        return career_predictor

    def feature_importance(self):
        """Extract feature importance from the model."""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet.")
        
        if self.model_type == "rf":
            # For Random Forest
            importances = self.model.named_steps['classifier'].feature_importances_
            
            # Get feature names after transformation
            preprocessor = self.model.named_steps['preprocessor']
            feature_names = []
            
            for name, transformer, features in preprocessor.transformers_:
                if name == 'cat':
                    # For categorical features, get the one-hot encoded feature names
                    for feature, categories in zip(features, transformer.categories_):
                        for category in categories:
                            feature_names.append(f"{feature}_{category}")
                else:
                    # For numerical features
                    feature_names.extend(features)
                    
            # Create a DataFrame of feature importances
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            return feature_importance_df
        
        elif self.model_type == "xgb":
            # For XGBoost
            importances = self.model.named_steps['classifier'].feature_importances_
            
            # Get feature names after transformation
            preprocessor = self.model.named_steps['preprocessor']
            feature_names = []
            
            for name, transformer, features in preprocessor.transformers_:
                if name == 'cat':
                    # For categorical features, get the one-hot encoded feature names
                    for feature, categories in zip(features, transformer.categories_):
                        for category in categories:
                            feature_names.append(f"{feature}_{category}")
                else:
                    # For numerical features
                    feature_names.extend(features)
                    
            # Create a DataFrame of feature importances
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            return feature_importance_df
        
        else:
            return None