import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data, split_data
from model import CareerPredictor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_model(data_path, target_column, model_type="rf", output_path="models"):
    """Train and evaluate the career prediction model."""
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Load and preprocess data
    print("Loading data...")
    df = load_data(data_path)
    
    # Print basic statistics
    print("\nDataset shape:", df.shape)
    print("\nTarget distribution:")
    print(df[target_column].value_counts())
    
    # Split data into train and test sets
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = split_data(df, target_column)
    
    # Initialize and train model
    print("\nTraining model...")
    predictor = CareerPredictor(model_type=model_type)
    predictor.fit(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = predictor.predict(X_test)
    
    # Print metrics
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=predictor.target_classes, 
                yticklabels=predictor.target_classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'confusion_matrix.png'))
    
    # Feature Importance
    if model_type in ["rf", "xgb"]:
        importances = predictor.feature_importance()
        if importances is not None:
            plt.figure(figsize=(12, 10))
            sns.barplot(x='Importance', y='Feature', data=importances.head(20))
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'feature_importance.png'))
            
            # Save feature importances to CSV
            importances.to_csv(os.path.join(output_path, 'feature_importances.csv'), index=False)
    
    # Save model
    model_filename = os.path.join(output_path, f'career_predictor_{model_type}.joblib')
    predictor.save_model(model_filename)
    
    return predictor, (X_test, y_test, y_pred)

if __name__ == "__main__":
    # Define parameters
    data_path = "data/career_pred.csv"
    target_column = "career"  # Replace with the actual target column name
    model_type = "rf"  # Use "rf" for Random Forest or "xgb" for XGBoost
    
    # Train model
    predictor, evaluation_data = train_model(data_path, target_column, model_type)
    
    print("\nModel training and evaluation completed successfully!")