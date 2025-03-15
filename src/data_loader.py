import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Loads the dataset and preprocesses it.
    
    Args:
        file_path (str): Path to the CSV file containing the data
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            # Try to construct a relative path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(current_dir)
            alternative_path = os.path.join(project_dir, file_path)
            
            if os.path.exists(alternative_path):
                file_path = alternative_path
            else:
                print(f"File not found at {file_path} or {alternative_path}")
                raise FileNotFoundError(f"Could not find file: {file_path}")
        
        # Load the CSV file
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        
        # Display basic info
        print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"Found {missing_values.sum()} missing values")
            print(missing_values[missing_values > 0])
            
            # Drop rows with missing values
            df.dropna(inplace=True)
            print(f"After dropping missing values: {df.shape[0]} rows")
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"Found {duplicates} duplicate rows")
            df.drop_duplicates(inplace=True)
            print(f"After dropping duplicates: {df.shape[0]} rows")
        
        # Convert categorical columns to appropriate types
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].astype('category')
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def split_data(df, target_column, test_size=0.2, val_size=0.0, random_state=42):
    """Splits data into train, validation, and test sets.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of data to use for validation
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test) if val_size > 0 else (X_train, X_test, y_train, y_test)
    """
    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    if val_size > 0:
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) > 1 else None
        )
        
        # Second split: separate validation set from training set
        # Calculate validation size relative to the remaining data
        relative_val_size = val_size / (1 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=relative_val_size, random_state=random_state, 
            stratify=y_temp if len(y_temp.unique()) > 1 else None
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        # If no validation set is requested, use simpler split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y if len(y.unique()) > 1 else None
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test

def perform_eda(df):
    """Performs exploratory data analysis on the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary containing EDA results
    """
    eda_results = {}
    
    # Basic statistics
    eda_results['shape'] = df.shape
    eda_results['dtypes'] = df.dtypes
    eda_results['numerical_summary'] = df.describe()
    
    # Categorical columns summary
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    categorical_summary = {}
    
    for col in categorical_columns:
        categorical_summary[col] = df[col].value_counts().to_dict()
    
    eda_results['categorical_summary'] = categorical_summary
    
    # Correlation of numerical features
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_columns) > 0:
        eda_results['correlation'] = df[numerical_columns].corr()
    
    return eda_results

def encode_categorical_features(df, encoding_method='one_hot'):
    """Encodes categorical features in the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        encoding_method (str): Encoding method ('one_hot' or 'label')
        
    Returns:
        pd.DataFrame: Dataframe with encoded features
    """
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    if encoding_method == 'one_hot':
        # Apply one-hot encoding
        return pd.get_dummies(df, columns=list(categorical_columns))
    
    elif encoding_method == 'label':
        # Apply label encoding
        from sklearn.preprocessing import LabelEncoder
        df_encoded = df.copy()
        
        for col in categorical_columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col])
        
        return df_encoded
    
    else:
        raise ValueError(f"Unknown encoding method: {encoding_method}")

def preprocess_for_training(X_train, X_test, y_train, y_test, scaling=True):
    """Preprocesses the data for model training.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Training target
        y_test (pd.Series): Test target
        scaling (bool): Whether to scale numerical features
        
    Returns:
        tuple: (X_train_processed, X_test_processed, y_train, y_test)
    """
    from sklearn.preprocessing import StandardScaler
    
    # Make copies to avoid modifying the original data
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    # Scale numerical features if requested
    if scaling:
        numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numerical_columns) > 0:
            scaler = StandardScaler()
            X_train_processed[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
            X_test_processed[numerical_columns] = scaler.transform(X_test[numerical_columns])
    
    return X_train_processed, X_test_processed, y_train, y_test

if __name__ == "__main__":
    # Try different potential locations for the data file
    potential_paths = [
        "data/career_pred.csv",
        "../data/career_pred.csv",
        "./data/career_pred.csv",
        "career_pred.csv"
    ]
    
    df = None
    for path in potential_paths:
        try:
            print(f"Trying to load from: {path}")
            df = load_data(path)
            print(f"Successfully loaded data from {path}")
            break
        except Exception as e:
            print(f"Could not load from {path}: {str(e)}")
    
    if df is None:
        print("Could not find the data file in any of the expected locations.")
        print("Please specify the correct file path when running this script.")
        exit(1)
    
    # Perform EDA
    eda_results = perform_eda(df)
    print("EDA completed")
    
    # Check target distribution
    target_column = "Suggested Job Role"  # Update with your actual target column
    if target_column in df.columns:
        print(f"\nTarget distribution ({target_column}):")
        print(df[target_column].value_counts())
    else:
        print(f"Target column '{target_column}' not found. Available columns:")
        print(df.columns.tolist())
    
    # Split data (no validation set by default)
    try:
        X_train, X_test, y_train, y_test = split_data(df, target_column, val_size=0.0)
        print("Data loading and preprocessing completed successfully")
    except Exception as e:
        print(f"Error splitting data: {str(e)}")