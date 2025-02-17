import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Loads the dataset and preprocesses it."""
    df = pd.read_csv("data\career_pred.csv")
    
    # Drop missing values
    df.dropna(inplace=True)
    
    return df

def split_data(df, target_column):
    """Splits data into train and test sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)


