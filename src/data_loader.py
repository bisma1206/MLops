import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_data():
    """
    Loads the Iris dataset and returns a split of train and test data.
    """
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, iris.target_names

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, target_names = load_data()
    print(f"Data loaded successfully. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
