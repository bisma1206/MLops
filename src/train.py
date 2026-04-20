import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib
import os
from data_loader import load_data

def train_model():
    # Load data
    X_train, X_test, y_train, y_test, target_names = load_data()

    # Start MLflow run
    with mlflow.start_run():
        # Define model parameters
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Initialize and train model
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted")
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log the model
        mlflow.sklearn.log_model(clf, "model")
        
        # Save model locally as well for the API to use easily without full MLflow setup
        os.makedirs("models", exist_ok=True)
        joblib.dump(clf, "models/model.joblib")
        
        print(f"Model trained. Metrics: {metrics}")
        print("Model saved to models/model.joblib and logged to MLflow.")

if __name__ == "__main__":
    train_model()
