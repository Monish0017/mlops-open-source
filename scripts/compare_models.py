import argparse
import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def evaluate_model(model_path, X_test, y_test):
    """Evaluate a model on the test set."""
    try:
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    except Exception as e:
        print(f"Error evaluating model {model_path}: {str(e)}")
        return 0.0

def main():
    parser = argparse.ArgumentParser(description='Compare models and update base model if new one is better')
    parser.add_argument('--base', required=True, help='Path to base model')
    parser.add_argument('--new', required=True, help='Directory with new models')
    args = parser.parse_args()
    
    # Load test data
    iris = load_iris()
    X = iris.data
    y = iris.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create base model directory if it doesn't exist
    base_dir = os.path.dirname(args.base)
    os.makedirs(base_dir, exist_ok=True)
    
    # If base model doesn't exist, create an empty one
    if not os.path.exists(args.base):
        print(f"Base model doesn't exist. Creating empty placeholder.")
        with open(args.base, 'wb') as f:
            f.write(b'')
    
    # Evaluate base model
    base_accuracy = evaluate_model(args.base, X_test, y_test)
    print(f"Base model accuracy: {base_accuracy:.4f}")
    
    # Find all new models
    new_models = []
    for root, _, files in os.walk(args.new):
        for file in files:
            if file.endswith('.pkl'):
                new_models.append(os.path.join(root, file))
    
    if not new_models:
        print("No new models found for comparison.")
        with open("validation_report.txt", "w") as f:
            f.write("No new models found for comparison.\n")
        return
    
    # Evaluate new models
    best_accuracy = base_accuracy
    best_model_path = None
    
    for model_path in new_models:
        accuracy = evaluate_model(model_path, X_test, y_test)
        print(f"Model {model_path} accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = model_path
    
    # Generate report
    with open("validation_report.txt", "w") as f:
        f.write(f"Base model accuracy: {base_accuracy:.4f}\n")
        for model_path in new_models:
            accuracy = evaluate_model(model_path, X_test, y_test)
            f.write(f"Model {model_path} accuracy: {accuracy:.4f}\n")
        
        if best_model_path:
            f.write(f"\nBest model: {best_model_path} (accuracy: {best_accuracy:.4f})\n")
            if best_accuracy > base_accuracy:
                f.write("Base model will be updated.\n")
            else:
                f.write("No update needed - base model is already optimal.\n")
        else:
            f.write("No model outperformed the base model.\n")
    
    # Update base model if a better one is found
    if best_model_path and best_accuracy > base_accuracy:
        print(f"Updating base model with {best_model_path} (accuracy: {best_accuracy:.4f})")
        try:
            # Explicitly copy the model file
            best_model = joblib.load(best_model_path)
            joblib.dump(best_model, args.base)
            print("Base model updated successfully.")
        except Exception as e:
            print(f"Error updating base model: {str(e)}")
    else:
        print("No update needed - base model is already optimal or no better model found.")

if __name__ == "__main__":
    main()
