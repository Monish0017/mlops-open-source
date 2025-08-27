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
    parser = argparse.ArgumentParser(description='Compare current model in main with new models in branch')
    parser.add_argument('--base', required=True, help='Path to current model from main')
    parser.add_argument('--new', required=True, help='Directory with new models from branch')
    args = parser.parse_args()
    
    # Load test data
    iris = load_iris()
    X = iris.data
    y = iris.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Evaluate current model from main
    current_accuracy = evaluate_model(args.base, X_test, y_test)
    print(f"Current model in main accuracy: {current_accuracy:.4f}")
    
    # Find all new models in branch
    new_models = []
    for root, _, files in os.walk(args.new):
        for file in files:
            if file.endswith('.pkl'):
                new_models.append(os.path.join(root, file))
    
    if not new_models:
        print("No new models found in branch.")
        with open("validation_report.txt", "w") as f:
            f.write("No new models found in branch.\n")
        return
    
    # Evaluate new models
    best_accuracy = current_accuracy
    best_model_path = None
    
    with open("validation_report.txt", "w") as f:
        f.write(f"Current model in main accuracy: {current_accuracy:.4f}\n\n")
        f.write("New models in branch evaluation:\n")
        
        for model_path in new_models:
            accuracy = evaluate_model(model_path, X_test, y_test)
            print(f"Model {model_path}: {accuracy:.4f}")
            f.write(f"- {model_path}: {accuracy:.4f}\n")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_path = model_path
        
        if best_model_path and best_accuracy > current_accuracy:
            f.write(f"\n✅ Found better model: {best_model_path} with accuracy: {best_accuracy:.4f}\n")
            f.write(f"Improvement: +{best_accuracy - current_accuracy:.4f}\n")
            f.write("Base model will be updated.\n")  # Keep this for workflow detection
        else:
            f.write("\n❌ No better model found. PR will not be merged.\n")
    
    # Update base model if a better one is found
    if best_model_path and best_accuracy > current_accuracy:
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
