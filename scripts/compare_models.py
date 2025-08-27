import argparse
import joblib
import pandas as pd
import os
from evaluate import evaluate
import subprocess

def score_function(metrics):
    """
    Weighted score combining Accuracy, F1, Recall, AUC, and CV metrics.
    Adjust weights as needed.
    """
    w_acc, w_f1, w_rec, w_auc, w_cv_acc, w_cv_f1, w_cv_rec = 0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1
    score = (
        w_acc * metrics["accuracy"] +
        w_f1 * metrics["f1"] +
        w_rec * metrics["recall"] +
        (w_auc * metrics["auc"] if metrics["auc"] else 0) +
        w_cv_acc * metrics["cv_accuracy"] +
        w_cv_f1 * metrics["cv_f1"] +
        w_cv_rec * metrics["cv_recall"]
    )
    return score

def get_committer_info():
    # Get committer name and email from git
    name = subprocess.run(["git", "log", "-1", "--pretty=format:%an"], capture_output=True, text=True).stdout.strip()
    email = subprocess.run(["git", "log", "-1", "--pretty=format:%ae"], capture_output=True, text=True).stdout.strip()
    return name, email

def main(base_model_path, new_models_dir):
    # Load validation data
    valid = pd.read_csv("dataset/valid.csv")
    X, y = valid.drop("label", axis=1), valid["label"]

    # Load base model if exists
    if os.path.exists(base_model_path):
        base_model = joblib.load(base_model_path)
        base_metrics = evaluate(base_model, X, y)
        base_score = score_function(base_metrics)
    else:
        base_score = -1  # No base, so any new model is better
        base_metrics = {}

    # Get committer info
    committer_name, committer_email = get_committer_info()

    # Find the latest new model (assume single file for simplicity)
    new_model_files = [f for f in os.listdir(new_models_dir) if f.endswith('.pkl')]
    if not new_model_files:
        print("No new model found")
        exit(1)
    new_model_path = os.path.join(new_models_dir, new_model_files[0])
    new_model = joblib.load(new_model_path)
    new_metrics = evaluate(new_model, X, y)
    new_score = score_function(new_metrics)

    with open("validation_report.txt", "w") as f:
        f.write(f"Committer: {committer_name} ({committer_email})\n")
        f.write(f"Base Model: {base_metrics}, Score={base_score}\n")
        f.write(f"New Model: {new_metrics}, Score={new_score}\n")

    if new_score > base_score:
        print("✅ New model accepted")
        os.makedirs(os.path.dirname(base_model_path), exist_ok=True)
        joblib.dump(new_model, base_model_path)
        with open("validation_report.txt", "a") as f:
            f.write(f"Base model updated by {committer_name}\n")
    else:
        print("❌ New model rejected")
        with open("validation_report.txt", "a") as f:
            f.write(f"Rejected: New model worse than base. Base metrics: {base_metrics}\n")
        exit(1)  # Fails CI/CD

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--new", type=str, required=True)
    args = parser.parse_args()

    main(args.base, args.new)
