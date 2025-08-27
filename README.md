# MLOPs Open-Source Model Validation Project

This repository allows developers to contribute ML models. Pull the repo, add your trained model to `models/` (e.g., `yourname_model.pkl`), and push. GitHub Actions will validate it against the base model using 20% of the dataset for metrics like Accuracy, F1, Recall, AUC, and cross-validation.

## Setup
1. Clone the repo: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Add your dataset to `dataset/` (ensure 80% train, 20% valid splits).
4. Train your model using `base_model/training_script.py` as a template.
5. Commit your model with Git LFS: `git lfs track models/*.pkl`

## Testing the Setup
1. Run `python scripts/test_training.py` to generate dataset and train a bad model.
2. Check files: `dataset/train.csv`, `dataset/valid.csv`, `models/test_model.pkl`.
3. Track with Git LFS: `git lfs track models/*.pkl`.
4. Add/commit: `git add .` then `git commit -m "Initial test model"`.
5. Push: `git push origin main` (or your branch).
6. Monitor GitHub Actions for validation.
7. Improve model (edit `n_estimators` in script), retrain, commit/push again.
8. Verify base update in `base_model/model.pkl` and Git LFS tracking.

## Validation
- If your model scores higher (via weighted equation), it becomes the new base model.
- If rejected, check `validation_report.txt` for details.

## Contributing
- Ensure models are serialized with joblib.
- Follow the dataset format in `dataset/data.csv`.
