import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib
import os

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='label')

# Split (80% train, 20% valid)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Save splits
train_df = pd.concat([X_train, y_train], axis=1)
valid_df = pd.concat([X_valid, y_valid], axis=1)

# Get project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create directories if they don't exist
os.makedirs(os.path.join(project_root, 'dataset'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)

# Save splits
train_df.to_csv(os.path.join(project_root, 'dataset', 'train.csv'), index=False)
valid_df.to_csv(os.path.join(project_root, 'dataset', 'valid.csv'), index=False)

# Train model (SVC for better accuracy)
model = SVC(kernel='rbf', C=1.0, random_state=42)  # Good alternative model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, os.path.join(project_root, 'models', 'test_model_svc.pkl'))
