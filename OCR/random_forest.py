import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler # Though less critical for RF, good practice
import numpy as np
import time
import joblib
# --- Configuration ---
TRAIN_FILE = '../Datasets/archive/emnist-balanced-train.csv'
TEST_FILE = '../Datasets/archive/emnist-balanced-test.csv'
MAPPING_FILE = '../Datasets/archive/emnist-balanced-mapping.txt' # Optional, for label interpretation

# Using a subset of data for faster training.
# Adjust these values. For Random Forest, you can often use more data than with SVM.
# Set to None to use the full dataset.
N_TRAIN_SAMPLES = 0.99  # Number of training samples to use
N_TEST_SAMPLES = 4000   # Number of testing samples to use

# --- 1. Load Data ---
print("Loading data...")
try:
    train_df = pd.read_csv(TRAIN_FILE, header=None)
    test_df = pd.read_csv(TEST_FILE, header=None)
except FileNotFoundError:
    print(f"Error: Make sure '{TRAIN_FILE}' and '{TEST_FILE}' are in the same directory as the script.")
    print("Download them from: https://www.kaggle.com/datasets/crawford/emnist (Balanced split)")
    exit()

print(f"Training data shape: {train_df.shape}")
print(f"Testing data shape: {test_df.shape}")

# --- 2. Preprocess Data ---
# Separate features (pixels) and labels
y_train_full = train_df.iloc[:, 0]
X_train_full = train_df.iloc[:, 1:]

y_test_full = test_df.iloc[:, 0]
X_test_full = test_df.iloc[:, 1:]

# Take a subset of the data if N_TRAIN_SAMPLES or N_TEST_SAMPLES is set
if N_TRAIN_SAMPLES and N_TRAIN_SAMPLES < len(X_train_full):
    X_train, _, y_train, _ = train_test_split(X_train_full, y_train_full, train_size=N_TRAIN_SAMPLES, stratify=y_train_full, random_state=42)
else:
    X_train, y_train = X_train_full, y_train_full

if N_TEST_SAMPLES and N_TEST_SAMPLES < len(X_test_full):
    X_test, _, y_test, _ = train_test_split(X_test_full, y_test_full, train_size=N_TEST_SAMPLES, stratify=y_test_full, random_state=42)
else:
    X_test, y_test = X_test_full, y_test_full

print(f"Using {X_train.shape[0]} samples for training and {X_test.shape[0]} samples for testing.")

# Normalize pixel values to the range [0, 1].
# While Random Forests are less sensitive to feature scaling than SVMs,
# it doesn't hurt and can be good practice if comparing with other algorithms.
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- 3. Train Random Forest Classifier ---
print("\nTraining Random Forest classifier...")
# Initialize Random Forest classifier.
# Key hyperparameters:
#   n_estimators: The number of trees in the forest. More trees can improve performance but increase training time.
#   max_depth: The maximum depth of each tree.
#   min_samples_split: The minimum number of samples required to split an internal node.
#   min_samples_leaf: The minimum number of samples required to be at a leaf node.
#   n_jobs: Number of jobs to run in parallel. -1 means using all available processors.
rf_classifier = RandomForestClassifier(n_estimators=200, # A common default, can be tuned
                                     random_state=42,
                                     n_jobs=4,       # Use all available cores
                                     verbose=1)       # Show some training progress

start_time = time.time()
rf_classifier.fit(X_train, y_train)
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

# --- 4. Evaluate Classifier ---
print("\nEvaluating classifier...")
y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
joblib.dump(rf_classifier, 'emnist_rf_model.joblib')
print("Random Forest model saved as emnist_rf_model.joblib")

# Save the scaler
joblib.dump(scaler, 'emnist_rf_scaler.joblib')
print("Random Forest scaler saved as emnist_rf_scaler.joblib")
print("\nClassification Report:")
# Load mapping for target names if available
target_names = None
try:
    mapping = pd.read_csv(MAPPING_FILE, delimiter=' ', header=None, index_col=0)
    unique_labels_in_data = sorted(np.unique(np.concatenate((y_train.unique(), y_test.unique()))))
    # Create target names like "3 (A)", "10 (0)"
    # This part can be tricky if the subset doesn't contain all classes.
    # For simplicity, we'll provide a basic report.
    print(classification_report(y_test, y_pred))

except FileNotFoundError:
    print("Mapping file not found. Displaying report with numeric labels.")
    print(classification_report(y_test, y_pred))
except KeyError as e:
    print(f"KeyError in mapping file (some labels might be missing if using a small subset): {e}")
    print(classification_report(y_test, y_pred))

# --- 5. Example: Predict a few samples (optional) ---
print("\nExample predictions (first 5 test samples):")
for i in range(min(5, X_test.shape[0])):
    sample_image = X_test[i] # Already scaled
    true_label_numeric = y_test.iloc[i] if isinstance(y_test, pd.Series) else y_test[i]
    predicted_label_numeric = y_pred[i]

    true_char = '?'
    predicted_char = '?'
    if 'mapping' in locals() and mapping is not None:
        try:
            if true_label_numeric in mapping.index:
                 true_char = chr(mapping.loc[true_label_numeric, 1])
            if predicted_label_numeric in mapping.index:
                predicted_char = chr(mapping.loc[predicted_label_numeric, 1])
        except KeyError:
            pass # Keep '?' if label not in mapping

    print(f"Sample {i+1}: True Label = {true_label_numeric} ({true_char}), Predicted Label = {predicted_label_numeric} ({predicted_char})")

# --- 6. Feature Importance (optional) ---
print("\nTop 10 most important pixel features (example):")
try:
    importances = rf_classifier.feature_importances_
    indices = np.argsort(importances)[::-1] # Sort features by importance

    print("Feature (Pixel Index) | Importance")
    print("----------------------|------------")
    for i in range(min(10, X_train.shape[1])):
        print(f"{indices[i]:<21} | {importances[indices[i]]:.4f}")
except Exception as e:
    print(f"Could not compute or display feature importances: {e}")