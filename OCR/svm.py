import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
import joblib
# --- Configuration ---
TRAIN_FILE = '../Datasets/archive/emnist-balanced-train.csv'
TEST_FILE = '../Datasets/archive/emnist-balanced-test.csv'
MAPPING_FILE = '../Datasets/archive/emnist-balanced-mapping.txt' # Optional, for label interpretation

# Using a subset of data for faster training.
# Adjust these values based on your computational resources.
# Set to None to use the full dataset (will take a long time).
N_TRAIN_SAMPLES = 0.99  # Number of training samples to use
N_TEST_SAMPLES = 1000   # Number of testing samples to use

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
if N_TRAIN_SAMPLES:
    X_train, _, y_train, _ = train_test_split(X_train_full, y_train_full, train_size=N_TRAIN_SAMPLES, stratify=y_train_full, random_state=42)
else:
    X_train, y_train = X_train_full, y_train_full

if N_TEST_SAMPLES:
    X_test, _, y_test, _ = train_test_split(X_test_full, y_test_full, train_size=N_TEST_SAMPLES, stratify=y_test_full, random_state=42)
else:
    X_test, y_test = X_test_full, y_test_full

print(f"Using {X_train.shape[0]} samples for training and {X_test.shape[0]} samples for testing.")

# Normalize pixel values to the range [0, 1]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- 3. Train SVM Classifier ---
print("\nTraining SVM classifier...")
# Initialize SVM classifier.
# Common kernels: 'rbf' (default), 'linear', 'poly', 'sigmoid'
# 'rbf' is a good general-purpose kernel.
# For large datasets, 'linear' kernel with LinearSVC might be faster.
# C is the regularization parameter.
svm_classifier = SVC(kernel='rbf', C=1.0, random_state=42, verbose=True) # Enable verbose output

start_time = time.time()
svm_classifier.fit(X_train, y_train)
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

# --- 4. Evaluate Classifier ---
print("\nEvaluating classifier...")
y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

print("\nClassification Report:")
# Load mapping for target names if available
target_names = None
try:
    mapping = pd.read_csv(MAPPING_FILE, delimiter=' ', header=None, index_col=0)
    # Create target names like "3 (A)", "10 (0)"
    # Some labels might not be in the subset if it's small
    unique_labels_in_data = sorted(np.unique(np.concatenate((y_train, y_test))))
    target_names = [f"{label} ({chr(mapping.loc[label, 1]) if label in mapping.index else '?'})" for label in unique_labels_in_data]
    report_labels = unique_labels_in_data
    # Ensure target_names correspond to the unique labels actually present
    # This is tricky if not all classes are in y_test after subsetting
    # For simplicity, we'll show the report with numeric labels if mapping is complex for subsets.
    # A more robust approach involves ensuring all classes or using a predefined list of all 47 EMNIST Balanced classes.
    # For now, if subsetting causes issues with target_names, it might print numeric labels.
    print(classification_report(y_test, y_pred)) # Simpler report without names if subsetting makes it complex
    # To use target_names, ensure all labels in y_test and y_pred are covered:
    # print(classification_report(y_test, y_pred, target_names=[name for i, name in enumerate(target_names) if report_labels[i] in np.unique(y_test)]))

except FileNotFoundError:
    print("Mapping file not found. Displaying report with numeric labels.")
    print(classification_report(y_test, y_pred))
except KeyError as e:
    print(f"KeyError in mapping file (some labels might be missing if using a small subset): {e}")
    print(classification_report(y_test, y_pred))

joblib.dump(svm_classifier, 'emnist_svm_model.joblib')
print("SVM model saved as emnist_svm_model.joblib")

# Save the scaler
joblib.dump(scaler, 'emnist_svm_scaler.joblib')
print("SVM scaler saved as emnist_svm_scaler.joblib")

# --- 5. Example: Predict a few samples (optional) ---
print("\nExample predictions (first 5 test samples):")
for i in range(min(5, X_test.shape[0])):
    sample_image = X_test[i]
    true_label_numeric = y_test.iloc[i]
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