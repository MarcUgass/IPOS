import cv2
import joblib
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

# --- Configuration ---
SVM_MODEL_PATH = 'emnist_svm_model.joblib'
SVM_SCALER_PATH = 'emnist_svm_scaler.joblib'
RF_MODEL_PATH = 'emnist_rf_model.joblib'
RF_SCALER_PATH = 'emnist_rf_scaler.joblib' # Assuming you might use a different scaler for RF, or the same one.
                                         # If it's the same, you can point RF_SCALER_PATH to SVM_SCALER_PATH.
MAPPING_FILE_PATH = '../Datasets/archive/emnist-balanced-mapping.txt' # Must be in the same directory or provide full path

def preprocess_image(image_path, invert_colors=True, target_size=(28, 28)):
    """
    Loads an image, converts it to grayscale, resizes, inverts colors (optional),
    flattens, and prepares it for the EMNIST model.
    """
    try:
        img = Image.open(image_path).convert('L') # Convert to grayscale
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None

    # Resize to 28x28
    img = img.resize(target_size, Image.LANCZOS) # Use LANCZOS for better quality downscaling
    if invert_colors:
        # EMNIST datasets typically have white characters on a black background (character pixels are high value).
        # If your image is black character on white background, inversion is needed.
        img = ImageOps.invert(img)

    
    # Convert image to numpy array
    pixel_data = np.array(img )
    print(pixel_data)
    mask = np.where(pixel_data > 70, 255, 0) # Convert to binary (1 for white, 0 for black)
    pixel_data = mask * pixel_data # Scale to 0-255 range
    """ cv2.imshow("Preprocessed Image", pixel_data.astype(np.uint8)) # Display the preprocessed image
    cv2.waitKey(0) """

    # Flatten the 28x28 image to a 784-element array
    flattened_image = mask.flatten()

    return flattened_image

def get_char_from_label(label, mapping_df):
    """Converts a numeric label to its character representation using the mapping."""
    try:
        return chr(mapping_df.loc[label, 1])
    except KeyError:
        return "Unknown Label"

if __name__ == "__main__":
    # --- Load Models and Scalers ---
    try:
        svm_model = joblib.load(SVM_MODEL_PATH)
        svm_scaler = joblib.load(SVM_SCALER_PATH)
        rf_model = joblib.load(RF_MODEL_PATH)
        # If you used the same scaler for RF during its training, you can load the SVM scaler:
        # rf_scaler = joblib.load(SVM_SCALER_PATH)
        rf_scaler = joblib.load(RF_SCALER_PATH)
        print("Models and scalers loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading model or scaler: {e}")
        print("Please ensure you have trained and saved the models and scalers using joblib,")
        print("and that the paths in this script are correct.")
        exit()

    # --- Load EMNIST Mapping ---
    try:
        emnist_mapping = pd.read_csv(MAPPING_FILE_PATH, delimiter=' ', header=None, index_col=0)
        print("EMNIST mapping file loaded.")
    except FileNotFoundError:
        print(f"Error: EMNIST mapping file '{MAPPING_FILE_PATH}' not found.")
        print("Predictions will be numeric labels only.")
        emnist_mapping = None

    # --- Get User Image Path ---
    user_image_path = "output.jpeg"

    # --- Preprocess the User's Image ---
    # Ask user about color inversion, as this is critical
    inversion_choice = input("Is your image black text on a white background? (yes/no, default: yes): ").strip().lower()
    should_invert_colors = inversion_choice != 'no'

    print(f"Preprocessing image: {user_image_path} (Invert colors: {should_invert_colors})")
    custom_image_flat = preprocess_image(user_image_path, invert_colors=should_invert_colors)

    if custom_image_flat is not None:
        # Reshape for scaler and model: (1 sample, 784 features)
        custom_image_reshaped = custom_image_flat.reshape(1, -1)

        # --- SVM Prediction ---
        print("\n--- SVM Model Prediction ---")
        # Scale the image data using the *loaded* SVM scaler
        custom_image_scaled_svm = svm_scaler.transform(custom_image_reshaped)
        svm_prediction_numeric = svm_model.predict(custom_image_scaled_svm)[0]
        svm_char_prediction = get_char_from_label(svm_prediction_numeric, emnist_mapping) if emnist_mapping is not None else "N/A"
        print(f"SVM Predicted Label: {svm_prediction_numeric} ({svm_char_prediction})")

        # --- Random Forest Prediction ---
        print("\n--- Random Forest Model Prediction ---")
        # Scale the image data using the *loaded* RF scaler
        custom_image_scaled_rf = rf_scaler.transform(custom_image_reshaped)
        rf_prediction_numeric = rf_model.predict(custom_image_scaled_rf)[0]
        rf_char_prediction = get_char_from_label(rf_prediction_numeric, emnist_mapping) if emnist_mapping is not None else "N/A"
        print(f"Random Forest Predicted Label: {rf_prediction_numeric} ({rf_char_prediction})")

    else:
        print("Could not process the image.")