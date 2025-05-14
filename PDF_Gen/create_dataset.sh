#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Cause pipelines to return the exit status of the last command
# that exited with a non-zero status, or zero if all commands
# in the pipeline exit successfully.
set -o pipefail

# --- Configuration ---
NUM_ITERATIONS=1000
MAIN_FOLDER="../Datasets/Base"
JSON_GENERATOR_SCRIPT="json_generator.py"
JSON_TO_PDF_SCRIPT="json2pdf.py"

# --- Check if Python scripts exist ---
if [ ! -f "$JSON_GENERATOR_SCRIPT" ]; then
    echo "Error: $JSON_GENERATOR_SCRIPT not found in the current directory."
    exit 1
fi

if [ ! -f "$JSON_TO_PDF_SCRIPT" ]; then
    echo "Error: $JSON_TO_PDF_SCRIPT not found in the current directory."
    exit 1
fi

# --- Create Main Folder ---
echo "Creating main folder: $MAIN_FOLDER"
mkdir -p "$MAIN_FOLDER"

# --- Iteration Loop ---
echo "Starting $NUM_ITERATIONS iterations..."

for i in $(seq 1 $NUM_ITERATIONS)
do
    # --- Create Subfolder ---
    SUBFOLDER_PATH="$MAIN_FOLDER/$i"
    mkdir -p "$SUBFOLDER_PATH"
    echo "--- Iteration $i: Created subfolder $SUBFOLDER_PATH ---"

    # --- Generate Random Parameters ---
    # Random float between 0.1 and 0.5 (using awk)
    RANDOM_FLOAT=$(awk -v min=0.2 -v max=0.5 'BEGIN{srand(); print min+rand()*(max-min)}' | sed 's/,/./')

    # Random integer between 5 and 11 (using shuf)
    RANDOM_INT=$(shuf -i 5-11 -n 1)

    echo "Iteration $i: Random Float = $RANDOM_FLOAT, Random Int = $RANDOM_INT"

    # --- Run json_generator.py ---
    echo "Iteration $i: Running $JSON_GENERATOR_SCRIPT..."
    # Assuming json_generator.py takes the folder path, float, and int as arguments
    # Adjust the command if your script takes arguments differently (e.g., using flags like --path)
    python "$JSON_GENERATOR_SCRIPT" "$SUBFOLDER_PATH" --ratio "$RANDOM_FLOAT" --total_blocks "$RANDOM_INT"
    echo "Iteration $i: Finished $JSON_GENERATOR_SCRIPT."

    # --- Run json2pdf.py ---
    echo "Iteration $i: Running $JSON_TO_PDF_SCRIPT..."
    # Assuming json2pdf.py takes the folder path as an argument
    # Adjust the command if your script takes arguments differently
    python "$JSON_TO_PDF_SCRIPT" "$SUBFOLDER_PATH"
    echo "Iteration $i: Finished $JSON_TO_PDF_SCRIPT."

    echo "--- Iteration $i: Complete ---"
    echo # Add a blank line for readability
done

echo "All $NUM_ITERATIONS iterations completed successfully."
