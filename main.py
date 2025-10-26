# Triggering the first build again
import functions_framework
import joblib
import pandas as pd
import json
import os

# --- Load Model and All Artifacts ---
# This code runs ONCE when the Cloud Function "cold starts".
# It loads the model from the file into memory.

model = None
artifacts = None
MODEL_FILE = 'model.joblib'
ARTIFACTS_FILE = 'model_artifacts.json'

def load_model():
    """Loads model and artifacts from disk."""
    global model, artifacts
    
    try:
        # These files (model.joblib, model_artifacts.json) 
        # must be in the same directory as this main.py file.
        model = joblib.load(MODEL_FILE)
        
        with open(ARTIFACTS_FILE, 'r') as f:
            artifacts = json.load(f)
            
        print("Model and artifacts loaded successfully.")
        return True
        
    except FileNotFoundError as e:
        print(f"ERROR: Missing file {e.filename}. Make sure model.joblib and model_artifacts.json are deployed.")
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Load the model when the function instance starts
load_model()

@functions_framework.http
def predict_telemetry(request):
    """
    HTTP Cloud Function to predict all 12 telemetry values
    based on a single 'raw_mill_load' input.
    """
    
    # --- CORS Headers (CRITICAL for Vercel) ---
    # Set CORS headers to allow requests from your Vercel app.
    # '*' is a wildcard, but for production, you should
    # use 'https://cementmindai.vercel.app'
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type'
    }

    # Handle 'preflight' (OPTIONS) requests sent by browsers
    if request.method == 'OPTIONS':
        return ('', 204, headers)

    # --- Model Check ---
    if model is None or artifacts is None:
        print("ERROR: Model is not loaded.")
        return ("Internal Server Error: Model not loaded", 500, headers)

    # --- Input Validation ---
    try:
        # Get request data (works for GET query params or POST JSON)
        if request.method == 'POST':
            request_data = request.get_json(silent=True)
        else: # Default to GET
            request_data = request.args

        if request_data is None:
             return ("Bad Request: Missing or invalid JSON payload.", 400, headers)

        # Get the specific input feature our model needs
        input_feature_name = artifacts['input_features'][0] # 'raw_mill_load'
        raw_mill_load_str = request_data.get(input_feature_name)

        if raw_mill_load_str is None:
            return (f"Bad Request: Missing '{input_feature_name}' parameter.", 400, headers)
            
        raw_mill_load_val = float(raw_mill_load_str)

    except (ValueError, TypeError):
        return (f"Bad Request: Invalid value for '{input_feature_name}'. Must be a number.", 400, headers)
    except Exception as e:
        print(f"Error parsing input: {e}")
        return ("Bad Request: Could not parse input.", 400, headers)

    # --- Prediction ---
    try:
        # 1. Create a DataFrame for the model (it expects a 2D array)
        input_df = pd.DataFrame([[raw_mill_load_val]], columns=artifacts['input_features'])
        
        # 2. Get prediction
        prediction_values = model.predict(input_df)
        
        # 3. Format as a dictionary
        # prediction_values is a 2D array (e.g., [[val1, val2, ...]]), so get the first row [0]
        output_features = artifacts['output_features']
        prediction_dict = dict(zip(output_features, prediction_values[0]))
        
        # 4. Round values to match the dashboard's precision
        for key, value in prediction_dict.items():
            if "vibration" in key:
                prediction_dict[key] = round(value, 2)
            elif "pressure" in key or "flow" in key:
                prediction_dict[key] = round(value, 0)
            else:
                # Temperatures, Currents, Load
                prediction_dict[key] = round(value, 1)
        
        # 5. Return as JSON
        # functions_framework automatically converts a dict to a JSON response
        return (prediction_dict, 200, headers)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return ("Internal Server Error: Prediction failed.", 500, headers)





