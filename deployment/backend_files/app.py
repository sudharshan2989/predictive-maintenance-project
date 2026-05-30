import joblib
import pandas as pd
from flask import Flask, request, jsonify
from huggingface_hub import hf_hub_download

# Initialize Flask app with a clean context name
engine_condition_api = Flask("EngineConditionPredictor")

# Load the trained engine condition prediction model
model_path = hf_hub_download(repo_id="sudharshanc/predictive-vehicle-maintenance", filename="best_predictive_vehicle_maintenance_model_v1.joblib")
model = joblib.load(model_path)


@engine_condition_api.get('/')
def home():
    return "Welcome to the Engine Condition Diagnostics API!"


@engine_condition_api.post('/v1/predict')
def predict_engine_condition():
    # Get raw sensor JSON array/object from the request
    sensor_data = request.get_json()

    # Convert incoming raw reading payload into a processing DataFrame
    df = pd.DataFrame([sensor_data])

    try:
        # Perform feature engineering calculations exactly as required by the pipeline
        df['lub oil temp K'] = df['lub oil temp'] + 273.15
        df['Coolant temp K'] = df['Coolant temp'] + 273.15

        # Compute systems health tracking ratio attributes
        df['Coolant_p_t_ratio'] = df['Coolant pressure'] / df['Coolant temp K']
        df['Lub_oil_p_t_ratio'] = df['Lub oil pressure'] / df['lub oil temp K']
        
        # Enforce exact inference feature structure layout and formatting order
        input_features = df[[
            'Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure',
            'lub oil temp K', 'Coolant temp K', 'Coolant_p_t_ratio', 'Lub_oil_p_t_ratio']]

        # Generate model evaluation output
        prediction = model.predict(input_features).tolist()[0]
        
        # Check if model supports standard structural confidence outputs
        try:
            probabilities = model.predict_proba(input_features).tolist()[0]
        except AttributeError:
            probabilities = None

        # Return the processed status evaluation metrics
        return jsonify({
            'Engine Condition Prediction': prediction,
            'Probabilities': probabilities,
            'Computed Diagnostics': {
                'Coolant_p_t_ratio': float(df['Coolant_p_t_ratio'].iloc[0]),
                'Lub_oil_p_t_ratio': float(df['Lub_oil_p_t_ratio'].iloc[0])
            }
        }), 200

    except KeyError as e:
        return jsonify({'error': f'Missing required input field sensor: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Run the API deployment framework setup
if __name__ == '__main__':
    engine_condition_api.run(debug=True)
