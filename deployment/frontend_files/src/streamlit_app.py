import streamlit as st
import requests

# Define backend endpoint URL
API_URL = "https://sudharshanc-predictive-vehicle-maintenance-backend.hf.space/v1/predict"

# Streamlit UI Header Config
st.set_page_config(page_title="Engine Diagnostics UI", page_icon="⚙️", layout="centered")
st.title("Engine Condition Classification & Diagnostics")
st.write("""
This interface collects real-time sensor metrics and updates diagnostic assessments 
by querying the predictive vehicle maintenance API microservice.
""")

# Input Sidebar/Form for Core Sensor Parameters
st.subheader("Engine Sensor Metrics")

col1, col2 = st.columns(2)

with col1:
    engine_rpm = st.number_input("Engine RPM", min_value=0, max_value=10000, value=1500, step=50)
    fuel_pressure = st.number_input("Fuel Pressure", min_value=0.0, max_value=100.0, value=5.1, step=0.1)
    lub_oil_pressure = st.number_input("Lubricating Oil Pressure", min_value=0.0, max_value=100.0, value=4.25, step=0.1)
    lub_oil_temp_c = st.number_input("Lubricating Oil Temp (°C)", min_value=-50.0, max_value=300.0, value=82.4, step=0.5)

with col2:
    coolant_pressure = st.number_input("Coolant Pressure", min_value=0.0, max_value=100.0, value=3.65, step=0.1)
    coolant_temp_c = st.number_input("Coolant Temp (°C)", min_value=-50.0, max_value=300.0, value=87.8, step=0.5)

# Build the exact raw request payload schema expected by your API backend
payload = {
    "Engine rpm": int(engine_rpm),
    "Lub oil pressure": float(lub_oil_pressure),
    "Fuel pressure": float(fuel_pressure),
    "Coolant pressure": float(coolant_pressure),
    "lub oil temp": float(lub_oil_temp_c),
    "Coolant temp": float(coolant_temp_c)
}

# Classification & Network Trigger Task
if st.button("Predict Engine Condition", type="primary"):
    with st.spinner("Querying API backend..."):
        try:
            # Dispatch the payload to the Hugging Face space endpoint
            response = requests.post(API_URL, json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                
                # Read prediction as a direct integer, do not try to index it with [0]
                prediction = result.get("Engine Condition Prediction")
                if isinstance(prediction, list):
                    prediction = prediction[0]
                
                probabilities = result.get("Probabilities")

                # Safely calculate model confidence output mapping using the nested float index
                prob_str = ""
                if isinstance(probabilities, list) and len(probabilities) > 0:
                    # If probabilities is a list of lists [[p0, p1]], flatten it first
                    if isinstance(probabilities[0], list):
                        probabilities = probabilities[0]
                    
                    # Pull confidence based on whether the model predicted 0 or 1
                    conf = probabilities[1] if prediction == 1 else probabilities[0]
                    prob_str = f" (Confidence: {conf:.2%})"

                # Render Diagnostic Output State Cards
                st.subheader("Diagnostic Assessment Result:")
                if prediction == 1:
                    st.error(f"🔴 Warning: Anomalous Engine Condition/Fault detected{prob_str}.")
                else:
                    st.success(f"🟢 Normal/Healthy Engine Condition detected{prob_str}.")
                    
            else:
                err_msg = response.json().get('error', response.text)
                st.error(f"⚠️ Pipeline Processing Exception ({response.status_code}): {err_msg}")
                
        except requests.exceptions.Timeout:
            st.error("⏱️ API Connection Timed Out. Your Hugging Face backend Space might be sleeping or spinning up.")
        except requests.exceptions.RequestException as e:
            st.error(f"🔌 Failed to communicate with backend microservice: {str(e)}")
