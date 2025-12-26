import streamlit as st
import numpy as np
import joblib

# Load model and scaler (CORRECT)
model = joblib.load("sale.pk1")     # LogisticRegression
scaler = joblib.load("scaler.pk1")  # StandardScaler

st.set_page_config(page_title="Region Prediction", layout="centered")
st.title("🌍 Region Prediction App")

# -------- User Inputs --------
value1 = st.number_input("Value1", min_value=0.0)
value2 = st.number_input("Value2", min_value=0.0)
score = st.number_input("Score", min_value=0)
rating = st.number_input("Rating (1–5)", min_value=1, max_value=5)
sales = st.number_input("Sales", min_value=0.0)
quantity = st.number_input("Quantity", min_value=0)
profit = st.number_input("Profit", value=0.0)

# -------- Prediction --------
if st.button("Predict Region"):
    input_data = np.array([[value1, value2, score, rating, sales, quantity, profit]])

    # ✅ ONLY scaler uses transform
    input_scaled = scaler.transform(input_data)

    # ✅ Model uses predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.success(f"Predicted Region: {prediction}")
    st.info(
        f"Confidence → Region 0: {probability[0]:.2f}, "
        f"Region 1: {probability[1]:.2f}"
    )


#st.write("Model:", type(model))
#st.write("Scaler:", type(scaler))