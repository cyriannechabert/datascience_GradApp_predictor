import streamlit as st
import pandas as pd
import joblib

# 1. Load the Model
# Ensure the .joblib file is in the same folder as this script
model = joblib.load("xgb_grad_admission_model_bootstrap.joblib")

# 2. App Title and Description
st.title("🎓 UCLA Grad School Admission Predictor")
st.write("Enter your scores below to see if UCLA is a Safe, Target, or Reach school for you.")

# 3. User Input Fields
col1, col2 = st.columns(2)

with col1:
    gre = st.number_input("GRE Score (290-340)", min_value=290, max_value=340, value=315)
    toefl = st.number_input("TOEFL Score (92-120)", min_value=92, max_value=120, value=105)
    univ_rating = st.selectbox("University Rating (1-5)", [1, 2, 3, 4, 5], index=2)
    research = st.radio("Do you have Research Experience?", ["No", "Yes"])

with col2:
    gpa = st.number_input("CGPA (1.0-4.0)", min_value=1.0, max_value=4.0, value=3.2)
    sop = st.slider("Statement of Purpose (SOP) Strength", 1.0, 5.0, 3.0, 0.5)
    lor = st.slider("Letter of Rec (LOR) Strength", 1.0, 5.0, 3.0, 0.5)

# Convert inputs to model format
research_val = 1 if research == "Yes" else 0

input_data = pd.DataFrame([{
    'GRE Score': gre,
    'TOEFL Score': toefl,
    'University Rating': univ_rating,
    'SOP': sop,
    'LOR ': lor, # Note the space if your training data had it
    'GPA': gpa,
    'Research': research_val
}])

# 4. Predict Button
if st.button("Predict Chance"):
    # Get Prediction
    pred_idx = model.predict(input_data)[0]
    classes = {0: "Reach 🔴", 1: "Medium 🟡", 2: "Safe 🟢"}
    result = classes[pred_idx]
    
    # Display Result
    st.subheader(f"Prediction: {result}")
    
    if pred_idx == 2:
        st.balloons()
        st.success("You have a very strong profile for UCLA!")
    elif pred_idx == 1:
        st.info("You are a competitive candidate. Focus on your SOP.")
    else:
        st.error("This is a Reach school. Consider improving your GRE/Research.")