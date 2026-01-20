import streamlit as st # streamlit cloud 
import numpy as np
import pickle

# -------------------------------
# Load trained AI model
# -------------------------------
with open("placement_model.pkl", "rb") as file:
    model = pickle.load(file)

# -------------------------------
# App Title & Description
# -------------------------------
st.set_page_config(page_title="Student Placement Predictor AI", layout="centered")

st.title("🎓 Student Placement Predictor AI")
st.write(
    "This AI application predicts whether a student is **likely to get placed** "
    "based on academic performance, skills, attendance, projects, and communication."
)

st.markdown("---")

# -------------------------------
# User Input Section
# -------------------------------
st.header("📌 Enter Student Details")

maths = st.slider("Maths Score", 0, 100, 70)
python = st.slider("Python Skill Score", 0, 100, 75)
sql = st.slider("SQL Skill Score", 0, 100, 65)
attendance = st.slider("Attendance Percentage", 0, 100, 80)
mini_projects = st.number_input("Number of Mini Projects", 0, 10, 2)
communication = st.slider("Communication Score", 0, 100, 70)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🔮 Predict Placement"):
    
    input_data = np.array([[
        maths,
        python,
        sql,
        attendance,
        mini_projects,
        communication
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.success("🎉 Prediction: Student is **LIKELY TO GET PLACED**")
    else:
        st.error("❌ Prediction: Student **NEEDS IMPROVEMENT**")

    st.info(f"📊 Placement Probability: **{probability * 100:.2f}%**")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with  using Python, Machine Learning & Streamlit")
