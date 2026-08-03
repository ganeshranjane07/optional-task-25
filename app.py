
import streamlit as st
import joblib
import numpy as np

model = joblib.load("best_model.pkl")

st.title("📚 Student Performance Prediction")

hours = st.number_input("Hours Studied", value=5)
attendance = st.number_input("Attendance", value=80)
parent = st.number_input("Parental Involvement", value=2)
resources = st.number_input("Access to Resources", value=2)
previous = st.number_input("Previous Scores", value=70)
motivation = st.number_input("Motivation Level", value=2)
internet = st.number_input("Internet Access", value=1)
tutoring = st.number_input("Tutoring Sessions", value=2)
income = st.number_input("Family Income", value=2)
teacher = st.number_input("Teacher Quality", value=2)
school = st.number_input("School Type", value=1)
peer = st.number_input("Peer Influence", value=2)
physical = st.number_input("Physical Activity", value=3)
learning = st.number_input("Learning Disabilities", value=0)
gender = st.number_input("Gender", value=1)

if st.button("Predict"):

    data = np.array([[
        hours,
        attendance,
        parent,
        resources,
        previous,
        motivation,
        internet,
        tutoring,
        income,
        teacher,
        school,
        peer,
        physical,
        learning,
        gender
    ]])

    prediction = model.predict(data)

    st.success(f"Predicted Exam Score: {prediction[0]:.2f}")
