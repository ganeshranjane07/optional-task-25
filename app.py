import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor

# Load Dataset
df = pd.read_csv("StudentPerformanceFactors.csv")

# Fill Missing Values
df.fillna(df.mode().iloc[0], inplace=True)

# Encode Categorical Columns
le = LabelEncoder()
for col in df.select_dtypes(include="object"):
    df[col] = le.fit_transform(df[col])

# Features & Target
X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]

# Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🎓 Student Performance Prediction")
st.write("Enter Student Details")

hours = st.number_input("Hours Studied", 0, 24, 5)
attendance = st.number_input("Attendance", 0, 100, 80)
parent = st.number_input("Parental Involvement", 0, 2, 1)
resources = st.number_input("Access to Resources", 0, 2, 1)
extra = st.number_input("Extracurricular Activities", 0, 1, 1)
sleep = st.number_input("Sleep Hours", 0, 12, 7)
previous = st.number_input("Previous Scores", 0, 100, 70)
motivation = st.number_input("Motivation Level", 0, 2, 1)
internet = st.number_input("Internet Access", 0, 1, 1)
tutoring = st.number_input("Tutoring Sessions", 0, 10, 2)
income = st.number_input("Family Income", 0, 2, 1)
teacher = st.number_input("Teacher Quality", 0, 2, 1)
school = st.number_input("School Type", 0, 1, 1)
peer = st.number_input("Peer Influence", 0, 2, 1)
physical = st.number_input("Physical Activity", 0, 10, 3)
learning = st.number_input("Learning Disabilities", 0, 1, 0)
parent_edu = st.number_input("Parental Education Level", 0, 2, 1)
distance = st.number_input("Distance from Home", 0, 2, 1)
gender = st.number_input("Gender", 0, 1, 1)

if st.button("Predict"):

    data = np.array([[
        hours,
        attendance,
        parent,
        resources,
        extra,
        sleep,
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
        parent_edu,
        distance,
        gender
    ]])

    prediction = model.predict(data)

    st.success(f"Predicted Exam Score: {prediction[0]:.2f}")