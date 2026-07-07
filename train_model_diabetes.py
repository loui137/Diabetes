import pandas as pd 

d1=pd.read_csv('Diabetes.csv')

d2=d1.loc[(d1['Glucose']!=0) & (d1['BloodPressure']!=0) & 
           (d1['SkinThickness']!=0) & (d1['Insulin']!=0) & (d1['BMI']!=0)]


d1['Glucose'].replace(0,d2['Glucose'].mean(),inplace=True)
d1['BloodPressure'].replace(0,d2['BloodPressure'].mean(),inplace=True)
d1['SkinThickness'].replace(0,d2['SkinThickness'].mean(),inplace=True)
d1['Insulin'].replace(0,d2['Insulin'].mean(),inplace=True)
d1['BMI'].replace(0, d2['BMI'].mean(), inplace = True)

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

st.title("Regenesys Diabetes Prediction Application")
st.write(
    "This application uses a **Decision Tree classification model** "
    "to predict whether a patient is **Diabetic or Not Diabetic** "
    "based on medical input parameters."
)

X = d1.drop("Outcome", axis=1)   # Features
y = d1["Outcome"]               # Target (0 = Not Diabetic, 1 = Diabetic)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(random_state=477)
model.fit(X_train, y_train)


st.subheader("Enter Patient Details")

# Accepting input from the user using streamlit widgets 
pregnancies = st.number_input("Pregnancies", min_value=0, value=1)
glucose = st.number_input("Glucose Level", min_value=0, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, value=20)
insulin = st.number_input("Insulin Level", min_value=0, value=80)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
age = st.number_input("Age", min_value=1, value=30)


# As soon as we click the button the input data gets stored as an array  
if st.button("Predict Diabetes"):

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][prediction]

    if prediction == 1:
        st.error(f"Prediction: **Diabetic** ({probability:.2f} confidence)")
    else:
        st.success(f"Prediction: **Not Diabetic** ({probability:.2f} confidence)")









