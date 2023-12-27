import pandas as pd
import numpy as np
import streamlit as st
import pickle
import requests
from streamlit_lottie import st_lottie
import sklearn


st.set_page_config(page_title='Heart Disease Predictor',page_icon=':❤️‍🩹:')


# ----------------------------------------------------------
# DECORATION
# https://www.webfx.com/tools/emoji-cheat-sheet/
# https://lottiefiles.com/

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    else:
        return r.json()

lottie_coding = load_lottieurl('https://lottie.host/159cee15-ad6d-4408-bcdf-cbddc8c15c66/uXFKNy5NRe.json')
st_lottie(lottie_coding,height=300)



# -------------------------------------------------------------
# CODE
model = pickle.load(open('heart_disease_model.pkl', 'rb'))
data = pd.read_csv('heart_disease2_modified.csv')

st.title('Heart Stroke Predictor')

gender_mapping = {'Male': 1, 'Female': 0}
gender = st.selectbox('**Choose gender**', options=['Male', 'Female'])
gen = gender_mapping[gender]

age = st.number_input("**Enter Age**")

smoker_mapping = {'yes': 1, 'no': 0}
Current_smoker = st.selectbox("**Is Patient Current Smoker**", options=['yes', 'no'])
cs = smoker_mapping[Current_smoker]

cigsperday = st.number_input("**Enter Cigarettes Per Day**")

BPMeds_mapping = {'yes': 1, 'no': 0}
BPMeds = st.selectbox("**Is Patient on BP Medication**", options=['yes', 'no'])
bpm = BPMeds_mapping[BPMeds]

prevalentstroke_mapping = {'yes': 1, 'no': 0}
prevalentStroke = st.selectbox("**Did the patient have a stroke**", options=['yes', 'no'])
ps = prevalentstroke_mapping[prevalentStroke]

prevalenthyp_mapping = {'yes': 1, 'no': 0}
prevalenthyp = st.selectbox("**Prevalent hypertension Status**",options=['yes', 'no'])
ph = prevalenthyp_mapping[prevalenthyp]

diab_mapping = {'yes': 1, 'no': 0}
diabetes = st.selectbox("**Enter Diabetes Status**",options=['yes', 'no'])
diab = diab_mapping[diabetes]

totchol = st.number_input("**Enter Total Cholesterol**")
systBP = st.number_input("**Enter systolic blood pressure**")
diaBP = st.number_input("**Enter diastolic blood pressure**")
BMI = st.number_input("**Enter BMI**")
heartRate = st.number_input("**Enter Heart rate**")
glucose = st.number_input("**Enter Glucose**")
st.write("---")

if st.button('**Predict**'):
    input_data = np.array([[gen, age, cs, cigsperday, bpm, ps, ph, diab, totchol, systBP, diaBP, BMI, heartRate, glucose]])

    output = model.predict(input_data)
    if output[0] == 0:
        stn = '**Healthy**'
        message = "Great news! You are predicted to be at a lower risk of future heart stroke. Keep prioritizing your health!"
    else:
        stn = '**Not Healthy**'
        message = "Caution! You are predicted to be at a higher risk of future heart stroke. Please consult with a healthcare professional for personalized advice."

    st.subheader(f"**Prediction Result:**   {stn}")
    st.write(message)



