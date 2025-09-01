import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64


st.title("❤ Heart Disease Predictor")
tab1,tab2,tab3 = st.tabs(['Predict', 'Bulk Predict', 'Model Information'])

with tab1:
    age = st.number_input("Age (year)", min_value = 0, max_value = 150)
    sex = st.selectbox("Sex", ['Male', 'Female'])
    chest_pain = st.number_input("Resting Blood Pressure (mmHg)", min_value = 0, max_value = 300)
