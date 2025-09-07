import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64


def get_binary_file_downloader_html(df):
    csv = df.to_csv(index = False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href = "data:file/csv;base64,{b64}" download = "Predictions.csv"> Download Predictions CSV </a>'
    return href



st.title("❤ Heart Disease Predictor")
tab1, tab2, tab3 = st.tabs(['🔮 Predict', '📦 Bulk Predict', '📊 Model Information'])

with tab1:
    # User inputs
    age = st.number_input("Age (year)", min_value=0, max_value=150)
    sex = st.selectbox("Sex", ['Male', 'Female'])
    chest_pain = st.selectbox("Resting Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope_option = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

    # Convert categorical inputs to numeric
    sex = 0 if sex == 'Male' else 1
    chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope_option)

    # Create DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    # Models and algorithm names (ensure same length)
    algorithms = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'Support Vector Machine', 'Grid Search']
    models_name = ['DecisionTree.pkl', 'LogisticRegression.pkl', 'RandomForest.pkl', 'SVM.pkl', 'Gridrf.pkl']

    # Prediction function
    def predict_heart_disease(df):
        predictions = []
        for model_name in models_name:
            with open(model_name, 'rb') as f:
                model = pickle.load(f)
            prediction = model.predict(df)
            predictions.append(prediction)
        return predictions

    # Submit button
    if st.button("Submit"):
        st.subheader("Results ...........")
        st.markdown('--------------------')

        results = predict_heart_disease(input_data)

        for i in range(len(results)):
            st.subheader(algorithms[i])
            if results[i][0] == 0:
                st.write("No Heart Disease Detected")
            else:
                st.write("Heart Disease Detected")
            st.markdown('--------------------')


with tab2:
    st.title("Upload CSV File")

    st.header("Instructions To Note Before Uploading The File") 
    st.info('''
        1. No NaN Values allowed.
        2. Total of 11 Features is required in this order ('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope').\n
        3. Check the spelling of the feature names.\n
        4. Features value conventions: \n
            - Age: age of the patients in [years] \n
            - Sex: gender of the patient [0: Male, 1: Female] \n
            - ChestPainType: [0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal Pain, 3: Asymptomatic] \n
            - RestingBP: resting blood pressure [mm Hg] \n
            - Cholesterol: serum cholesterol [mm/dl] \n
            - FastingBS: fasting blood sugar [1: if fastingBS > 120 mg/dl, 0: otherwise] \n
            - RestingECG: [0: Normal, 1: ST-T wave abnormality, 2: Left Ventricular Hypertrophy] \n
            - MaxHR: maximum heart rate achieved \n
            - ExerciseAngina: [1: Yes, 0: No] \n
            - Oldpeak: ST depression \n
            - ST_Slope: [0: Upsloping, 1: Flat, 2: Downsloping] \n
    ''')

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)

        # Models and algorithm names (must match filenames in your project)
        algorithms = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'Support Vector Machine', 'Grid Search']
        models_name = ['DecisionTree.pkl', 'LogisticRegression.pkl', 'RandomForest.pkl', 'SVM.pkl', 'Gridrf.pkl']

        expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
                            'Oldpeak', 'ST_Slope']

        if set(expected_columns).issubset(input_data.columns):
            # Loop through each model and add predictions as a new column
            for i, model_file in enumerate(models_name):
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)

                col_name = f"Predictions_{algorithms[i].replace(' ', '_')}"
                input_data[col_name] = ''

                for j in range(len(input_data)):
                    arr = input_data.iloc[j].values
                    input_data.loc[j, col_name] = model.predict([arr])[0]

            st.subheader("Predictions:")
            st.write(input_data)

            st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please Make Sure The Uploaded CSV File Has The Correct Columns.")
    else:
        st.info("⬆️ Upload a CSV File To Get Predictions")


with tab3:
    import matplotlib.pyplot as plt
import seaborn as sns

with tab3:
    # Model accuracy data
    data = {
        'Decision Tree': 80.98,
        'Logistic Regression': 85.87,
        'Random Forest': 84.24,
        'Support Vector Machine': 84.24,
        'Grid Search': 86.15
    }

    models = list(data.keys())
    accuracies = list(data.values())

    # Create DataFrame
    df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracies
    })

    # Seaborn bar plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Model", y="Accuracy", data=df, palette="viridis")

    # Add accuracy values on top of bars
    for index, value in enumerate(df['Accuracy']):
        plt.text(index, value + 0.5, f"{value:.2f}%", ha='center', fontsize=10, weight='bold')

    # Customize chart
    plt.ylim(0, 100)
    plt.title("Model Accuracy Comparison", fontsize=14, weight='bold')
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=30, ha='right')

    # Display chart in Streamlit
    st.pyplot(plt)

    # Show table below chart
    st.subheader("📋 Accuracy Table")
    st.dataframe(df)

