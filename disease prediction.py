# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 10:49:08 2025
@author: chamo
"""

import os
import pickle
import streamlit as st
import numpy as np
import joblib
from streamlit_option_menu import option_menu
import logging
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Cache model loading
@st.cache_resource
def load_model(file_path, use_joblib=False):
    try:
        if use_joblib:
            return joblib.load(file_path)
        return pickle.load(open(file_path, 'rb'))
    except FileNotFoundError:
        st.error(f"Model file not found at: {file_path}")
        logging.error(f"Model file not found: {file_path}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        logging.error(f"Error loading model: {str(e)}")
        st.stop()

# Getting the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Define model file paths
diabetes_model_path = os.path.join(working_dir, "models", "diabetes_model.sav")
heart_disease_model_path = os.path.join(working_dir, "models", "heart_disease_model.sav")
stroke_model_path = os.path.join(working_dir, "models", "stroke.sav")
scaler_path = os.path.join(working_dir, "models", "scaler.sav")

# Load models
diabetes_model = load_model(diabetes_model_path)
heart_disease_model = load_model(heart_disease_model_path)
stroke_model = load_model(stroke_model_path, use_joblib=True)
scaler = load_model(scaler_path, use_joblib=True)

# Input validation ranges
VALIDATION_RANGES = {
    'Pregnancies': (0, 20), 'Glucose': (0, 200), 'BloodPressure': (0, 200),
    'SkinThickness': (0, 100), 'Insulin': (0, 900), 'BMI': (0, 70),
    'DiabetesPedigreeFunction': (0, 2.5), 'Age': (0, 120),
    'SerumCholesterol': (0, 600), 'MaxHeartRate': (60, 220),
    'STDepression': (0, 6), 'MajorVessels': (0, 4), 'ChestPain': (0, 3),
    'RestingECG': (0, 2), 'Slope': (0, 2), 'Thal': (0, 3)
}

# Helper function for input validation
def validate_input(value, field, min_val, max_val):
    try:
        val = float(value)
        if min_val <= val <= max_val:
            return True
        st.error(f"{field} must be between {min_val} and {max_val}")
        return False
    except ValueError:
        st.error(f"Please enter a valid numerical value for {field}")
        return False

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                          ['Diabetes Prediction', 'Heart Disease Prediction', 'Stroke Prediction'],
                          menu_icon='hospital-fill',
                          icons=['activity', 'heart', 'clipboard2-pulse-fill'],
                          default_index=0)

# Initialize session state for input persistence
if 'inputs' not in st.session_state:
    st.session_state.inputs = {}

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    st.markdown("Enter the following details to predict diabetes risk. All fields are required.")

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies', value=st.session_state.inputs.get('Pregnancies', '0'), 
                                   help="Number of times pregnant (0-20)")
    with col2:
        Glucose = st.text_input('Glucose Level', value=st.session_state.inputs.get('Glucose', '0'),
                               help="Plasma glucose concentration (0-200 mg/dL)")
    with col3:
        BloodPressure = st.text_input('Blood Pressure', value=st.session_state.inputs.get('BloodPressure', '0'),
                                     help="Diastolic blood pressure (0-200 mmHg)")
    with col1:
        SkinThickness = st.text_input('Skin Thickness', value=st.session_state.inputs.get('SkinThickness', '0'),
                                     help="Triceps skin fold thickness (0-100 mm)")
    with col2:
        Insulin = st.text_input('Insulin Level', value=st.session_state.inputs.get('Insulin', '0'),
                               help="2-Hour serum insulin (0-900 mu U/ml)")
    with col3:
        BMI = st.text_input('BMI', value=st.session_state.inputs.get('BMI', '0'),
                           help="Body mass index (0-70)")
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function', 
                                               value=st.session_state.inputs.get('DiabetesPedigreeFunction', '0'),
                                               help="Diabetes pedigree function (0-2.5)")
    with col2:
        Age = st.text_input('Age', value=st.session_state.inputs.get('Age', '0'),
                           help="Age in years (0-120)")

    # Save inputs to session state
    st.session_state.inputs.update({
        'Pregnancies': Pregnancies, 'Glucose': Glucose, 'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness, 'Insulin': Insulin, 'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction, 'Age': Age
    })

    # Reset button
    if st.button('Reset Inputs'):
        st.session_state.inputs = {}
        st.rerun()

    if st.button('Diabetes Test Result'):
        with st.spinner('Predicting...'):
            try:
                # Validate inputs
                inputs_valid = all([
                    validate_input(Pregnancies, 'Pregnancies', *VALIDATION_RANGES['Pregnancies']),
                    validate_input(Glucose, 'Glucose', *VALIDATION_RANGES['Glucose']),
                    validate_input(BloodPressure, 'Blood Pressure', *VALIDATION_RANGES['BloodPressure']),
                    validate_input(SkinThickness, 'Skin Thickness', *VALIDATION_RANGES['SkinThickness']),
                    validate_input(Insulin, 'Insulin', *VALIDATION_RANGES['Insulin']),
                    validate_input(BMI, 'BMI', *VALIDATION_RANGES['BMI']),
                    validate_input(DiabetesPedigreeFunction, 'Diabetes Pedigree Function', 
                                 *VALIDATION_RANGES['DiabetesPedigreeFunction']),
                    validate_input(Age, 'Age', *VALIDATION_RANGES['Age'])
                ])

                if inputs_valid:
                    user_input = [float(x) for x in [Pregnancies, Glucose, BloodPressure, SkinThickness, 
                                                    Insulin, BMI, DiabetesPedigreeFunction, Age]]
                    diab_prediction = diabetes_model.predict([user_input])
                    diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
                    st.success(diab_diagnosis)
                    logging.info(f"Diabetes prediction: {diab_diagnosis}")
                    
                    # Export result
                    result_df = pd.DataFrame([user_input], columns=['Pregnancies', 'Glucose', 'BloodPressure', 
                                                                  'SkinThickness', 'Insulin', 'BMI', 
                                                                  'DiabetesPedigreeFunction', 'Age'])
                    result_df['Prediction'] = diab_diagnosis
                    csv = result_df.to_csv(index=False)
                    st.download_button('Download Result', csv, f'diabetes_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                logging.error(f"Diabetes prediction error: {str(e)}")

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    st.markdown("Enter the following details to predict heart disease risk. All fields are required.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age', value=st.session_state.inputs.get('heart_age', '0'),
                           help="Age in years (0-120)")
        sex = st.text_input('Sex (0 = female, 1 = male)', value=st.session_state.inputs.get('sex', '0'),
                           help="0 for female, 1 for male")
        cp = st.text_input('Chest Pain Type (0-3)', value=st.session_state.inputs.get('cp', '0'),
                          help="0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic")
    with col2:
        trestbps = st.text_input('Resting Blood Pressure', value=st.session_state.inputs.get('trestbps', '0'),
                                help="Resting blood pressure (0-200 mmHg)")
        chol = st.text_input('Serum Cholesterol', value=st.session_state.inputs.get('chol', '0'),
                            help="Serum cholesterol in mg/dl (0-600)")
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (0 = false, 1 = true)', 
                           value=st.session_state.inputs.get('fbs', '0'),
                           help="0 for false, 1 for true")
    with col3:
        restecg = st.text_input('Resting ECG Results (0-2)', value=st.session_state.inputs.get('restecg', '0'),
                               help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy")
        thalach = st.text_input('Maximum Heart Rate', value=st.session_state.inputs.get('thalach', '0'),
                               help="Maximum heart rate achieved (60-220)")
        exang = st.text_input('Exercise Induced Angina (0 = no, 1 = yes)', 
                             value=st.session_state.inputs.get('exang', '0'),
                             help="0 for no, 1 for yes")
    with col1:
        oldpeak = st.text_input('ST Depression', value=st.session_state.inputs.get('oldpeak', '0'),
                               help="ST depression induced by exercise (0-6)")
        slope = st.text_input('Slope of Peak ST Segment (0-2)', value=st.session_state.inputs.get('slope', '0'),
                             help="0: Upsloping, 1: Flat, 2: Downsloping")
        ca = st.text_input('Major Vessels (0-3)', value=st.session_state.inputs.get('ca', '0'),
                          help="Number of major vessels colored by fluoroscopy (0-3)")
    with col2:
        thal = st.text_input('Thal (0-3)', value=st.session_state.inputs.get('thal', '0'),
                            help="0: Normal, 1: Fixed defect, 2: Reversible defect, 3: Not described")

    # Save inputs to session state
    st.session_state.inputs.update({
        'heart_age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
        'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
        'ca': ca, 'thal': thal
    })

    # Reset button
    if st.button('Reset Inputs'):
        st.session_state.inputs = {}
        st.rerun()

    if st.button('Heart Disease Test Result'):
        with st.spinner('Predicting...'):
            try:
                # Validate inputs
                inputs_valid = all([
                    validate_input(age, 'Age', *VALIDATION_RANGES['Age']),
                    validate_input(sex, 'Sex', 0, 1),
                    validate_input(cp, 'Chest Pain Type', *VALIDATION_RANGES['ChestPain']),
                    validate_input(trestbps, 'Resting Blood Pressure', *VALIDATION_RANGES['BloodPressure']),
                    validate_input(chol, 'Serum Cholesterol', *VALIDATION_RANGES['SerumCholesterol']),
                    validate_input(fbs, 'Fasting Blood Sugar', 0, 1),
                    validate_input(restecg, 'Resting ECG Results', *VALIDATION_RANGES['RestingECG']),
                    validate_input(thalach, 'Maximum Heart Rate', *VALIDATION_RANGES['MaxHeartRate']),
                    validate_input(exang, 'Exercise Induced Angina', 0, 1),
                    validate_input(oldpeak, 'ST Depression', *VALIDATION_RANGES['STDepression']),
                    validate_input(slope, 'Slope', *VALIDATION_RANGES['Slope']),
                    validate_input(ca, 'Major Vessels', *VALIDATION_RANGES['MajorVessels']),
                    validate_input(thal, 'Thal', *VALIDATION_RANGES['Thal'])
                ])

                if inputs_valid:
                    user_input = [float(x) for x in [age, sex, cp, trestbps, chol, fbs, restecg, 
                                                    thalach, exang, oldpeak, slope, ca, thal]]
                    heart_prediction = heart_disease_model.predict([user_input])
                    heart_diagnosis = ('The person is having heart disease' if heart_prediction[0] == 1 
                                     else 'The person does not have any heart disease')
                    st.success(heart_diagnosis)
                    logging.info(f"Heart disease prediction: {heart_diagnosis}")

                    # Export result
                    result_df = pd.DataFrame([user_input], columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 
                                                                  'Cholesterol', 'FastingBS', 'RestingECG', 
                                                                  'MaxHR', 'ExerciseAngina', 'Oldpeak', 
                                                                  'ST_Slope', 'MajorVessels', 'Thal'])
                    result_df['Prediction'] = heart_diagnosis
                    csv = result_df.to_csv(index=False)
                    st.download_button('Download Result', csv, f'heart_disease_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                logging.error(f"Heart disease prediction error: {str(e)}")

# Stroke Prediction Page
if selected == 'Stroke Prediction':
    st.title('Stroke Prediction using ML')
    st.markdown("Enter the following details to predict stroke risk. All fields are required.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age', value=st.session_state.inputs.get('stroke_age', '0'),
                           help="Age in years (0-120)")
        hypertension = st.selectbox('Hypertension', ['No', 'Yes'], 
                                   index=1 if st.session_state.inputs.get('hypertension', 'No') == 'Yes' else 0,
                                   help="Select if patient has hypertension")
        heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'],
                                    index=1 if st.session_state.inputs.get('heart_disease', 'No') == 'Yes' else 0,
                                    help="Select if patient has heart disease")
    with col2:
        avg_glucose_level = st.text_input('Average Glucose Level', 
                                        value=st.session_state.inputs.get('avg_glucose_level', '0'),
                                        help="Average glucose level (0-200 mg/dL)")
        bmi = st.text_input('BMI', value=st.session_state.inputs.get('bmi', '0'),
                           help="Body mass index (0-70)")
        gender = st.selectbox('Gender', ['Male', 'Female', 'Other'],
                             index=['Male', 'Female', 'Other'].index(st.session_state.inputs.get('gender', 'Male')),
                             help="Select gender")
    with col3:
        ever_married = st.selectbox('Ever Married', ['No', 'Yes'],
                                   index=1 if st.session_state.inputs.get('ever_married', 'No') == 'Yes' else 0,
                                   help="Select marital status")
        work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'],
                                index=['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'].index(
                                    st.session_state.inputs.get('work_type', 'Private')),
                                help="Select work type")
        Residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'],
                                     index=1 if st.session_state.inputs.get('Residence_type', 'Urban') == 'Rural' else 0,
                                     help="Select residence type")
        smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'],
                                     index=['formerly smoked', 'never smoked', 'smokes', 'Unknown'].index(
                                         st.session_state.inputs.get('smoking_status', 'Unknown')),
                                     help="Select smoking status")

    # Save inputs to session state
    st.session_state.inputs.update({
        'stroke_age': age, 'hypertension': hypertension, 'heart_disease': heart_disease,
        'avg_glucose_level': avg_glucose_level, 'bmi': bmi, 'gender': gender,
        'ever_married': ever_married, 'work_type': work_type, 'Residence_type': Residence_type,
        'smoking_status': smoking_status
    })

    # Reset button
    if st.button('Reset Inputs'):
        st.session_state.inputs = {}
        st.rerun()

    if st.button('Stroke Test Result'):
        with st.spinner('Predicting...'):
            try:
                # Validate numerical inputs
                inputs_valid = all([
                    validate_input(age, 'Age', *VALIDATION_RANGES['Age']),
                    validate_input(avg_glucose_level, 'Average Glucose Level', *VALIDATION_RANGES['Glucose']),
                    validate_input(bmi, 'BMI', *VALIDATION_RANGES['BMI'])
                ])

                if inputs_valid:
                    numerical_input = [float(age), 1 if hypertension == 'Yes' else 0, 
                                     1 if heart_disease == 'Yes' else 0, float(avg_glucose_level), float(bmi)]
                    categorical_input = [
                        1 if gender == 'Male' else 0,
                        1 if gender == 'Other' else 0,
                        1 if ever_married == 'Yes' else 0,
                        1 if work_type == 'Private' else 0,
                        1 if work_type == 'Self-employed' else 0,
                        1 if work_type == 'children' else 0,
                        1 if work_type == 'Never_worked' else 0,
                        1 if Residence_type == 'Urban' else 0,
                        1 if smoking_status == 'never smoked' else 0,
                        1 if smoking_status == 'smokes' else 0,
                        1 if smoking_status == 'Unknown' else 0
                    ]
                    user_input = numerical_input + categorical_input
                    user_input = np.asarray(user_input).reshape(1, -1)
                    user_input_scaled = scaler.transform(user_input)
                    stroke_prediction = stroke_model.predict(user_input_scaled)
                    stroke_diagnosis = ('The person is at risk of stroke' if stroke_prediction[0] == 1 
                                      else 'The person is not at risk of stroke')
                    st.success(stroke_diagnosis)
                    logging.info(f"Stroke prediction: {stroke_diagnosis}")

                    # Export result
                    result_df = pd.DataFrame([numerical_input + categorical_input], 
                                            columns=['Age', 'Hypertension', 'Heart_Disease', 'Avg_Glucose_Level', 'BMI',
                                                    'Gender_Male', 'Gender_Other', 'Ever_Married_Yes', 
                                                    'Work_Type_Private', 'Work_Type_Self-employed', 
                                                    'Work_Type_children', 'Work_Type_Never_worked', 
                                                    'Residence_Type_Urban', 'Smoking_Status_never_smoked',
                                                    'Smoking_Status_smokes', 'Smoking_Status_Unknown'])
                    result_df['Prediction'] = stroke_diagnosis
                    csv = result_df.to_csv(index=False)
                    st.download_button('Download Result', csv, f'stroke_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                logging.error(f"Stroke prediction error: {str(e)}")