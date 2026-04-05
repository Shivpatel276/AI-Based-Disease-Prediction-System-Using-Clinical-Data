import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64
import plotly.express as px
import google.generativeai as genai

def get_binary_file_download_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert to string
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions as CSV</a>'
    return href

st.title("Heart Disease Prediction App")
tab1, tab2, tab3,tab4=st.tabs(["Home","For CSV","Model Information","AI"])

with tab1:
    st.header("Enter Patient Data")
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    
    sex = st.selectbox("Sex", ["Male", "Female"])
    sex = 0 if sex == "Male" else 1

    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    chest_pain = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}[chest_pain]

    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
    
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    fasting_bs = 0 if fasting_bs == "<= 120 mg/dl" else 1

    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
    resting_ecg = {"Normal": 0, "ST": 1, "LVH": 2}[resting_ecg]

    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    exercise_angina = 0 if exercise_angina == "No" else 1

    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Up", "Flat", "Down"])
    st_slope = {"Up": 0, "Flat": 1, "Down": 2}[st_slope]
    
    input_data = pd.DataFrame({
            'Age':            [age],
            'Sex':            [sex],
            'ChestPainType':  [chest_pain],
            'RestingBP':      [resting_bp],
            'Cholesterol':    [cholesterol],
            'FastingBS':      [fasting_bs],
            'RestingECG':     [resting_ecg],
            'MaxHR':          [max_hr],
            'ExerciseAngina': [exercise_angina],
            'Oldpeak':        [oldpeak],
            'ST_Slope':       [st_slope]
        })

    algorithms = ['Logistic Regression', 'SVM', 'Decision Tree', 'Random Forest']
    modelnames = ['logistic.pkl', 'svm_model.pkl', 'DecisionTree.pkl', 'random_forest.pkl']

    predictions = []  

    def predict_heart_disease(data):
        predictions = []  
        for modelname in modelnames:  
            model = pickle.load(open(modelname, 'rb'))
            prediction = model.predict(data)
            predictions.append(prediction)  # ✅ append to list, not to itself
        return predictions

    if st.button("Predict"):
        st.subheader("Prediction Results:")
        st.markdown("--------------------------------------------------")
        
        result = predict_heart_disease(input_data)  
        
        for i in range(len(result)):  # ✅ loop over result not prediction
            st.subheader(algorithms[i])
            if result[i][0] == 0:
                st.write("The model predicts that the patient has no heart disease.")
            else:
                st.write("The model predicts that the patient has heart disease.")
            st.markdown("--------------------------------------------------")

with tab2:
    st.title("Upload CSV File")
    
    st.subheader("Instructions to note before uploading the file:")
    st.info("""
    1. No NaN values allowed.
    2. Total 11 features in this order: 'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
    3. Check the spellings of the feature names.
    4. Feature values conventions:
        - Age: age of the patient [years]
        - Sex: [0: Male, 1: Female]
        - ChestPainType: [0: Atypical Angina, 1: Non-Anginal Pain, 2: Asymptomatic, 3: Typical Angina]
        - RestingBP: resting blood pressure [mm Hg]
        - Cholesterol: serum cholesterol [mm/dl]
        - FastingBS: [1: if FastingBS > 120 mg/dl, 0: otherwise]
        - RestingECG: [0: Normal, 1: ST-T wave abnormality, 2: Left Ventricular Hypertrophy]
        - MaxHR: maximum heart rate achieved [60 to 202]
        - ExerciseAngina: [1: Yes, 0: No]
        - Oldpeak: ST depression [Numeric]
        - ST_Slope: [0: Upsloping, 1: Flat, 2: Downsloping]
    """)
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        
        model = pickle.load(open('logistic.pkl', 'rb'))
        
        expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                            'Oldpeak', 'ST_Slope']
        
        if set(expected_columns).issubset(input_data.columns):
            input_data['prediction LR'] = ''
            
            for i in range(len(input_data)):
                arr = input_data[expected_columns].iloc[i].values  # ✅ use only 11 columns
                input_data['prediction LR'][i] = model.predict([arr])[0]
            
            input_data.to_csv('predictedHeartLR.csv')
            
            st.subheader("Predictions:")
            st.write(input_data)
            st.markdown(get_binary_file_download_html(input_data), unsafe_allow_html=True)
        else:
            st.warning("The uploaded CSV file does not contain the required columns. Please check the instructions and try again.")
    else:
        st.warning("Please upload a CSV file to get predictions.")
        
with tab3:
    
    data = { 'decision tree': 86.41, 'random forest': 88.04, 'logistic regression': 86.95, 'SVM': 86.33 }
    models = list(data.keys())
    accuracies = list(data.values())    
    df = pd.DataFrame(list(zip(models, accuracies)), columns=['Model', 'Accuracy'])
    fig = px.bar(df, x='Model', y='Accuracy', title='Model Accuracies')
    st.plotly_chart(fig)
    
    
with tab4:
    st.title(" AI Health Assistant")
    st.write("Describe your symptoms and get AI-powered health insights.")

    genai.configure(api_key="AIzaSyCDSikk4DY0HBIAE67qJ26llasGhCQERCU")
    model_ai = genai.GenerativeModel("gemini-2.5-flash") 

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    user_input = st.chat_input("Describe your symptoms here...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        st.session_state.chat_history.append({"role": "user", "content": user_input})

        prompt = f"""You are a helpful medical AI assistant.
            A patient is describing their symptoms. Provide helpful health insights, possible causes, 
            and recommend whether they should see a doctor.
            Always remind them that you are an AI and they should consult a real doctor for proper diagnosis.

            Patient symptoms: {user_input}"""

        with st.spinner("AI is thinking..."):
            response = model_ai.generate_content(prompt)
            ai_response = response.text

        with st.chat_message("assistant"):
            st.write(ai_response)

        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()