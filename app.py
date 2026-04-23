import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64
import plotly.express as px
import google.generativeai as genai
import os
from dotenv import load_dotenv

with open("style.css",encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def get_binary_file_download_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions as CSV</a>'
    return href


def range_indicator(value, low, high, unit="", reverse=False):
    """Show a colored badge based on whether value is in normal range."""
    if value == 0:
        return  # skip if user hasn't entered a value yet
    if low <= value <= high:
        status = "🟢 Normal"
        color = "green"
    elif (value < low and not reverse) or (value > high and reverse):
        status = "🟡 Below Normal" if value < low else "🟡 Above Normal"
        color = "orange"
    else:
        status = "🔴 High Risk Range"
        color = "red"
    st.markdown(f"<small style='color:{color}'>{status} &nbsp;|&nbsp; Normal range: {low}–{high} {unit}</small>", unsafe_allow_html=True)


st.title("❤️ Heart Disease Prediction App")
tab1, tab2, tab3, tab4 = st.tabs(["Home", "For CSV", "Model Information", "AI"])

# ─────────────────────────────────────────────
# TAB 1 — Manual Input with tooltips, expanders, range badges
# ─────────────────────────────────────────────
with tab1:
    st.header("Enter Patient Data")
    st.info("💡 Not sure about a value? Click the **ℹ️ Learn more** expander under each field for guidance.")

    # ── Age ──
    age = st.number_input(
        "Age (years)",
        min_value=0, max_value=150,
        help="Enter the patient's age in years. Heart disease risk increases with age, especially after 45 for men and 55 for women."
    )

    # ── Sex ──
    sex_label = st.selectbox(
        "Sex",
        ["Male", "Female"],
        help="Biological sex affects heart disease risk. Men tend to develop it earlier than women."
    )
    sex = 0 if sex_label == "Male" else 1

    # ── Chest Pain ──
    chest_pain_label = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "ASY", "TA"],
        help="Type of chest pain experienced. ASY (Asymptomatic) is the most common in heart disease patients."
    )
    with st.expander("ℹ️ What do these chest pain types mean?"):
        st.markdown("""
        | Code | Full Name | Description |
        |------|-----------|-------------|
        | **ATA** | Atypical Angina | Chest pain that doesn't fit the classic heart pattern |
        | **NAP** | Non-Anginal Pain | Chest pain unrelated to the heart (e.g. muscle, acid reflux) |
        | **ASY** | Asymptomatic | No chest pain at all — surprisingly common in heart disease |
        | **TA** | Typical Angina | Classic heart-related chest pain: pressure/squeezing, triggered by exertion |
        
        *If you're unsure, select **ASY** (no chest pain) if the patient hasn't reported any.*
        """)
    chest_pain = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}[chest_pain_label]

    # ── Resting Blood Pressure ──
    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=0, max_value=300,
        help="Blood pressure measured while at rest. Normal is around 120 mm Hg. High BP is a major heart disease risk factor."
    )
    with st.expander("ℹ️ How do I find my Resting Blood Pressure?"):
        st.markdown("""
        - Found on any **blood pressure monitor** reading (the top/systolic number)
        - Available from a recent **doctor visit** or **pharmacy checkup**
        - **Normal:** 90–120 mm Hg &nbsp;|&nbsp; **Elevated:** 120–129 &nbsp;|&nbsp; **High:** 130+
        
        *If unknown, a common average for adults is **120 mm Hg**.*
        """)
    range_indicator(resting_bp, 90, 120, "mm Hg")

    # ── Cholesterol ──
    cholesterol = st.number_input(
        "Serum Cholesterol (mm/dl)",
        min_value=0,
        help="Total cholesterol level from a blood test. High cholesterol increases plaque buildup in arteries."
    )
    with st.expander("ℹ️ What is Serum Cholesterol and where do I find it?"):
        st.markdown("""
        - Found on a **lipid panel / blood test** report — look for *"Total Cholesterol"*
        - **Desirable:** below 200 mg/dl
        - **Borderline High:** 200–239 mg/dl
        - **High:** 240+ mg/dl
        
        *If you don't have a test result, ask your doctor or enter **200** as a neutral default.*
        """)
    range_indicator(cholesterol, 0, 200, "mg/dl")

    # ── Fasting Blood Sugar ──
    fasting_bs_label = st.selectbox(
        "Fasting Blood Sugar",
        ["<= 120 mg/dl", "> 120 mg/dl"],
        help="Blood sugar level after fasting for 8+ hours. Values above 120 mg/dl may indicate diabetes, a heart disease risk factor."
    )
    with st.expander("ℹ️ What is Fasting Blood Sugar?"):
        st.markdown("""
        - Measured after **not eating for at least 8 hours** (usually a morning blood test)
        - **Normal:** 70–100 mg/dl &nbsp;|&nbsp; **Pre-diabetic:** 100–125 &nbsp;|&nbsp; **Diabetic:** 126+
        - Found on a **routine blood test** or **diabetes screening report**
        
        *If you haven't had a test, select **<= 120 mg/dl** as the safer default.*
        """)
    fasting_bs = 0 if fasting_bs_label == "<= 120 mg/dl" else 1

    # ── Resting ECG ──
    resting_ecg_label = st.selectbox(
        "Resting ECG Results",
        ["Normal", "ST", "LVH"],
        help="Result of an electrocardiogram taken at rest. Abnormal results can signal heart stress or structural changes."
    )
    with st.expander("ℹ️ What is a Resting ECG and what do the results mean?"):
        st.markdown("""
        An **ECG (Electrocardiogram)** records the electrical activity of your heart.
        
        | Result | Meaning |
        |--------|---------|
        | **Normal** | No abnormalities detected |
        | **ST** | ST-T wave abnormality — possible sign of coronary artery disease |
        | **LVH** | Left Ventricular Hypertrophy — thickening of the heart's main pumping chamber |
        
        *If you haven't had an ECG, select **Normal**. You can get one at any cardiology clinic.*
        """)
    resting_ecg = {"Normal": 0, "ST": 1, "LVH": 2}[resting_ecg_label]

    # ── Max Heart Rate ──
    max_hr = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=60, max_value=202,
        help="The highest heart rate recorded during physical exertion (e.g. stress test or intense exercise). A lower max HR can indicate reduced heart capacity."
    )
    with st.expander("ℹ️ How do I find my Maximum Heart Rate?"):
        st.markdown("""
        - Ideally measured during a **cardiac stress test** at a hospital
        - A rough estimate: **220 minus your age** (e.g. age 50 → ~170 bpm)
        - You can also check a fitness tracker during intense exercise
        
        *Use the formula **220 − age** as a reasonable estimate if unsure.*
        """)
    if age > 0:
        estimated_max = 220 - age
        range_indicator(max_hr, estimated_max - 20, estimated_max + 10, "bpm")

    # ── Exercise-Induced Angina ──
    exercise_angina_label = st.selectbox(
        "Exercise-Induced Angina",
        ["No", "Yes"],
        help="Do you experience chest pain or tightness during physical activity? This is a classic sign of reduced blood flow to the heart."
    )
    with st.expander("ℹ️ What is Exercise-Induced Angina?"):
        st.markdown("""
        **Angina** is chest discomfort caused by reduced blood flow to the heart.
        
        - **Exercise-induced** means it appears *during* physical activity and goes away with rest
        - Symptoms: chest tightness, pressure, pain, or shortness of breath while exercising
        - It's different from a heart attack — it's temporary and usually stops with rest
        
        *If you've never noticed chest pain during exercise, select **No**.*
        """)
    exercise_angina = 0 if exercise_angina_label == "No" else 1

    # ── Oldpeak ──
    oldpeak = st.number_input(
        "Oldpeak (ST Depression)",
        min_value=0.0, max_value=10.0,
        help="The degree of ST segment depression on an ECG during exercise compared to rest. Higher values suggest more stress on the heart."
    )
    with st.expander("ℹ️ What is Oldpeak / ST Depression?"):
        st.markdown("""
        - This is a measurement from an **ECG during a stress test**
        - It reflects how much the ST segment drops below the baseline during exercise
        - **Normal:** 0.0 &nbsp;|&nbsp; **Mild concern:** 0.5–1.5 &nbsp;|&nbsp; **High concern:** > 2.0
        
        *If you haven't had a stress test, enter **0.0** as the default (no depression detected).*
        """)
    if oldpeak > 0:
        if oldpeak <= 1.0:
            st.markdown("<small style='color:green'>🟢 Mild — Low concern</small>", unsafe_allow_html=True)
        elif oldpeak <= 2.0:
            st.markdown("<small style='color:orange'>🟡 Moderate — Worth monitoring</small>", unsafe_allow_html=True)
        else:
            st.markdown("<small style='color:red'>🔴 High — Consult a doctor</small>", unsafe_allow_html=True)

    # ── ST Slope ──
    st_slope_label = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        ["Up", "Flat", "Down"],
        help="The shape of the ST segment on an ECG at peak exercise. A downsloping or flat pattern is more associated with heart disease."
    )
    with st.expander("ℹ️ What is the ST Slope?"):
        st.markdown("""
        This describes the **direction of the ST wave** on an ECG during peak exercise:
        
        | Slope | Meaning |
        |-------|---------|
        | **Up** (Upsloping) | Generally normal, less concerning |
        | **Flat** | Borderline — can indicate mild ischemia |
        | **Down** (Downsloping) | Most concerning, associated with heart disease |
        
        *This value comes from a **stress ECG test**. If unavailable, select **Up** as the most common normal finding.*
        """)
    st_slope = {"Up": 0, "Flat": 1, "Down": 2}[st_slope_label]

    # ── Build input DataFrame ──
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

    @st.cache_resource
    def load_models():
        models = {}
        for name, file in zip(algorithms, modelnames):
            try:
                models[name] = pickle.load(open(file, 'rb'))
            except FileNotFoundError:
                models[name] = None
        return models

    def predict_heart_disease(data):
        loaded = load_models()
        predictions = []
        for name in algorithms:
            model = loaded.get(name)
            if model:
                predictions.append(model.predict(data))
            else:
                predictions.append(None)
        return predictions

    if st.button("🔍 Predict"):
        st.subheader("Prediction Results:")
        st.markdown("---")
        result = predict_heart_disease(input_data)
        positive_count = 0

        for i in range(len(result)):
            st.subheader(algorithms[i])
            if result[i] is None:
                st.warning(f"Model file for {algorithms[i]} not found.")
            elif result[i][0] == 0:
                st.success("✅ No heart disease detected.")
            else:
                st.error("⚠️ Heart disease detected.")
                positive_count += 1
            st.markdown("---")

        # Voting summary
        st.subheader("🗳️ Overall Consensus")
        valid_results = [r for r in result if r is not None]
        if valid_results:
            if positive_count >= len(valid_results) / 2:
                st.error(f"⚠️ {positive_count}/{len(valid_results)} models predict **heart disease**. Please consult a doctor.")
            else:
                st.success(f"✅ {len(valid_results) - positive_count}/{len(valid_results)} models predict **no heart disease**.")


# ─────────────────────────────────────────────
# TAB 2 — CSV Upload (unchanged)
# ─────────────────────────────────────────────
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

        expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
                            'Oldpeak', 'ST_Slope']

        # Check 1: Required columns exist
        if not set(expected_columns).issubset(input_data.columns):
            st.warning("The uploaded CSV file does not contain the required columns. Please check the instructions and try again.")

        # Check 2: No NaN values
        elif input_data[expected_columns].isnull().any().any():
            st.warning("The uploaded CSV file contains NaN values. Please clean your data and try again.")

        else:
            # Load model
            model = pickle.load(open('logistic.pkl', 'rb'))

            # Vectorized prediction (no loop)
            input_data['Prediction LR'] = model.predict(input_data[expected_columns])

            # Map prediction values to labels
            input_data['Prediction LR'] = input_data['Prediction LR'].map({0: 'No Disease', 1: 'Heart Disease'})

            # Convert to CSV in memory for download
            csv_bytes = input_data.to_csv(index=False).encode('utf-8')

            st.subheader("Predictions:")
            st.write(input_data)

            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=csv_bytes,
                file_name="predictedHeartLR.csv",
                mime="text/csv"
            )
    else:
        st.warning("Please upload a CSV file to get predictions.")

# ─────────────────────────────────────────────
# TAB 3 — Model Information (unchanged)
# ─────────────────────────────────────────────
with tab3:
    data = {'decision tree': 86.41, 'random forest': 88.04, 'logistic regression': 86.95, 'SVM': 86.33}
    models = list(data.keys())
    accuracies = list(data.values())
    df = pd.DataFrame(list(zip(models, accuracies)), columns=['Model', 'Accuracy'])
    fig = px.bar(df, x='Model', y='Accuracy', title='Model Accuracies')
    st.plotly_chart(fig)


# ─────────────────────────────────────────────
# TAB 4 — AI Health Assistant (unchanged)
# ─────────────────────────────────────────────
with tab4:
    load_dotenv()
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    st.title("🤖 AI Health Assistant")
    st.write("Describe your symptoms and get AI-powered health insights.")
    model_ai = genai.GenerativeModel("gemini-2.5-flash")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

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