import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

artifact      = joblib.load("thyroid_svm_pipeline.pkl")
_preprocessor = artifact["preprocessor"]
_model        = artifact["model"]
BINARY_COLS   = artifact["binary_cols"]

st.set_page_config(page_title="Thyroid Cancer Risk", page_icon="🏥")
st.title("🏥 Thyroid Cancer Risk Prediction")
st.markdown("Fill in the patient details below and click **Predict**.")

col1, col2 = st.columns(2)

with col1:
    age     = st.number_input("Age", 1, 120, 45)
    gender  = st.selectbox("Gender", ["Male", "Female"])
    country = st.text_input("Country", "Ethiopia")
    ethnic  = st.text_input("Ethnicity", "Other")
    fhist   = st.selectbox("Family History", ["No", "Yes"])
    rad     = st.selectbox("Radiation Exposure", ["No", "Yes"])
    iodine  = st.selectbox("Iodine Deficiency", ["No", "Yes"])

with col2:
    smoke   = st.selectbox("Smoking", ["No", "Yes"])
    obesity = st.selectbox("Obesity", ["No", "Yes"])
    diab    = st.selectbox("Diabetes", ["No", "Yes"])
    tsh     = st.number_input("TSH Level (mIU/L)", 0.0, 50.0, 2.5)
    t3      = st.number_input("T3 Level (nmol/L)", 0.0, 10.0, 1.5)
    t4      = st.number_input("T4 Level (nmol/L)", 0.0, 30.0, 8.0)
    nodule  = st.number_input("Nodule Size (cm)", 0.0, 20.0, 1.0)

risk = st.selectbox("Thyroid Cancer Risk Assessment", ["Low", "Medium", "High"])

if st.button("🔍 Predict Diagnosis", use_container_width=True):
    raw = {
        "Age": age, "TSH_Level": tsh, "T3_Level": t3, "T4_Level": t4,
        "Nodule_Size": nodule, "Gender": gender, "Family_History": fhist,
        "Radiation_Exposure": rad, "Iodine_Deficiency": iodine,
        "Smoking": smoke, "Obesity": obesity, "Diabetes": diab,
        "Country": country, "Ethnicity": ethnic, "Thyroid_Cancer_Risk": risk
    }
    df = pd.DataFrame([raw])
    le = LabelEncoder()
    for col in BINARY_COLS:
        df[col] = le.fit_transform(df[col].astype(str))
    X     = _preprocessor.transform(df)
    pred  = _model.predict(X)[0]
    prob  = _model.predict_proba(X)[0].max()
    label = "Malignant" if pred == 1 else "Benign"

    st.markdown("---")
    if label == "Malignant":
        st.error(f"⚠️ Diagnosis: **{label}**  —  Confidence: **{prob*100:.1f}%**")
    else:
        st.success(f"✅ Diagnosis: **{label}**  —  Confidence: **{prob*100:.1f}%**")
    st.caption("⚕️ This is a decision-support tool only. Always confirm with clinical evaluation.")
