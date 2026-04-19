import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# ── Column definitions ────────────────────────────────────────
NUMERIC_COLS    = ["Age", "TSH_Level", "T3_Level", "T4_Level", "Nodule_Size"]
BINARY_COLS     = ["Gender", "Family_History", "Radiation_Exposure",
                   "Iodine_Deficiency", "Smoking", "Obesity", "Diabetes"]
MULTICLASS_COLS = ["Country", "Ethnicity", "Thyroid_Cancer_Risk"]

@st.cache_resource
def train_model():
    df = pd.read_csv("thyroid_cancer_risk_data.csv")
    df = df.drop(columns=["Patient_ID"])

    X = df.drop(columns=["Diagnosis"])
    y = df["Diagnosis"].map({"Benign": 0, "Malignant": 1})

    le = LabelEncoder()
    for col in BINARY_COLS:
        X[col] = le.fit_transform(X[col].astype(str))

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer,     NUMERIC_COLS),
        ("cat", categorical_transformer, MULTICLASS_COLS),
        ("bin", "passthrough",           BINARY_COLS)
    ])

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    X_train_proc = preprocessor.fit_transform(X_train)

    model = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000, random_state=42), cv=3)
    model.fit(X_train_proc, y_train)

    # store country/ethnicity options
    countries   = sorted(df["Country"].dropna().unique().tolist())
    ethnicities = sorted(df["Ethnicity"].dropna().unique().tolist())

    return preprocessor, model, countries, ethnicities

# ── Load / train ──────────────────────────────────────────────
with st.spinner("⏳ Loading model (first run trains it — takes ~30 seconds)..."):
    preprocessor, model, countries, ethnicities = train_model()

# ── UI ────────────────────────────────────────────────────────
st.set_page_config(page_title="Thyroid Cancer Risk", page_icon="🏥")
st.title("🏥 Thyroid Cancer Risk Prediction")
st.markdown("Fill in the patient details and click **Predict**.")

col1, col2 = st.columns(2)

with col1:
    age     = st.number_input("Age", 1, 120, 45)
    gender  = st.selectbox("Gender", ["Male", "Female"])
    country = st.selectbox("Country", countries)
    ethnic  = st.selectbox("Ethnicity", ethnicities)
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
    input_df = pd.DataFrame([raw])
    le = LabelEncoder()
    for col in BINARY_COLS:
        input_df[col] = le.fit_transform(input_df[col].astype(str))

    X_input = preprocessor.transform(input_df)
    pred    = model.predict(X_input)[0]
    prob    = model.predict_proba(X_input)[0].max()
    label   = "Malignant" if pred == 1 else "Benign"

    st.markdown("---")
    if label == "Malignant":
        st.error(f"⚠️ Diagnosis: **{label}**  —  Confidence: **{prob*100:.1f}%**")
    else:
        st.success(f"✅ Diagnosis: **{label}**  —  Confidence: **{prob*100:.1f}%**")
    st.caption("⚕️ Decision-support tool only. Always confirm with clinical evaluation.")
