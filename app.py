import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")

# ------------------- LOAD DATA & MODEL -------------------
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'model.pkl' is in the app folder.")
        st.stop()

data = load_data()
model = load_model()

# ------------------- NAVIGATION -------------------
menu = st.radio(
    "Navigation",
    ["Data Exploration", "Visualization", "Model Prediction", "Model Performance"],
    horizontal=True
)

# ------------------- TITLE & INTRO -------------------
st.title("Titanic Survival Prediction")
st.write("An interactive web app to explore Titanic passenger data and predict survival outcomes.")

# ------------------- DATA EXPLORATION -------------------
if menu == "Data Exploration":
    st.subheader("Dataset Overview")
    st.write(f"Shape: {data.shape}")
    st.write(f"Columns: {list(data.columns)}")

    st.write("Sample Data:")
    st.dataframe(data.head())

    # Filters
    st.subheader("Filter Data")
    pclass_filter = st.selectbox("Passenger Class", options=[0, 1, 2, 3], index=0)
    sex_filter = st.selectbox("Sex", options=["all", "male", "female"], index=0)

    filtered_data = data.copy()
    if pclass_filter != 0:
        filtered_data = filtered_data[filtered_data["Pclass"] == pclass_filter]
    if sex_filter != "all":
        filtered_data = filtered_data[filtered_data["Sex"] == sex_filter]

    st.write("Filtered Data:")
    st.dataframe(filtered_data)

# ------------------- VISUALIZATION -------------------
elif menu == "Visualization":
    st.subheader("Survival Count by Passenger Class")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Pclass", hue="Survived", data=data, ax=ax1)
    ax1.set_title("Survival Count by Class")
    st.pyplot(fig1)

    st.subheader("Age Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(data["Age"].dropna(), bins=30, kde=True, ax=ax2)
    ax2.set_title("Age Distribution")
    st.pyplot(fig2)

    st.subheader("Fare vs Age (Survival)")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x="Fare", y="Age", hue="Survived", data=data, ax=ax3)
    ax3.set_title("Fare vs Age (Colored by Survival)")
    st.pyplot(fig3)

# ------------------- MODEL PREDICTION -------------------
elif menu == "Model Prediction":
    st.subheader("Enter Passenger Information")

    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.number_input("Age", 0, 100, 30)  # Changed from slider to number_input
        sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
    with col2:
        parch = st.number_input("Parents/Children aboard", 0, 10, 0)
        fare = st.number_input("Fare", 0.0, 500.0, 32.0)
        embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    # Encoding
    sex_male = 1 if sex == "male" else 0
    embarked_Q = 1 if embarked == "Q" else 0
    embarked_S = 1 if embarked == "S" else 0

    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Sex_male": [sex_male],
        "Embarked_Q": [embarked_Q],
        "Embarked_S": [embarked_S]
    })


    if st.button("Predict Survival"):
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][prediction]

        if prediction == 1:
            st.success(f"Survived ✅ (Confidence: {prediction_proba:.2f})")
        else:
            st.error(f"Did not survive ❌ (Confidence: {prediction_proba:.2f})")

# ------------------- MODEL PERFORMANCE -------------------
elif menu == "Model Performance":
    st.subheader("Random Forest Accuracy")
    rf_accuracy = 0.8212
    st.metric("Accuracy", f"{rf_accuracy*100:.2f}%")

    st.subheader("Confusion Matrix")
    rf_conf_matrix = [[91, 14], [18, 56]]
    fig4, ax4 = plt.subplots()
    sns.heatmap(rf_conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax4)
    ax4.set_xlabel("Predicted")
    ax4.set_ylabel("Actual")
    st.pyplot(fig4)

st.markdown("---")
st.caption("Titanic Survival Prediction App | Streamlit")
