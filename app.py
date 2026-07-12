
import streamlit as st
import pandas as pd

from src.predict import predict_wine

st.set_page_config(
    page_title="🍷 Red Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

# Sidebar
st.sidebar.title("🍷 Red Wine Quality")
st.sidebar.markdown("---")
st.sidebar.write("### Machine Learning Mini Project")
st.sidebar.write("**Algorithm:** Logistic Regression")
st.sidebar.write("**Dataset:** Red Wine Quality")
st.sidebar.write("**Accuracy:** 86.56%")
st.sidebar.markdown("---")
st.sidebar.success("Enter wine details and click Predict.")

# Main Title
st.title("🍷 Red Wine Quality Prediction System")
st.markdown(
    "This application predicts whether a **Red Wine** is **Good Quality** or **Bad Quality** using a Logistic Regression model."
)

st.markdown("---")

# Input columns
col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
    volatile_acidity = st.number_input("Volatile Acidity", value=0.70)
    citric_acid = st.number_input("Citric Acid", value=0.00)
    residual_sugar = st.number_input("Residual Sugar", value=1.9)
    chlorides = st.number_input("Chlorides", value=0.076)
    free_so2 = st.number_input("Free Sulfur Dioxide", value=11.0)

with col2:
    total_so2 = st.number_input("Total Sulfur Dioxide", value=34.0)
    density = st.number_input("Density", value=0.9978)
    ph = st.number_input("pH", value=3.51)
    sulphates = st.number_input("Sulphates", value=0.56)
    alcohol = st.number_input("Alcohol", value=9.4)

features = [
    fixed_acidity,
    volatile_acidity,
    citric_acid,
    residual_sugar,
    chlorides,
    free_so2,
    total_so2,
    density,
    ph,
    sulphates,
    alcohol
]

st.markdown("---")

if st.button("🔍 Predict Wine Quality", use_container_width=True):

    prediction, confidence = predict_wine(features)

    st.subheader("Prediction")

    if prediction == 1:
        st.success(f"🍷 Good Quality Wine\n\nConfidence: {confidence:.2f}%")
    else:
        st.error(f"🍷 Bad Quality Wine\n\nConfidence: {confidence:.2f}%")

    st.markdown("---")

    st.subheader("Input Values")

    df = pd.DataFrame({
        "Feature": [
            "Fixed Acidity",
            "Volatile Acidity",
            "Citric Acid",
            "Residual Sugar",
            "Chlorides",
            "Free SO₂",
            "Total SO₂",
            "Density",
            "pH",
            "Sulphates",
            "Alcohol"
        ],
        "Value": features
    })

    st.dataframe(df, use_container_width=True)

    st.subheader("Feature Visualization")

    chart_df = df.set_index("Feature")
    st.bar_chart(chart_df)

with st.expander("📖 About this Project"):

    st.write("""
    **Project Name**
    - Red Wine Quality Prediction

    **Machine Learning Algorithm**
    - Logistic Regression

    **Dataset**
    - Red Wine Quality Dataset

    **Objective**
    - Predict whether a wine is Good or Bad based on its physicochemical properties.

    **Developed Using**
    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    """)

st.markdown("---")
st.caption("Developed for Mini Project using Python & Streamlit")
