import streamlit as st

st.set_page_config(
    page_title="YouTube Video Analysis",
    page_icon="📊",
    layout="centered",
)

st.markdown(
    "<h1 style='text-align: center; color: #1f77b4;'>📊 YouTube Video Analysis and Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align: center; color: grey;'>Explore trends and predict popularity using machine learning</h3>",
    unsafe_allow_html=True
)

st.markdown("---")

model = st.selectbox(
    "Choose a model for prediction",
    ["Logistic Regression", "Random Forest", "XGBoost", "Neural Network"]
)

st.success(f"You selected: {model}")