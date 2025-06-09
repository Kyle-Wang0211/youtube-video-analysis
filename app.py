import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="YouTube Video Analysis", layout="wide")

st.title("📊 YouTube Video Analysis and Popularity Prediction")

uploaded_file = st.file_uploader("Upload your YouTube dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    st.subheader("📈 Basic Statistics")
    st.write(df.describe(include='all'))

    if 'views' in df.columns and 'likes' in df.columns:
        st.subheader("📊 Views vs Likes")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='views', y='likes', ax=ax)
        st.pyplot(fig)
