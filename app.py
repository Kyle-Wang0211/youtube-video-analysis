import streamlit as st

# 设置页面配置
st.set_page_config(
    page_title="📊 YouTube Video Analysis App",
    page_icon="📊",
    layout="wide"
)

# 页面标题与说明
st.markdown("# 📊 YouTube Video Analysis App")
st.markdown(
    "This platform uses a linear regression model to analyze YouTube video trends and predict popularity. "
    "Use the sidebar on the left to navigate to different functional pages."
)

# 显示每个子页面功能说明
st.markdown("## 📚 Contents")

# 子页面卡片介绍（非跳转，仅说明用途）
st.markdown("### 📘 01 Introduction")
st.markdown("- Learn about the background, goals, and usage of this project.")

st.markdown("### 📊 02 Dataset Visualization")
st.markdown("- Explore the dataset through visualizations to identify patterns and trends.")

st.markdown("### 🧠 03 Model Architecture")
st.markdown("- Understand the model's structure, features used, and evaluation metrics.")

st.markdown("### 🔮 04 Prediction")
st.markdown("- Input new data to get predictions and interpret outcomes.")

st.markdown("### 📈 05 Business Prospects")
st.markdown("- Discuss potential business or social insights based on model outputs.")
