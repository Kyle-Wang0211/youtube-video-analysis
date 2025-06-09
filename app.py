import streamlit as st

# 设置网页配置
st.set_page_config(
    page_title="📊 YouTube Video Analysis APP",
    layout="wide"
)

# 页面主标题
st.markdown("""
    <h1 style='text-align: center;'>📊 YouTube Video Analysis App</h1>
""", unsafe_allow_html=True)

# 页面说明文字
st.markdown("""
    <p style='text-align: center;'>
    This platform uses a linear regression model to analyze YouTube video trends and predict popularity. <br>
    Use the sidebar on the left or the overview cards below to navigate to different functional pages.
    </p>
""", unsafe_allow_html=True)

# 添加可折叠导航模块（使用 st.expander 实现视觉折叠效果）
with st.expander("📚 Click here to view all sections", expanded=False):
    st.page_link(".streamlit/pages/1_Introduction.py", label="📘 01 Introduction")
    st.page_link(".streamlit/pages/2_Dataset_Visualization.py", label="📊 02 Dataset Visualization")
    st.page_link(".streamlit/pages/3_Model_Architecture.py", label="🧠 03 Model Architecture")
    st.page_link(".streamlit/pages/4_Prediction.py", label="🔮 04 Prediction")
    st.page_link(".streamlit/pages/5_Business_Prospects.py", label="📈 05 Business Prospects")

# 项目简介
st.markdown("""
---

### 🎯 Objective
This app aims to help users understand and apply linear regression in analyzing trends and predicting the popularity of YouTube videos. It simplifies complex concepts and enables intuitive interaction with the data.

### 💡 Motivation
- YouTube content creators, marketers, and analysts need effective tools to anticipate video performance.
- By predicting future popularity using historical metrics, stakeholders can make informed decisions on content planning, advertising, and engagement.
- Our app demonstrates the power of machine learning to support real-world decisions.

### 🛠️ Technologies Used
- **Python** & **Streamlit** for building the interface
- **Pandas**, **Seaborn**, and **Matplotlib** for data processing and visualization
- **Scikit-learn** for building and evaluating the linear regression model

### 🧪 Dataset
The dataset includes attributes such as view count, likes, comments, and video duration. These features are used to model and predict video popularity using linear regression.

---
""")

# 用户导航提示
st.info("📌 Use the left navigation menu to switch between different functional pages.")
