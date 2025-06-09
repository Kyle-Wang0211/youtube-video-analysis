import streamlit as st

# 设置页面配置
st.set_page_config(
    page_title="📊 YouTube Video Analysis App",
    layout="wide"
)

# 页面主标题
st.markdown("<h1 style='text-align: center;'>📊 YouTube Video Analysis App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>This platform uses a linear regression model to analyze YouTube video trends and predict popularity.<br>Use the menu below to navigate between functional modules.</p>", unsafe_allow_html=True)

# 选择框导航
section = st.selectbox("📚 Contents", [
    "01 Introduction",
    "02 Dataset Visualization",
    "03 Model Architecture",
    "04 Prediction",
    "05 Business Prospects"
])

# 加载不同子页面内容
if section == "01 Introduction":
    st.header("📘 Introduction")
    st.markdown("""
    This section explains the background, motivation, and overall goal of the app.  
    Linear regression is a widely used technique in data analysis to uncover relationships between variables.  
    In this case, it helps us model how YouTube video features affect view counts or popularity.
    """)

elif section == "02 Dataset Visualization":
    st.header("📊 Dataset Visualization")
    st.markdown("""
    Visualize variables such as view count, likes, duration, and comments.  
    Use scatter plots and heatmaps to uncover data trends.
    """)

elif section == "03 Model Architecture":
    st.header("🧠 Model Architecture")
    st.markdown("""
    The linear regression model is built using scikit-learn.  
    We preprocess the dataset, split into training/testing sets, and evaluate using R² and MSE.
    """)

elif section == "04 Prediction":
    st.header("🔮 Prediction")
    st.markdown("""
    You can input custom values (e.g. video duration, like count) to predict expected views.
    """)

elif section == "05 Business Prospects":
    st.header("📈 Business Prospects")
    st.markdown("""
    Reflect on how prediction results can help content creators make data-driven decisions.  
    For instance, determine optimal video length, timing, or engagement strategy.
    """)

# 页脚提示
st.info("📌 Use the dropdown above to view each section. You are currently viewing: **" + section + "**")


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


# 提示用户操作
st.info("📌 Use the left navigation menu to switch between different functional pages.")
