import streamlit as st

# 设置网页的标题与图标
st.set_page_config(
    page_title="📊 YouTube Video Analysis App",
    page_icon="📊",
    layout="wide"
)

# 页面标题
st.markdown("""
    <h1 style='text-align: center;'>📊 YouTube Video Analysis App</h1>
""", unsafe_allow_html=True)

# 页面简介
st.markdown("""
This platform uses a linear regression model to address real-world business or social problems through the analysis of YouTube video datasets.
Use the sidebar on the left or click on the overview cards below to navigate to each functional page:
""")

# 页面功能卡片（从上到下排列，每个卡片都带跳转链接提示）
st.markdown("""
---

### 🔍 01 Introduction
介绍项目背景、目标与用途，帮助你快速了解本应用。请点击左侧导航栏中的 "01 Introduction" 进入。

---

### 📊 02 Dataset Visualization
可视化数据特征，发现变量之间的趋势与模式。请点击左侧导航栏中的 "02 Dataset Visualization" 进入。

---

### 🧠 03 Model Architecture
说明模型的构建方式与所用指标，帮助理解其预测逻辑。请点击左侧导航栏中的 "03 Model Architecture" 进入。

---

### 🔮 04 Prediction
输入特征，获得模型预测结果，用于实际决策模拟。请点击左侧导航栏中的 "04 Prediction" 进入。

---

### 📈 05 Business Prospects
结合模型输出，讨论预测结果对业务的潜在意义。请点击左侧导航栏中的 "05 Business Prospects" 进入。

---

📌 请使用左侧导航菜单，在各功能页面间自由切换。
""")
