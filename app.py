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

# 添加可折叠导航模块
with st.expander("📚 Click here to view all sections", expanded=False):
    st.markdown("📘 [01 Introduction](01_Introduction)")
    st.markdown("📊 [02 Dataset Visualization](02_Dataset_Visualization)")
    st.markdown("🧠 [03 Model Architecture](03_Model_Architecture)")
    st.markdown("🔮 [04 Prediction](04_Prediction)")
    st.markdown("📈 [05 Business Prospects](05_Business_Prospects)")

# 概览卡片区块（采用列布局）
with st.container():
    # 创建五行一列的布局
    st.markdown("---")
    st.markdown("## 🧭 Contents")

    # 第一页
    st.markdown("""
    ### 📘 01 Introduction
    Learn about the background, goals, and usage of this project.
    """)

    # 第二页
    st.markdown("""
    ### 📊 02 Dataset Visualization
    Explore the dataset through visualizations to identify patterns and trends.
    """)

    # 第三页
    st.markdown("""
    ### 🧠 03 Model Architecture
    Understand the model's structure, features used, and evaluation metrics.
    """)

    # 第四页
    st.markdown("""
    ### 🔮 04 Prediction
    Input new data to get predictions and interpret outcomes.
    """)

    # 第五页
    st.markdown("""
    ### 📈 05 Business Prospects
    Discuss potential business or social insights based on model outputs.
    """)

# 提示用户操作
st.info("📌 Use the left navigation menu to switch between different functional pages.")
