import streamlit as st

# 设置网页配置
st.set_page_config(
    page_title="📊 YouTube Video Analysis APP",
    layout="wide"
)

# 页面主标题
st.markdown("""
    <h1 style='text-align: center;'>📊 YouTube Video Analysis</h1>
""", unsafe_allow_html=True)

# 页面说明文字
st.markdown("""
    <p style='text-align: center;'>
    This platform uses a linear regression model to address real-world business or social problems. <br>
    Use the sidebar on the left or the overview buttons below to navigate to different functional pages:
    </p>
""", unsafe_allow_html=True)

# 概览按钮区块（模拟导航卡片）
st.markdown("---")
st.markdown("## 🧭 Navigation Overview")

# 采用五个列按钮，每行一个按钮
if st.button("🧭 Go to 01 Introduction"):
    st.switch_page("pages/01_Introduction.py")

if st.button("📊 Go to 02 Dataset Visualization"):
    st.switch_page("pages/02 Dataset Visualization.py")

if st.button("🧠 Go to 03 Metrics and Model Architecture"):
    st.switch_page("pages/03 Model Architecture.py")

if st.button("🔮 Go to 04 Prediction"):
    st.switch_page("pages/04_Prediction.py")

if st.button("📈 Go to 05 Business Prospects"):
    st.switch_page("pages/05_Business_Prospects.py")

# 提示用户操作
st.info("📌 You can also use the left navigation menu to switch between different functional pages.")
