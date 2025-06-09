import streamlit as st
import pandas as pd

# 页面配置
st.set_page_config(
    page_title="YouTube Video Analysis",
    page_icon="📊",
    layout="centered"
)

# 顶部导航栏样式（仿 Hugging Face）
with st.container():
    col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 2, 2])
    with col1:
        st.markdown("🧑‍💻 **Spaces**", unsafe_allow_html=True)
    with col2:
        st.markdown("📁 **NYU-DS-4-Everyone / face**", unsafe_allow_html=True)
    with col3:
        st.markdown("🟢 **Running**", unsafe_allow_html=True)
    with col4:
        st.markdown("🍭 **Community**", unsafe_allow_html=True)
    with col5:
        st.markdown("⚙️ **Settings**", unsafe_allow_html=True)
st.markdown("---")

# 标题与副标题
st.markdown(
    "<h1 style='text-align: center; color: #1f77b4;'>📊 YouTube Video Analysis and Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center; color: grey;'>Explore trends and predict popularity using machine learning</h3>",
    unsafe_allow_html=True
)
st.markdown("---")

# 模型选择
model = st.selectbox(
    "Choose a model for prediction",
    ["Logistic Regression", "Random Forest", "XGBoost", "Neural Network"]
)
st.success(f"You selected: {model}")

# 上传数据
uploaded_file = st.file_uploader("Upload your YouTube dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df.head())
