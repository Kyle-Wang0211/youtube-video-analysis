import streamlit as st

# 页面基本配置
st.set_page_config(
    page_title="Linear Regression Business App",
    page_icon="💼",
    layout="wide"
)

# 欢迎标题
st.markdown("<h1 style='text-align: center;'>💼 Linear Regression Business App</h1>", unsafe_allow_html=True)

# 简要说明
st.markdown("### 👋 欢迎使用本应用！")
st.write(
    """
    本平台旨在通过线性回归模型，解决现实中的业务或社会问题。请通过左侧导航栏，或点击下方模块介绍卡片，访问不同功能页面：
    """
)

# 分五列展示每个页面模块
col1, col2, col3 = st.columns(3)
col4, col5 = st.columns(2)

with col1:
    st.markdown("### 🧭 01 Introduction")
    st.write("介绍项目背景、目标与用途，帮助你快速了解本应用。")

with col2:
    st.markdown("### 📊 02 Dataset Visualization")
    st.write("可视化数据特征，发现变量之间的趋势与模式。")

with col3:
    st.markdown("### 🧮 03 Model Architecture")
    st.write("说明模型的构建方式与所用指标，帮助理解其预测逻辑。")

with col4:
    st.markdown("### 🔮 04 Prediction")
    st.write("输入特征，获得模型预测结果，用于实际决策模拟。")

with col5:
    st.markdown("### 📈 05 Business Prospects")
    st.write("结合模型输出，讨论预测结果对业务的潜在意义。")

# 额外提示
st.markdown("---")
st.info("📌 请使用左侧导航菜单，在各功能页面间自由切换。")
