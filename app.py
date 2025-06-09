import streamlit as st

# 页面配置
st.set_page_config(
    page_title="Linear Regression Business App",  # 网页标题
    page_icon="💼",                                # 网页图标
    layout="wide"                                 # 页面布局：宽屏
)

# 应用标题和欢迎语
st.markdown("<h1 style='text-align: center;'>💼 Linear Regression Business App</h1>", unsafe_allow_html=True)
st.markdown("### 👋 Welcome!")
st.write(
    """
    This platform uses a linear regression model to address real-world business or social problems.
    Use the sidebar on the left or the overview cards below to navigate to different functional pages:
    """
)

# 卡片式模块导航（分栏排布）
col1, col2, col3 = st.columns(3)
col4, col5 = st.columns(2)

# 模块 1：项目介绍
with col1:
    st.markdown("### 🧭 01 Introduction")
    st.write("Overview of the project background, objectives, and scope.")  # 项目背景与目标

# 模块 2：数据可视化
with col2:
    st.markdown("### 📊 02 Dataset Visualization")
    st.write("Visualize key dataset features and identify data patterns.")  # 数据特征展示

# 模块 3：模型结构
with col3:
    st.markdown("### 🧮 03 Model Architecture")
    st.write("Explain model structure and performance metrics for interpretation.")  # 模型结构解释

# 模块 4：预测结果
with col4:
    st.markdown("### 🔮 04 Prediction")
    st.write("Input variables and obtain predictions from the regression model.")  # 输入特征，获取预测

# 模块 5：业务意义
with col5:
    st.markdown("### 📈 05 Business Prospects")
    st.write("Interpret prediction results and reflect on business implications.")  # 结合预测探讨商业潜力

# 页脚提示
st.markdown("---")
st.info("📌 Use the sidebar to freely navigate across all functional pages.")  # 使用左侧导航栏切换页面
