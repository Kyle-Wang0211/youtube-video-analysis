import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    return pd.read_csv("processed_youtube.csv")
df = load_data()

st.set_page_config(
    page_title="📊 YouTube Video Analysis APP",
    layout="wide"
)

# 在侧边栏放置导航菜单
with st.sidebar:
    st.title("📚 Contents")
    section = st.selectbox(
        "Select a section",
        [
            "Home",
            "01 Introduction",
            "02 Dataset Visualization",
            "03 Model Architecture",
            "04 Prediction",
            "05 Business Prospects"
        ]
    )

# 页面标题
st.markdown("""
    <h1 style='text-align: center;'>📊 YouTube Video Analysis App</h1>
""", unsafe_allow_html=True)

# 根据选择渲染不同内容
if section == "Home":
    st.markdown("""
        <p style='text-align: center;'>
        This platform uses a linear regression model to analyze YouTube video trends and predict popularity. <br>
        Use the dropdown menu above to explore different sections.
        </p>
        ---
    """, unsafe_allow_html=True)

    # 主页内容
    st.markdown("""
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
    """)

elif section == "01 Introduction":
    st.header("🎯 Objective")
    st.markdown("""
    This project aims to build an interactive application that demonstrates the power of linear regression in predicting YouTube video performance using historical metrics.  
    It simplifies technical concepts for non-technical users and supports real-world decision-making.
    """)

    st.header("💡 Motivation")
    st.markdown("""
    In the digital age, YouTube creators and marketers need to forecast how content will perform to optimize engagement and growth.  
    This app:
    - Helps users understand what makes videos popular.
    - Provides an educational tool for learning regression-based prediction.
    - Assists marketers in campaign planning using data insights.
    """)

elif section == "02 Dataset Visualization":
    st.header("📘 Dataset Overview")
    # —— 打印一下所有列名，排查用 ——  
    df.columns = df.columns.str.strip()
    st.write("所有列名：", df.columns.tolist())

    # —— 然后再画相关性热力图 ——  
    num_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    st.markdown("🔢 **数值列：** " + ", ".join(num_cols))

    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)


elif section == "03 Model Architecture":
    st.markdown("## 🧠 03 Model Architecture")
    st.write("This section describes the structure and logic of the model.")

elif section == "04 Prediction":
    st.markdown("## 🔮 04 Prediction")
    st.write("This section enables user input and prediction display.")

elif section == "05 Business Prospects":
    st.markdown("## 📈 05 Business Prospects")
    st.write("This section discusses the implications of model output.")
