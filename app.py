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
            "02 Business Case & Data Presentation",
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
    # —— 标题 & 概览 ——  
    st.header("📘 Dataset Overview")
    st.markdown(f"- **Number of Videos:** {df.shape[0]}  •  **Number of Columns:** {df.shape[1]}")
    st.dataframe(df.head(5))

    # —— 栏目名检查 ——  
    st.markdown("🧾 **Column names:** " + ", ".join(df.columns.tolist()))

    # —— 分布图 ——  
    st.subheader("🔍 Metric Distributions")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("📺 **Views Distribution**")
        fig1, ax1 = plt.subplots()
        sns.histplot(df["views"], bins=50, kde=True, ax=ax1)
        ax1.set_xlabel("Views")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

    with col2:
        st.markdown("👍 **Likes Distribution**")
        fig2, ax2 = plt.subplots()
        sns.histplot(df["likes"], bins=50, kde=True, ax=ax2)
        ax2.set_xlabel("Likes")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

    with col3:
        st.markdown("💬 **Comments Distribution**")
        fig3, ax3 = plt.subplots()
        sns.histplot(df["comment_count"], bins=50, kde=True, ax=ax3)
        ax3.set_xlabel("Comments")
        ax3.set_ylabel("Count")
        st.pyplot(fig3)

    # —— 相关性热力图 ——  
    st.subheader("📊 Feature Correlation")
    numeric_cols = ["views", "likes", "comment_count", "dislikes"]
    corr = df[numeric_cols].corr()
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="PuBuGn", ax=ax4)
    ax4.set_title("Correlation Matrix")
    st.pyplot(fig4)

    st.markdown("""
    **Insights:**  
    - **Views & Likes** 显示出非常强的相关性。  
    - **Comments** 与 **Views/Likes** 也有中等相关性。  
    - **Dislikes** 相对独立，但也能反映观众反馈。  
    """)



elif section == "03 Model Architecture":
    st.markdown("## 🧠 03 Model Architecture")
    st.write("This section describes the structure and logic of the model.")

elif section == "04 Prediction":
    st.markdown("## 🔮 04 Prediction")
    st.write("This section enables user input and prediction display.")

elif section == "05 Business Prospects":
    st.markdown("## 📈 05 Business Prospects")
    st.write("This section discusses the implications of model output.")
