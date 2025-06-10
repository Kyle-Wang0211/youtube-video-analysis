import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
@st.cache(allow_output_mutation=True)
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
    st.markdown("""
    The dataset includes the following attributes for YouTube videos:
    - View count
    - Like count
    - Comment count
    - Duration
    - Category, title, and tags
    """)

    st.header("📊 Visualization Insights")
    st.markdown("""
    - **View count** and **like count** are highly correlated.  
    - Longer videos do not always perform better; the effect of duration is nonlinear.  
    - Comments often correlate with deeper viewer engagement.
    """)

    st.subheader("🔍 Sample Chart: View Count Distribution")
    # st.pyplot(fig_view_dist)  # Replace with your actual chart variable
    st.info("Replace this with your actual chart, e.g., st.pyplot() or st.line_chart()")


elif section == "03 Model Architecture":
    st.markdown("## 🧠 03 Model Architecture")
    st.write("This section describes the structure and logic of the model.")

elif section == "04 Prediction":
    st.markdown("## 🔮 04 Prediction")
    st.write("This section enables user input and prediction display.")

elif section == "05 Business Prospects":
    st.markdown("## 📈 05 Business Prospects")
    st.write("This section discusses the implications of model output.")
