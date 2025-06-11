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

# Second section: Dataset Visualization
elif section == "02 Dataset Visualization":
    st.title("💼 Business Case & Data Presentation")
    st.markdown("""
    **Background**  
    - YouTube is a dominant platform for video content, with billions of active users daily. The ability to predict which videos will trend can help creators and marketers optimize content strategy, audience engagement, and advertising revenue.
    - The app is designed to help stakeholders understand trends, predict viral content, and guide business decisions based on data insights.  
    
    **Objectives**  
    1. **Increase User Retention**: Help content creators and marketers optimize their videos to increase user engagement and retention.
    2. **Boost Revenue**: Provide insights into which types of videos are more likely to generate revenue through ads or subscriptions.
    3. **Enhance Recommendation Models**: Use trending data to improve video recommendations, making them more personalized and relevant to users.
  
    
    **Key Stakeholders**  
    - **Product Teams**: Develop better video recommendation algorithms.
    - **Marketing Teams**: Optimize ad targeting and campaign strategies.
    - **Data Science Teams**: Analyze data to create more accurate predictive models.  
    
    **核心问题**  
    - What factors influence video performance and trending status?
    - How can we predict whether a video will go viral?
    - Can we provide actionable insights based on available data to optimize content strategies?

    """)
    
    st.markdown("---")
    
    # —— 第二部分：Data Presentation ——  
    st.header("📊 Data Presentation")
    
    # Load dataset
    df = load_data()  # Use the defined function to load the data
    st.write(df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Overview")
        st.markdown(f"- **Number of Videos:** {df.shape[0]}")
        st.markdown(f"- **Number of Columns:** {df.shape[1]}")
        st.markdown("**Columns:**  " + ", ".join(list(df.columns)))
        
    with col2:
        st.subheader("Sample Data")
        st.dataframe(df.head(5), use_container_width=True)
        
    if "is_viral" in df.columns:
        st.subheader("Viral vs. Non-Viral Ratio")
        ratio = df["is_viral"].value_counts(normalize=True)
        st.bar_chart(ratio)

elif section == "03 Model Architecture":
    st.markdown("## 🧠 03 Model Architecture")
    st.write("This section describes the structure and logic of the model.")

elif section == "04 Prediction":
    st.markdown("## 🔮 04 Prediction")
    st.write("This section enables user input and prediction display.")

elif section == "05 Business Prospects":
    st.markdown("## 📈 05 Business Prospects")
    st.write("This section discusses the implications of model output.")
