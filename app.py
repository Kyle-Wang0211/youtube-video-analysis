import streamlit as st


# 页面配置
st.set_page_config(
    page_title="📊 YouTube Video Analysis App",
    layout="wide"
)

# 页面顶部主标题（居中）
st.markdown("<h1 style='text-align: center;'>📊 YouTube Video Analysis App</h1>", unsafe_allow_html=True)

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



# 🏠 首页介绍
if section == "🏠 Home":
    st.markdown("<p style='text-align: center;'>This platform uses a linear regression model to analyze YouTube video trends and predict popularity.<br>Use the dropdown menu above to explore different sections.</p>", unsafe_allow_html=True)

    st.markdown("---")
    st.header("🎯 Objective")
    st.markdown("This app aims to help users understand and apply linear regression in analyzing trends and predicting the popularity of YouTube videos. It simplifies complex concepts and enables intuitive interaction with the data.")

    st.header("💡 Motivation")
    st.markdown("""
    - YouTube content creators, marketers, and analysts need effective tools to anticipate video performance.  
    - By predicting future popularity using historical metrics, stakeholders can make informed decisions on content planning, advertising, and engagement.  
    - Our app demonstrates the power of machine learning to support real-world decisions.
    """)

    st.header("🛠️ Technologies Used")
    st.markdown("""
    - **Python** & **Streamlit** for building the interface  
    - **Pandas**, **Seaborn**, and **Matplotlib** for data processing and visualization  
    - **Scikit-learn** for building and evaluating the linear regression model
    """)

    st.header("🧪 Dataset")
    st.markdown("The dataset includes attributes such as view count, likes, comments, and video duration. These features are used to model and predict video popularity using linear regression.")

# 各功能页
elif section == "📘 01 Introduction":
    st.header("📘 Introduction")
    st.markdown("Explain the background, goals, and purpose of the app.")

elif section == "📊 02 Dataset Visualization":
    st.header("📊 Dataset Visualization")
    st.markdown("Show plots and data insights.")

elif section == "🧠 03 Model Architecture":
    st.header("🧠 Model Architecture")
    st.markdown("Display regression structure and metrics.")

elif section == "🔮 04 Prediction":
    st.header("🔮 Prediction")
    st.markdown("Enter new features and get predicted results.")

elif section == "📈 05 Business Prospects":
    st.header("📈 Business Prospects")
    st.markdown("Interpret predictions and discuss implications.")
