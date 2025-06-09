import streamlit as st

# 设置网页配置
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
    st.markdown("## 🧭 01 Introduction")
    st.write("This section introduces the project scope and goals.")

elif section == "02 Dataset Visualization":
    st.markdown("## 📊 02 Dataset Visualization")
    st.write("This section will include visualizations of dataset features.")

elif section == "03 Model Architecture":
    st.markdown("## 🧠 03 Model Architecture")
    st.write("This section describes the structure and logic of the model.")

elif section == "04 Prediction":
    st.markdown("## 🔮 04 Prediction")
    st.write("This section enables user input and prediction display.")

elif section == "05 Business Prospects":
    st.markdown("## 📈 05 Business Prospects")
    st.write("This section discusses the implications of model output.")
