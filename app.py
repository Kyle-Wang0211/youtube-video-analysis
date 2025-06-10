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

elif section == "02 Business Case & Data Presentation":
    st.markdown("<h2 style='text-align: center;'>📊 Business Case + Dataset Walkthrough</h2>", unsafe_allow_html=True)

    st.divider()

    # -------- Business Section --------
    st.subheader("💼 Why Does This Project Matter?")
    st.markdown("""
    In this project, we wanted to explore how data from YouTube videos can help **predict popularity** based on visible metrics like views, likes, comments, and duration.

    ### 📍 Background
    - YouTube is one of the most popular platforms for content creation.
    - Creators and marketers often wonder: _What kind of content will perform well?_
    - We were curious: _Can we build a simple tool that gives clues before uploading?_

    ### 🧠 What We Learned
    - Some video metrics are strongly connected (e.g., views & likes).
    - Not all long videos perform well — engagement isn't always about duration.
    - Even with a simple model (linear regression), predictions are reasonably accurate.

    ### 🎯 Who Might Use This?
    - **Student Creators** 🧑‍🎓: plan uploads for a school club, vlog, or mini documentary
    - **Media Clubs** 🎬: estimate which videos get more traction for editing priority
    - **Beginner Marketers** 📣: test hypotheses without diving into advanced AI

    👉 *We want to emphasize that this is a learning project — not a final product.*
    """)

    st.divider()

    # -------- Dataset Section --------
    st.subheader("📘 What Does the Dataset Look Like?")
    st.markdown(f"""
    - Number of Videos: **{df.shape[0]}**
    - Number of Columns: **{df.shape[1]}**
    """)
    st.dataframe(df.head(5))

    st.markdown("⬇️ Here are some simple visualizations to help us understand the data:")

    # -------- Distribution Charts --------
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")  # lighter background

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("📺 **View Count Distribution**")
        fig1, ax1 = plt.subplots()
        sns.histplot(df["view_count"], bins=40, kde=True, color="#91c8f6", ax=ax1)
        ax1.set_xlabel("Views")
        ax1.set_ylabel("Number of Videos")
        st.pyplot(fig1)

    with col2:
        st.markdown("👍 **Like Count Distribution**")
        fig2, ax2 = plt.subplots()
        sns.histplot(df["like_count"], bins=40, kde=True, color="#f6a091", ax=ax2)
        ax2.set_xlabel("Likes")
        ax2.set_ylabel("Number of Videos")
        st.pyplot(fig2)

    # -------- Correlation Heatmap --------
    st.markdown("📊 **How Are the Metrics Related?**")
    numeric_cols = ["view_count", "like_count", "comment_count", "duration"]
    fig3, ax3 = plt.subplots()
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="PuBuGn", fmt=".2f", ax=ax3)
    ax3.set_title("Correlation Heatmap")
    st.pyplot(fig3)

    st.markdown("""
    ✅ From the heatmap:
    - View count & like count show strong correlation (~0.85)
    - Comments are moderately correlated with views/likes
    - Duration doesn’t have a strong linear relationship — interesting!

    These patterns gave us confidence that a basic linear model could work.
    """)

    st.divider()
    st.caption("📝 This page was built and written entirely by students as part of a midterm project.")



elif section == "03 Model Architecture":
    st.markdown("## 🧠 03 Model Architecture")
    st.write("This section describes the structure and logic of the model.")

elif section == "04 Prediction":
    st.markdown("## 🔮 04 Prediction")
    st.write("This section enables user input and prediction display.")

elif section == "05 Business Prospects":
    st.markdown("## 📈 05 Business Prospects")
    st.write("This section discusses the implications of model output.")
