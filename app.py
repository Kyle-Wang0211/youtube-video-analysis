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
            "03 Dataset Visualization",
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
elif section == "02 Business Case & Data Presentation":
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
    
    **Core Questions**  
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

elif section == "03 Dataset Visualization":
    st.title("📊 Data Visualization")
    st.markdown("""
    In this section, we will explore the dataset visually to uncover trends and patterns that can provide valuable insights into YouTube video performance.
    """)

    # Distribution of Video Views
    st.subheader("📊 Distribution of Video Views")
    fig, ax = plt.subplots()
    ax.hist(df['views'], bins=30, color='skyblue', edgecolor='black')
    ax.set_title("Video Views Distribution")
    ax.set_xlabel("Number of Views")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    
    # Top 10 Trending Videos by Views
    st.subheader("📊 Top 10 Trending Videos by Views")
    top_trending = df.nlargest(10, 'views')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='views', y='title', data=top_trending, ax=ax, palette="viridis")
    ax.set_title("Top 10 Trending Videos by Views")
    ax.set_xlabel("Views")
    ax.set_ylabel("Video Title")
    st.pyplot(fig)

    # Filter by Views
    st.subheader("🔧 Filter by Views")
    min_views = st.slider("Min Views", 0, int(df['views'].max()), 1000000)
    filtered_df = df[df['views'] >= min_views]
    st.write(f"Showing videos with at least {min_views} views.")
    
    # Show top 10 filtered videos by views
    top_filtered = filtered_df.nlargest(10, 'views')
    st.dataframe(top_filtered[['title', 'views']])

    # Viral vs Non-Viral Video Distribution
    st.subheader("📊 Viral vs Non-Viral Video Distribution")
    viral_ratio = df['is_viral'].value_counts(normalize=True)
    st.write(f"Viral vs Non-Viral video ratio: {viral_ratio.to_dict()}")
    st.bar_chart(viral_ratio)

    # Handle only numeric columns for the correlation matrix
    numeric_df = df.select_dtypes(include=[float, int])

    # Compute the correlation matrix
    correlation_matrix = numeric_df.corr()

    st.subheader("🔍 Correlation Between Features")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title("Correlation Matrix of Features")
    st.pyplot(fig)
    st.write("""
    High Correlation:
    - Views and Likes: The correlation of 0.88 between views and likes indicates a strong positive correlation, meaning that videos with more views tend to have more likes.
    - Views and Comment Count: A correlation of 0.80 suggests that videos with more views also tend to receive more comments, which makes sense as more popular videos are likely to get more engagement.
    
    """)
        
    #Viral vs Non-Viral Videos
    st.subheader("🆚 Comparison: Viral vs Non-Viral Videos")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='is_viral', y='views', data=df, ax=ax)
    ax.set_title("Comparison of Views: Viral vs Non-Viral Videos")
    st.pyplot(fig)

    # Convert publish_time to datetime (if necessary)
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    
    # Check for missing values in publish_time
    st.write(f"Missing publish_time values: {df['publish_time'].isnull().sum()}")
    
    # Extract year and month
    df['publish_year'] = df['publish_time'].dt.year
    df['publish_month'] = df['publish_time'].dt.month
    
    # Verify the new columns
    st.write(df[['publish_time', 'publish_year', 'publish_month']].head())
    # Trends Over Months (Month-wise views or likes)
    views_per_month = df.groupby('publish_month')['views'].sum()
    likes_per_month = df.groupby('publish_month')['likes'].sum()
    
    # Plot Likes per Month
    st.subheader("📊 Likes Per Month")
    fig, ax = plt.subplots(figsize=(10, 6))
    likes_per_month.plot(kind='bar', ax=ax, color='orange')
    ax.set_title("Likes Per Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Likes")
    st.pyplot(fig)

    df_filtered = df[df['views'] <= 1e9]
    views_per_month_filtered = df_filtered.groupby('publish_month')['views'].sum()
    # Plot the filtered data
    st.subheader("📊 Views Per Month (Filtered)")
    fig, ax = plt.subplots(figsize=(10, 6))
    views_per_month_filtered.plot(kind='bar', ax=ax, color='purple')
    ax.set_title("Views Per Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Views")
    st.pyplot(fig)
    # Check for any missing or invalid months
    st.write(df['publish_month'].value_counts())  # Check how many entries exist for each month


elif section == "04 Prediction":
    st.markdown("## 🔮 04 Prediction")
    st.write("This section enables user input and prediction display.")

    # Load and preprocess data
    df = pd.read_csv("processed_youtube.csv")
    df = df.dropna(subset=['views', 'likes', 'comment_count'])

    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    df['publish_month'] = df['publish_time'].dt.month

    # Select features and target
    features = ['likes', 'comment_count', 'title_length', 'tag_count', 'publish_hour', 'publish_month']
    X = df[features]
    y = df['views']

    # Train/test split
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.markdown(f"**Model R² Score:** `{r2:.3f}`")
    st.markdown(f"**Mean Absolute Error (MAE):** `{mae:,.0f}` views")

    # User inputs
    st.header("🎯 Predict Views for a New Video")
    likes = st.number_input("👍 Number of Likes", 0, 1_000_000, 50000)
    comments = st.number_input("💬 Number of Comments", 0, 500_000, 10000)
    title_length = st.slider("📝 Title Length (characters)", 5, 100, 40)
    tag_count = st.slider("🏷️ Number of Tags", 0, 30, 10)
    publish_hour = st.slider("🕐 Publish Hour (24h)", 0, 23, 17)
    publish_month = st.selectbox("📅 Publish Month", list(range(1, 13)))

    # Prediction
    import numpy as np
    input_data = np.array([[likes, comments, title_length, tag_count, publish_hour, publish_month]])
    predicted_views = model.predict(input_data)[0]

    st.success(f"📺 **Predicted Views:** `{int(predicted_views):,}`")



elif section == "05 Business Prospects":
    st.markdown("## 📈 05 Business Prospects")
    st.write("This section discusses the implications of model output.")
