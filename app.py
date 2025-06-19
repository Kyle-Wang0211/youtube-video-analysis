import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import shap
import mlflow
from dagshub import DAGsHubLogger
import os

st.set_page_config(
    page_title="📊 YouTube Video Analysis APP",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("processed_youtube.csv")
df = load_data()


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
            "05 Feature Importance & Driving Variables",
            "06 Hyperparameter Tuning",
            "07 Business Prospects"
        ]
    )

# 页面标题
st.markdown("""
    <h1 style='text-align: center;'>📊 YouTube Video Analysis App</h1>
""", unsafe_allow_html=True)

# 根据选择渲染不同内容
if section == "Home":
    st.image("assets/illustration.png",use_container_width=True)
  

    # 主页内容
    st.markdown("----")

    # 📌 团队介绍
    st.markdown("### 👥 Team Introduction")
    
    st.markdown("""
    - **Kyle Wang**  
      Developed Streamlit components and optimized UI/UX processes, established the app framework, participated in the selection of themes in the early stages of the project, and explored the future business prospects of the project.
    
    - **Josephine Wang**  
      Data visualization and user interface specialist. Built Streamlit components and optimized UI/UX flow.
    
    - **Yusheng Qian**  
      Handled data preprocessing, feature engineering, and model development. Built and evaluated the linear regression model that predicts video views based on upload-time features such as title length, tag count, and publish time.
    """)

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
        The Trending YouTube Video Statistics dataset was downloaded from kaggle.com, includes attributes such as view count, likes, comments, and video duration. These features are used to model and predict video popularity using linear regression.
    """)


# Section 1
elif section == "01 Introduction":
    st.header("🎯 Objective")
    st.markdown("""
    This project builds an interactive web application that uses linear regression to predict the number of views a YouTube video might receive, based on pre-publication features such as likes (early likes), comments, title length, tag count, and upload time.
    Our goal is to provide a transparent, educational, and practical tool for exploring what factors may contribute to video popularity on YouTube.
    """)

    st.header("🔍 Background")
    st.markdown("""
    In a digital landscape flooded with video content, understanding what drives virality has become vital for content creators, marketers, and analysts.  
    YouTube, as the world’s largest video-sharing platform, offers a rich trove of user engagement data.  
    While tech giants use deep learning for personalized recommendations, our project focuses on **linear regression**—a simpler and explainable model—to uncover insights from basic video features.
    """)

    st.header("🧠 Research Questions")
    st.markdown("""
    - What features (likes, comments, publish hour) influence a video's views?
    - Can we estimate future popularity using historical patterns?
    - How can data support smarter content strategy decisions?
    """)

    st.header("🛠️ Technologies Used")
    st.markdown("""
    - **Streamlit** for interface development  
    - **Pandas**, **Seaborn**, and **Matplotlib** for data processing and visualization  
    - **Scikit-learn** for training and evaluating the linear regression model
    """)

    st.header("📊 Dataset Overview")
    st.markdown("""
    - **Source**: YouTube Trending Video Statistics (Kaggle)  
    - **Size**: 12,440 videos with 11 attributes  
    - **Key Fields**: views, likes, comment_count, title_length, tag_count, publish_hour, is_viral
    """)

    st.header("🧑‍🏫 Use Cases")
    st.markdown("""
    - 🎓 Educational: Demonstrates regression modeling in an intuitive way  
    - 📈 Marketing: Assists in upload timing and content strategy planning  
    - 🧪 Analytical: Serves as a replicable ML pipeline for future experiments
    """)

# Second section: Dataset Visualization
elif section == "02 Business Case & Data Presentation":
    st.title("💼 Business Case & Data Presentation")
    st.markdown("""
    **Background**  
    - YouTube is a dominant platform for video content, with billions of active users daily. The ability to predict which videos will trend can help creators and marketers optimize content strategy, audience engagement, and advertising revenue.
    - The app is designed to help stakeholders understand trends, predict viral content, and guide business decisions based on data insights.  
    
    **Objectives**  
    - **Increase User Retention**: Help content creators and marketers optimize their videos to increase user engagement and retention.
    - **Boost Revenue**: Provide insights into which types of videos are more likely to generate revenue through ads or subscriptions.
    - **Enhance Recommendation Models**: Use trending data to improve video recommendations, making them more personalized and relevant to users.
  
    
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

    # —— KPI Cards ——  
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("📺 Total Views", f"{int(df['views'].sum()):,}")
    kpi2.metric("👍 Avg. Likes", f"{df['likes'].mean():.0f}")
    kpi3.metric("💬 Avg. Comments", f"{df['comment_count'].mean():.0f}")
    kpi4.metric("🔥 Viral Rate", f"{df['is_viral'].mean()*100:.1f}%")
    st.markdown("---")

    st.header("🔍 Data Quality & Processing")
    st.markdown("""
    **1. Data Cleaning**  
    - **Deduplication:** Remove duplicate rows based on video_id or the combination of title and channel_title.  
        - Remove rows if key metrics like **views**, **likes**, or **comment_count** are blank.  
        - For other fields (e.g. **tag_count**, **title_length**), replace blanks with 0 or the column’s median.  
        - Drop rows with bad or unreadable **publish_time** entries.  
    - **Remove Outliers:** Use box-plot rules (IQR) to filter out extreme values in **views**, **likes**, and **comment_count**.

    **2. Feature Engineering & Scaling**  
    - **Date Features:** Extract hour and day of week from **publish_time**.  
    - **Text Features:** Calculate title length and tag count.  
    - **Log Scaling:** Apply log1p to **views**, **likes**, and **comment_count** to even out skewed data.  
    - **Normalization:** Scale key numeric features to the same range for modeling.

    **3. “is_viral” Label Definition**  
    - Sort videos by **original** views in descending order and label the top 10% as is_viral = 1, others as 0.  
    - **Threshold Justification:** A 10% positive rate balances representation and model training needs; adjustable to 5% or 15% based on business context.  
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

        # —— 交互式业务假设验证 ——  
    st.sidebar.header("🧪 Business Hypothesis Filter")
    # 时间段筛选
    time_slot = st.sidebar.multiselect(
        "Publish Time Slot",
        ["Morning", "Afternoon", "Evening"],
        default=["Morning", "Afternoon", "Evening"]
    )
    # 标签数量范围
    tag_min, tag_max = st.sidebar.slider(
        "Tag Count Range",
        int(df["tag_count"].min()),
        int(df["tag_count"].max()),
        (0, int(df["tag_count"].max()))
    )

    # 根据筛选条件过滤 df
    def slot(h):
        if h < 12:   return "Morning"
        if h < 18:   return "Afternoon"
        return "Evening"

    df["time_slot"] = df["publish_hour"].apply(slot)
    df_filtered = df[
        df["time_slot"].isin(time_slot) &
        df["tag_count"].between(tag_min, tag_max)
    ]

    # —— Top Channels 排名 ——  
    st.subheader("🏆 Top 5 Channels by Total Views")
    ch_stats = (
        df_filtered
        .groupby("channel_title")["views"]
        .agg(total_views="sum", avg_views="mean")
        .sort_values("total_views", ascending=False)
        .head(5)
    )
    # 总播放量柱状图
    st.bar_chart(ch_stats["total_views"])
    # 排名表格
    st.table(ch_stats.style.format({"total_views":"{:,}","avg_views":"{:.0f}"}))

    # —— Viral 视频案例剖析 ——  
    st.subheader("🎬 Viral Video Case Study")
    top_viral = df_filtered[df_filtered["is_viral"]==1].nlargest(2, "views")
    for _, row in top_viral.iterrows():
        st.markdown(f"**{row['title']}**  |  Published: {row['publish_time']}")
        st.write({
            "Views": f"{row['views']:,}",
            "Likes": row["likes"],
            "Comments": row["comment_count"],
            "Tags": row["tag_count"]
        })
        # 如果你有播放量随时间的序列数据，可以在这里画折线：
        # st.line_chart(your_time_series_df[row['video_id']])
        st.markdown("---")


elif section == "03 Dataset Visualization":
    st.title("📊 Data Visualization")
    st.markdown("""
    In this section, we will explore the dataset visually to uncover trends and patterns that can provide valuable insights into YouTube video performance.
    """)

    st.markdown("---")

    import matplotlib.ticker as ticker
    filtered_df = df[df['views'] <= 5e6]  # 过滤到500万以下
    fig, ax = plt.subplots()
    ax.hist(filtered_df['views'], bins=100, color='skyblue', edgecolor='black')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1e6)}M'))
    
    ax.set_title("Video Views Distribution (Only Views ≤ 5M)")
    ax.set_xlabel("Number of Views")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)


    
    # Top 10 Trending Videos by Views
    st.subheader("Top 10 Trending Videos by Views")
    top_trending = df.nlargest(10, 'views')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='views', y='title', data=top_trending, ax=ax, palette="viridis")
    ax.set_title("Top 10 Trending Videos by Views")
    ax.set_xlabel("Views")
    ax.set_ylabel("Video Title")
    st.pyplot(fig)

    # Filter by Views
    st.subheader("Filter by Views")
    min_views = st.slider("Min Views", 0, int(df['views'].max()), 1000000)
    filtered_df = df[df['views'] >= min_views]
    st.write(f"Showing videos with at least {min_views} views.")
    
    # Show top 10 filtered videos by views
    top_filtered = filtered_df.nlargest(10, 'views')
    st.dataframe(top_filtered[['title', 'views']])

    # Viral vs Non-Viral Video Distribution
    st.subheader("Viral vs Non-Viral Video Distribution")
    viral_ratio = df['is_viral'].value_counts(normalize=True)
    st.write(f"Viral vs Non-Viral video ratio: {viral_ratio.to_dict()}")
    st.bar_chart(viral_ratio)

    # Handle only numeric columns for the correlation matrix
    numeric_df = df.select_dtypes(include=[float, int])

    # Compute the correlation matrix
    correlation_matrix = numeric_df.corr()

    st.subheader("Correlation Between Features")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title("Correlation Matrix of Features")
    st.pyplot(fig)
    st.write("""
    High Correlation:
    - Views and Likes: The correlation of 0.88 between views and likes indicates a strong positive correlation, meaning that videos with more views tend to have more likes.
    - Views and Comment Count: A correlation of 0.80 suggests that videos with more views also tend to receive more comments, which makes sense as more popular videos are likely to get more engagement.
    
    """)

    # Convert publish_time to datetime (if necessary)
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    
    # Check for missing values in publish_time
    st.write(f"Missing publish_time values: {df['publish_time'].isnull().sum()}")
    
    # Extract year and month
    df['publish_year'] = df['publish_time'].dt.year
    df['publish_month'] = df['publish_time'].dt.month
    
    
    # Group and filter Likes Per Month
    monthly_likes = df.groupby('publish_month')['likes'].sum().reset_index()
    threshold = 10000
    filtered_likes = monthly_likes[monthly_likes['likes'] > threshold]
    
    # Plot Likes Per Month (Filtered)
    st.subheader("Likes Per Month (Filtered)")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(filtered_likes['publish_month'], filtered_likes['likes'], color='orange')
    ax.set_title("Likes Per Month (Filtered)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Likes")
    st.pyplot(fig)
    
    # Add Explanation Note
    st.markdown("""
    > ℹ️ **Note**: Only months with total likes greater than 10,000 are shown.  
    """)
    
    # Group and filter Views Per Month
    monthly_views = df.groupby('publish_month')['views'].sum().reset_index()
    filtered_views = monthly_views[monthly_views['views'] > 1000000]  # 你也可以调整这个 threshold
    
    # Plot Views Per Month (Filtered)
    st.subheader("Views Per Month (Filtered)")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(filtered_views['publish_month'], filtered_views['views'], color='purple')
    ax.set_title("Views Per Month (Filtered)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Views")
    st.pyplot(fig)
    
    # Add Explanation Note
    st.markdown("""
    > ℹ️ **Note**: Only months with total views greater than 1,000,000 are shown.  
    """)




elif section == "04 Prediction":
    st.title("📈 YouTube Video Views Prediction")

    # 用户选择模型
    model_choice = st.selectbox("🧠 Choose Model", ["Linear Regression", "XGBoost Regressor"])

    # 用户选择评价指标
    selected_metrics = st.multiselect(
        "📊 Select Evaluation Metrics",
        ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "R² Score"],
        default=["R² Score", "Mean Absolute Error (MAE)"]
    )

    # 数据加载与处理
    df2 = pd.read_csv("processed_youtube.csv")
    df2 = df2.dropna()
    df2['publish_time'] = pd.to_datetime(df2['publish_time'], errors='coerce')
    df2['publish_month'] = df2['publish_time'].dt.month

    features = ['likes', 'comment_count', 'title_length', 'tag_count', 'publish_hour', 'publish_month']
    X = df2[features]
    y = df2['views']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 初始化模型
    if model_choice == "Linear Regression":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    else:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    from sklearn import metrics
    r2, mae, mse = None, None, None

    st.subheader("📈 Model Evaluation")
    if "R² Score" in selected_metrics:
        r2 = metrics.r2_score(y_test, predictions)
        st.write(f"- **R² Score:** {r2:.3f}")
    if "Mean Absolute Error (MAE)" in selected_metrics:
        mae = metrics.mean_absolute_error(y_test, predictions)
        st.write(f"- **MAE:** {mae:,.2f}")
    if "Mean Squared Error (MSE)" in selected_metrics:
        mse = metrics.mean_squared_error(y_test, predictions)
        st.write(f"- **MSE:** {mse:,.2f}")

    if r2 is not None:
        st.markdown(f"**Model R² Score:** `{r2:.3f}`")
    if mae is not None:
        st.markdown(f"**Mean Absolute Error (MAE):** `{mae:,.0f}` views")

    st.header("🔮 Predict Views for a New Video")
    likes = st.number_input("👍 Number of Likes", 0, 1_000_000, 50000)
    comments = st.number_input("💬 Number of Comments", 0, 500_000, 10000)
    title_length = st.slider("📝 Title Length", 5, 100, 40)
    tag_count = st.slider("🏷️ Tag Count", 0, 30, 10)
    publish_hour = st.slider("🕐 Publish Hour", 0, 23, 17)
    publish_month = st.selectbox("📅 Publish Month", list(range(1, 13)))

    input_data = pd.DataFrame([{
        'likes': likes,
        'comment_count': comments,
        'title_length': title_length,
        'tag_count': tag_count,
        'publish_hour': publish_hour,
        'publish_month': publish_month
    }])

    predicted_views = model.predict(input_data)[0]
    st.success(f"📺 **Predicted Views ({model_choice}):** `{int(predicted_views):,}`")

elif section == "06 Feature Importance & Driving Variables":
    st.title("🔍 Feature Importance & Driving Variables")

    st.markdown("""
    This page helps users understand which input variables have the greatest impact on YouTube video view predictions using SHAP (SHapley Additive exPlanations).
    """)

    import shap
    import xgboost
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split

    df_shap = df.dropna(subset=['views', 'likes', 'comment_count', 'title_length', 'tag_count', 'publish_hour', 'publish_time'])
    df_shap['publish_month'] = pd.to_datetime(df_shap['publish_time'], errors='coerce').dt.month

    # Select features and target
    features = ['likes', 'comment_count', 'title_length', 'tag_count', 'publish_hour', 'publish_month']
    X = df_shap[features]
    y = df_shap['views']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)

    # Explain predictions using SHAP
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(X_train)

    # Plot summary
    st.subheader("📊 SHAP Summary Plot (Top Features)")
    fig_summary, ax_summary = plt.subplots(figsize=(10, 5))
    shap.plots.beeswarm(shap_values, max_display=6, ax=ax_summary, show=False)
    st.pyplot(fig_summary)

    # Plot bar
    st.subheader("📈 SHAP Feature Importance (Bar)")
    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
    shap.plots.bar(shap_values, max_display=6, ax=ax_bar, show=False)
    st.pyplot(fig_bar)

    # Plot waterfall
    st.subheader("💧 SHAP Waterfall Plot (Single Sample)")
    sample_index = st.slider("Select a sample index", 0, len(X_train) - 1, 0)
    fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[sample_index], max_display=6, show=False)
    st.pyplot(fig_waterfall)

    # Interpretation
    st.markdown("""
    ### 🔑 Key Takeaways

    - **Publish Hour** is often a key driver: videos posted at certain times may attract more attention.
    - **Likes and Comments** are strong engagement indicators; their SHAP values show high predictive influence.
    - **Title Length and Tag Count** help with discoverability via SEO and search ranking on YouTube.

    These insights can guide content creators to refine their strategy and schedule for better performance.
    """)

    st.success("These insights make the machine learning model explainable and trustworthy.")

elif section == "06 Hyperparameter Tuning":
    st.title("🔧 MLflow + DAGsHub Hyperparameter Tuning")
    
    # 🔁 Prepare data again
    df2 = df.dropna()
    df2['publish_month'] = pd.to_datetime(df2['publish_time'], errors='coerce').dt.month
    features = ['likes', 'comment_count', 'title_length', 'tag_count', 'publish_hour', 'publish_month']
    X = df2[features]
    y = df2['views']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    os.environ["MLFLOW_TRACKING_USERNAME"] = "Yusheng-Qian"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "fc89fc3a53e2948f33bd036fba14b61528360901"
    mlflow.set_tracking_uri("https://dagshub.com/Yusheng-Qian/YouTubeVideoPrediction.mlflow")
    mlflow.set_experiment("youtube_xgb_tuning")

    params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
    }

    with mlflow.start_run():
        mlflow.log_params(params)
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        mlflow.log_metric("mse", mse)
        st.write(f"MLflow logged MSE: {mse:.2f}")
        
          # Add RMSE display
        rmse = np.sqrt(mse)
        st.write(f"📏 RMSE: {rmse:,.0f} views")
        
    
elif section == "07 Business Prospects":
    st.markdown("## 📈 07 Business Prospects")
    st.write("This section discusses the implications of model output.")

    st.markdown("### 💼 Target Users")
    st.markdown("""
    - **YouTube Creators**: Understand what content strategies drive views and engagement.  
    - **Marketing Teams**: Predict campaign success before launch and optimize ad timing.  
    - **Talent Management Agencies**: Use prediction tools to evaluate rising creators.  
    - **EdTech & Data Education**: Showcase real-life machine learning use cases with explainable models.
    """)

    st.markdown("---")

    st.markdown("### 🚀 Business Applications")
    st.markdown("""
    1. **Content Optimization Platform**  
       Integrate this prediction model into a tool that gives creators daily suggestions on when to post, how long titles should be, or whether a video might trend based on its draft metadata.
       
    2. **Brand-Influencer Matching**  
       Help brands choose creators not only based on past performance, but based on future projected performance (using our regression predictions).

    3. **YouTube SEO Assistant**  
       Recommend tag count, title length, and upload time based on historical performance to increase discoverability and engagement.
    """)

    st.markdown("---")

    st.markdown("### 🧠 Future Improvements")
    st.markdown("""
    - **Model Enhancements**:  
      Replace linear regression with more advanced models like random forests or XGBoost to improve accuracy.  

    - **Real-Time API**:  
      Deploy model as a web API for integration with platforms like Notion, Discord bots, or Chrome extensions for creators.  

    - **Multi-language Dataset**:  
      Expand beyond English-speaking content to support global creators and advertisers.  

    - **Time-Series Forecasting**:  
      Incorporate trends over time and viewer retention patterns to forecast not only total views but expected growth trajectory.
    """)

    st.markdown("---")

    st.markdown("### 🌍 Vision")
    st.markdown("""
    In a creator economy that moves fast, having a **lightweight, interpretable, and interactive tool** like ours can bridge the gap between creators and data.  
    Whether it’s a solo vlogger in Barcelona or a brand team in New York, predictive insights give **data-driven creators a competitive edge**.
    """)

    st.success("Thank you for exploring our project! We now welcome your questions.")



