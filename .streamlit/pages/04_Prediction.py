import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Title
st.title("🤖 YouTube Video Views Prediction Model")

# Load and preprocess data
df = pd.read_csv("data/processed_youtube.csv")
df = df.dropna(subset=['views', 'likes', 'comment_count'])

# Convert publish_time to datetime and extract month
df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
df['publish_month'] = df['publish_time'].dt.month

# Select features and target
features = ['likes', 'comment_count', 'title_length', 'tag_count', 'publish_hour', 'publish_month']
X = df[features]
y = df['views']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.markdown(f"**Model R² Score:** `{r2:.3f}`")
st.markdown(f"**Mean Absolute Error (MAE):** `{mae:,.0f}` views")

# Prediction Interface
st.header("🎯 Predict Views for a New Video")

likes = st.number_input("👍 Number of Likes", 0, 1_000_000, 50000)
comments = st.number_input("💬 Number of Comments", 0, 500_000, 10000)
title_length = st.slider("📝 Title Length (characters)", 5, 100, 40)
tag_count = st.slider("🏷️ Number of Tags", 0, 30, 10)
publish_hour = st.slider("🕐 Publish Hour (24h)", 0, 23, 17)
publish_month = st.selectbox("📅 Publish Month", list(range(1, 13)))

# Create prediction input
input_data = np.array([[likes, comments, title_length, tag_count, publish_hour, publish_month]])
predicted_views = model.predict(input_data)[0]

# Show result
st.success(f"📺 **Predicted Views:** `{int(predicted_views):,}`")

