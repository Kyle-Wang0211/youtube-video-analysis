import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 加载数据
@st.cache_data
def load_data():
    return sns.load_dataset('diamonds')

df = load_data()

# 设置页面配置
st.set_page_config(page_title="Linear Regression Business App", layout="wide")

# 创建页面选择器
page = st.sidebar.selectbox("Select a page", ["Business Case", "Data Visualization", "Model Prediction"])

# 页面 1：Business Case
if page == "Business Case":
    st.title("💼 Business Case")
    st.markdown("""
        ### 🎯 Problem:
        Many jewelry companies struggle to price diamonds appropriately.
        
        ### 📌 Objective:
        Use a **linear regression** model to predict the price of a diamond based on its **carat** weight.

        ### 🧾 Dataset:
        Using the `diamonds` dataset from Seaborn, containing 50,000+ diamond entries with cut, clarity, color, carat, and price.

        ---
    """)
    st.dataframe(df.head())

# 页面 2：Data Visualization
elif page == "Data Visualization":
    st.title("📊 Data Visualization")
    
    # 图一：Carat vs Price 散点图
    st.subheader("Carat vs Price")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=df, x="carat", y="price", alpha=0.3, ax=ax1)
    st.pyplot(fig1)

    # 图二：Cut 类型与平均价格柱状图
    st.subheader("Average Price by Cut")
    avg_price_by_cut = df.groupby("cut")["price"].mean().sort_values()
    fig2, ax2 = plt.subplots()
    avg_price_by_cut.plot(kind='barh', ax=ax2)
    ax2.set_xlabel("Average Price")
    st.pyplot(fig2)

# 页面 3：Model Prediction
elif page == "Model Prediction":
    st.title("📈 Linear Regression Prediction")

    st.markdown("We predict `price` using `carat` with a linear regression model.")

    # 准备数据
    X = df[["carat"]]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 建模
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 输入预测
    carat_input = st.slider("Select carat value", min_value=0.2, max_value=5.0, value=0.5, step=0.01)
    predicted_price = model.predict([[carat_input]])[0]

    st.success(f"Predicted Price for {carat_input} carat diamond: ${predicted_price:.2f}")

    # 模型评估
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Model Mean Squared Error (MSE): {mse:.2f}")
