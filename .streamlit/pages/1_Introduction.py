import streamlit as st

st.title("📈 Business Case")

st.markdown("""
### 💡 Problem:
Many jewelry companies struggle to price diamonds appropriately.
Incorrect pricing may lead to loss of revenue or customer dissatisfaction.

### 🎯 Goal:
Build a linear regression model that helps estimate the price of a diamond based on its characteristics (e.g., carat, cut, color, clarity, etc.).

### 📊 Dataset Description:
We are using the `diamonds` dataset from Seaborn, which contains over 50,000 records and the following features:
- `carat`: Weight of the diamond
- `cut`: Quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- `color`: Diamond color, from D (best) to J (worst)
- `clarity`: A measurement of how clear the diamond is
- `depth`: Total depth percentage
- `table`: Width of top of diamond relative to widest point
- `price`: Price in US dollars (target variable)

### 🧩 Business Impact:
This model could be used by diamond retailers to:
- Set optimal pricing strategies
- Understand what features affect price most
- Improve customer trust through data-backed pricing
""")
