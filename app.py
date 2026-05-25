import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from model import load_and_train
from utils import simulate, recommend

st.set_page_config(page_title="Factory Optimization", layout="wide")

st.title("🏭 Factory Reallocation & Shipping Optimization System")

# Load data
df, model = load_and_train()

X_cols = ['Region','Ship Mode','Division','Product Name','Sales','Cost','Units']

# Sidebar
st.sidebar.header("⚙️ Controls")

product = st.sidebar.selectbox("Select Product", df['Product Name'].unique())
region = st.sidebar.selectbox("Select Region", df['Region'].unique())
ship_mode = st.sidebar.selectbox("Select Ship Mode", df['Ship Mode'].unique())

# KPI Section
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Avg Lead Time", round(df['Shipping Duration'].mean(),2))
col2.metric("Total Sales", int(df['Sales'].sum()))
col3.metric("Total Profit", int(df['Gross Profit'].sum()))

# Visualization
st.subheader("📈 Data Insights")

fig, ax = plt.subplots()
sns.boxplot(data=df, x='Ship Mode', y='Lead Time', ax=ax)
st.pyplot(fig)

# Clustering View
st.subheader("🧠 Clustering Analysis")

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)
df['Cluster'] = kmeans.fit_predict(df[X_cols])

fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x='Sales', y='Lead Time', hue='Cluster', ax=ax2)
st.pyplot(fig2)

# Simulation
st.subheader("🔄 Scenario Simulation")

if st.button("Run Simulation"):

    sim_df = simulate(df, model, X_cols, product, region, ship_mode)

    st.write("### Simulation Results")
    st.dataframe(sim_df)

    # Recommendation
    rec = recommend(sim_df)

    st.write("### 🏆 Top Recommendations")
    st.dataframe(rec)

    # Visualization
    fig3, ax3 = plt.subplots()
    sns.barplot(data=rec, x='Factory', y='Predicted Lead Time', ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)
