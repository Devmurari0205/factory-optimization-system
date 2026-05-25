import streamlit as st
import pandas as pd
from model import load_and_train

# FIX IMPORT ERROR
import sys, os
sys.path.append(os.path.dirname(__file__))
from utils import simulate_factory, recommend_top

st.set_page_config(layout="wide")
st.title("🏭 Factory Optimization Dashboard")

df, model, encoders = load_and_train()

# SIDEBAR
st.sidebar.header("Controls")
product = st.sidebar.selectbox("Product", df['Product Name'].unique())
region = st.sidebar.selectbox("Region", df['Region'].unique())
ship = st.sidebar.selectbox("Ship Mode", df['Ship Mode'].unique())
priority = st.sidebar.slider("Speed vs Profit", 0, 100, 50)

# TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "Simulator", "What-If", "Recommendations", "Risk"
])

# -------- TAB 1 --------
with tab1:
    st.subheader("Factory Simulator")

    sim = simulate_factory(df, model, encoders, product, region, ship)

    st.dataframe(sim)
    st.bar_chart(sim.set_index("Factory"))

# -------- TAB 2 --------
with tab2:
    st.subheader("What-If Analysis")

    current = sim.iloc[-1]
    best = sim.iloc[0]

    col1, col2 = st.columns(2)

    col1.metric("Current Lead Time", current["Predicted Lead Time"])
    col2.metric("Best Lead Time", best["Predicted Lead Time"])

    improvement = ((current["Predicted Lead Time"] - best["Predicted Lead Time"]) / current["Predicted Lead Time"]) * 100
    st.success(f"Improvement: {round(improvement,2)}%")

# -------- TAB 3 --------
with tab3:
    st.subheader("Recommendations")

    top = recommend_top(sim)

    st.success(f"Best Factory: {top['Factory']}")
    st.metric("Lead Time", top["Predicted Lead Time"])

# -------- TAB 4 --------
with tab4:
    st.subheader("Risk Panel")

    risk = 100 - priority
    st.metric("Risk Score", risk)

    if risk > 50:
        st.error("High Risk ⚠️")
    else:
        st.success("Low Risk ✅")
