import streamlit as st
import pandas as pd
import sys, os

# Fix import path
sys.path.append(os.path.dirname(__file__))

from model import load_and_train
from utils import simulate_factory, recommend_top

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Factory Optimization", layout="wide")

st.title("🏭 Factory Reallocation & Shipping Optimization System")

# ---------------- LOAD DATA ----------------
df, model, encoders = load_and_train()

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ User Controls")

product = st.sidebar.selectbox("📦 Select Product", df['Product Name'].unique())
region = st.sidebar.selectbox("🌍 Select Region", df['Region'].unique())
ship_mode = st.sidebar.selectbox("🚚 Ship Mode", df['Ship Mode'].unique())

priority = st.sidebar.slider(
    "⚖️ Optimization Priority (Speed vs Profit)",
    0, 100, 50
)

# ---------------- SIMULATION ----------------
sim_df = simulate_factory(df, model, encoders, product, region, ship_mode)

best = sim_df.iloc[0]
current = sim_df.iloc[-1]

improvement = ((current["Predicted Lead Time"] - best["Predicted Lead Time"]) / current["Predicted Lead Time"]) * 100

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🏭 Simulator",
    "📊 What-If Analysis",
    "🏆 Recommendations",
    "⚠️ Risk & Impact"
])

# =========================================================
# 🏭 1. FACTORY OPTIMIZATION SIMULATOR
# =========================================================
with tab1:
    st.subheader("Factory Optimization Simulator")

    st.markdown("### 📊 Predicted Performance Across Factories")

    st.dataframe(sim_df, use_container_width=True)

    st.bar_chart(sim_df.set_index("Factory"))

# =========================================================
# 📊 2. WHAT-IF SCENARIO ANALYSIS
# =========================================================
with tab2:
    st.subheader("What-If Scenario Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric("Current Factory Time", round(current["Predicted Lead Time"], 2))
    col2.metric("Best Factory Time", round(best["Predicted Lead Time"], 2))
    col3.metric("Improvement %", f"{round(improvement,2)}%")

    st.markdown("### 📉 Lead Time Comparison")

    compare_df = pd.DataFrame({
        "Scenario": ["Current", "Optimized"],
        "Lead Time": [current["Predicted Lead Time"], best["Predicted Lead Time"]]
    })

    st.bar_chart(compare_df.set_index("Scenario"))

# =========================================================
# 🏆 3. RECOMMENDATION DASHBOARD
# =========================================================
with tab3:
    st.subheader("Top Factory Recommendations")

    st.success(f"✅ Recommended Factory: {best['Factory']}")

    st.metric("Predicted Lead Time", best["Predicted Lead Time"])

    st.markdown("### 📊 Ranked Factory Suggestions")

    st.dataframe(sim_df)

# =========================================================
# ⚠️ 4. RISK & IMPACT PANEL
# =========================================================
with tab4:
    st.subheader("Risk & Impact Analysis")

    # Risk calculation (simple logic)
    risk_score = 100 - priority

    col1, col2 = st.columns(2)

    col1.metric("⚠️ Risk Score", risk_score)
    col2.metric("📉 Expected Improvement", f"{round(improvement,2)}%")

    st.markdown("### 🚨 Risk Alerts")

    if risk_score > 70:
        st.error("High Risk: Profit may be impacted ⚠️")
    elif risk_score > 40:
        st.warning("Moderate Risk: Monitor carefully ⚠️")
    else:
        st.success("Low Risk: Safe optimization ✅")

# ---------------- FOOTER ----------------
st.write("---")
st.caption("Nassau Candy Optimization System | Built with Streamlit 🚀")
