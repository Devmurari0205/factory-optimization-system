import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from model import load_and_train
from utils import simulate, recommend
from utils import simulate_factory, recommend_top

# --- MOVE UTILS CODE HERE ---

def simulate_factory(df, model, encoders, product, region, ship_mode):
    results = []

    factories = [
        "Lot's O' Nuts",
        "Wicked Choccy's",
        "Sugar Shack",
        "Secret Factory",
        "The Other Factory"
    ]

    for factory in factories:
        sample = df[df['Product Name'] == product].iloc[0]

        X = pd.DataFrame({
            'Region_enc': [encoders['region'].transform([region])[0]],
            'Ship Mode_enc': [encoders['ship'].transform([ship_mode])[0]],
            'Product_enc': [encoders['product'].transform([product])[0]],
            'Units': [sample['Units']],
            'Cost': [sample['Cost']]
        })

        predicted_time = model.predict(X)[0]

        results.append({
            "Factory": factory,
            "Predicted Lead Time": predicted_time
        })

    return pd.DataFrame(results).sort_values("Predicted Lead Time")


def recommend_top(df_sim):
    return df_sim.iloc[0]

st.set_page_config(page_title="Factory Optimization", layout="wide")

st.title("🏭 Factory Reallocation & Shipping Optimization Dashboard")

# Load model + data
df, model, encoders = load_and_train()

# Sidebar controls
st.sidebar.header("⚙️ Controls")

product = st.sidebar.selectbox("Select Product", df['Product Name'].unique())
region = st.sidebar.selectbox("Select Region", df['Region'].unique())
ship_mode = st.sidebar.selectbox("Ship Mode", df['Ship Mode'].unique())

priority = st.sidebar.slider("Optimization Priority (Speed vs Profit)", 0, 100, 50)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Simulator",
    "📈 What-If Analysis",
    "🏆 Recommendations",
    "⚠️ Risk Panel"
])

# ------------------ TAB 1 ------------------
with tab1:
    st.subheader("Factory Optimization Simulator")

    sim_df = simulate_factory(df, model, encoders, product, region, ship_mode)

    st.dataframe(sim_df)

    st.bar_chart(sim_df.set_index("Factory"))

# ------------------ TAB 2 ------------------
with tab2:
    st.subheader("What-If Scenario")

    current = sim_df.iloc[-1]
    best = sim_df.iloc[0]

    st.write("### Current vs Best Factory")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Current Lead Time", round(current["Predicted Lead Time"], 2))

    with col2:
        st.metric("Best Lead Time", round(best["Predicted Lead Time"], 2))

    improvement = ((current["Predicted Lead Time"] - best["Predicted Lead Time"]) / current["Predicted Lead Time"]) * 100

    st.success(f"🚀 Improvement: {round(improvement,2)}%")

# ------------------ TAB 3 ------------------
with tab3:
    st.subheader("Top Recommendations")

    top = recommend_top(sim_df)

    st.write("### Best Factory Recommendation")
    st.success(f"🏭 {top['Factory']}")

    st.metric("Predicted Lead Time", round(top["Predicted Lead Time"], 2))

# ------------------ TAB 4 ------------------
with tab4:
    st.subheader("Risk & Impact Panel")

    st.warning("⚠️ Profit Impact Risk Analysis")

    # Simple proxy logic
    risk_score = 100 - priority

    st.metric("Risk Score", risk_score)

    if risk_score > 50:
        st.error("High Risk: May impact profit margins")
    else:
        st.success("Low Risk: Safe optimization")

# ------------------ Footer ------------------
st.write("---")
st.caption("Built for Nassau Candy Optimization System 🚀")
