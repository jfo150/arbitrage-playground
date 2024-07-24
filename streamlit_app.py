import streamlit as st

st.title("Cross-LP Arbitrage Playground")
st.write(
    "Use this app to experiment with different cross-liquidity pool arbitrage scenarios. Select a model and click run to simulate performance."
)

# Sidebar
st.sidebar.header("Model Selection")

# 
models = [
    "Basic Arbitrage Model",
    "LSTM",
    "LGBM",
    "XGBoost"
    
]

selected_model = st.sidebar.selectbox(
    "Choose an arbitrage model:",
    models
)

# display the selected model
st.sidebar.write(f"You selected: {selected_model}")


# Run button
if st.sidebar.button("Run Simulation"):
    st.write(f"Running simulation for {selected_model}...")
    # Placeholder for future simulation logic
    st.write("Simulation results will be displayed here.")