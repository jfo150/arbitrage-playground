import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import requests
from sklearn.preprocessing import MinMaxScaler
import joblib

st.title("Arbitrage Playground")
st.write(
    "Use this app to experiment with different cross-liquidity pool arbitrage scenarios in WETH/USDC liquidity pools. Select a model and click run to simulate performance."
)

# fetch data from Etherscan API
@st.cache_data
def fetch_etherscan_data(api_key, address, action='tokentx'):
    base_url = 'https://api.etherscan.io/api'
    params = {
        'module': 'account',
        'action': action,
        'address': address,
        'startblock': 0,
        'endblock': 99999999,
        'sort': 'desc',
        'apikey': api_key
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        st.error(f"API request failed with status code {response.status_code}")
        return None
    
    data = response.json()
    if data['status'] != '1':
        st.error(f"API returned an error: {data['result']}")
        return None
    
    return pd.DataFrame(data['result'])

# data preprocessing function
def preprocess_data(df):
    df['timeStamp'] = pd.to_numeric(df['timeStamp'])
    df = df.sort_values(by='timeStamp', ascending=False).head(10000)
    
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['value'] = np.where(df['tokenDecimal']=='6', df['value']/1e6, df['value']/1e18)
    
    consolidated = df.groupby('hash').agg({
        'timeStamp': 'first',
        'from': 'first',
        'to': 'first',
        'value': lambda x: [sum(x[df['tokenSymbol'] == 'WETH']), sum(x[df['tokenSymbol'] == 'USDC'])],
        'gas': 'first',
        'gasPrice': 'first',
        'gasUsed': 'first',
    }).reset_index()
    
    consolidated['WETH_value'] = consolidated['value'].apply(lambda x: x[0])
    consolidated['USDC_value'] = consolidated['value'].apply(lambda x: x[1])
    consolidated.drop('value', axis=1, inplace=True)
    
    consolidated['time'] = pd.to_datetime(consolidated['timeStamp'], unit='s')
    return consolidated

# load pre-trained model and scaler
@st.cache_resource
def load_model_and_scaler(model_name):
    with open(f'{model_name}_final.pkl', 'rb') as f:
        model = pickle.load(f)
    scaler = joblib.load('scaler.save')
    return model, scaler

# Sidebar
st.sidebar.header("Model Selection")

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

# API key input ??
api_key = st.sidebar.text_input("Etherscan API Key", "YOUR_API_KEY_HERE")
address = st.sidebar.text_input("Contract Address", "0x7bea39867e4169dbe237d55c8242a8f2fcdcc387")

# Run button
if st.sidebar.button("Run Simulation"):
    st.write(f"Running simulation for {selected_model}...")
    
    # Fetch and preprocess data
    with st.spinner("Fetching data from Etherscan..."):
        df = fetch_etherscan_data(api_key, address)
    
    if df is not None:
        st.success("Data fetched successfully!")
        
        with st.spinner("Preprocessing data..."):
            processed_df = preprocess_data(df)
        
        st.subheader("Processed Data Preview")
        st.write(processed_df.head())
        
        # load model and scalar
        model, scaler = load_model_and_scaler(selected_model)
        
        # model data prep
        scaled_data = scaler.transform(processed_df[['WETH_value', 'USDC_value']])
        
        # sequences (for LSTM)
        if selected_model == "LSTM":
            seq_length = 60
            X = []
            for i in range(len(scaled_data) - seq_length):
                X.append(scaled_data[i:i+seq_length])
            X = np.array(X)
        else:
            X = scaled_data
        
        # make preds
        with st.spinner("Running simulation..."):
            predictions = model.predict(X)
        
        if selected_model == "LSTM":
            predictions = scaler.inverse_transform(np.column_stack((predictions, np.zeros_like(predictions))))[:, 0]
            actual = processed_df['WETH_value'].values[seq_length:]
            time_series = processed_df['time'][seq_length:]
        else:
            predictions = scaler.inverse_transform(np.column_stack((predictions, np.zeros_like(predictions))))[:, 0]
            actual = processed_df['WETH_value'].values
            time_series = processed_df['time']
        
        # plot preds vs actual
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_series, actual, label='Actual')
        ax.plot(time_series, predictions, label='Predicted')
        ax.set_xlabel('Time')
        ax.set_ylabel('WETH Value')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # calc metrics
        mse = np.mean((actual - predictions)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual - predictions))
        
        st.subheader("Model Performance Metrics")
        st.write(f"Mean Squared Error: {mse:.4f}")
        st.write(f"Root Mean Squared Error: {rmse:.4f}")
        st.write(f"Mean Absolute Error: {mae:.4f}")

else:
    st.write("Click 'Run Simulation' to start.")