import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle
import requests
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb
st.write(f"XGBoost version: {xgb.__version__}")

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
    
    # calc additional features
    consolidated['gas_price'] = pd.to_numeric(consolidated['gasPrice']) / 1e9  # Convert to Gwei
    consolidated['total_gas_cost'] = consolidated['gas_price'] * pd.to_numeric(consolidated['gasUsed'])
    consolidated['price_ratio'] = consolidated['WETH_value'] / consolidated['USDC_value']
    consolidated['log_price_ratio'] = np.log(consolidated['price_ratio'])
    
    # lag features
    for i in range(1, 6):  # Create 5 lag features
        consolidated[f'WETH_value_lag_{i}'] = consolidated['WETH_value'].shift(i)
        consolidated[f'USDC_value_lag_{i}'] = consolidated['USDC_value'].shift(i)
    
    consolidated = consolidated.dropna()
    
    return consolidated

# load pre-trained model (and scaler??)
@st.cache_resource
def load_model(model_name):
    models_dir = os.path.join(os.getcwd(), 'models')
    st.write(f"Looking for models in: {models_dir}")
    st.write(f"Contents of models directory: {os.listdir(models_dir) if os.path.exists(models_dir) else 'Directory not found'}")
    
    model_path = os.path.join(models_dir, f'{model_name}_final.pkl')
    st.write(f"Attempting to load model from: {model_path}")
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        st.success(f"Model {model_name} loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Sidebar
st.sidebar.header("Model Selection")

models = [
    "LSTM",
    "LGBM",
    "XGB"
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
        
        # load model
        model = load_model(selected_model)
        if model is None:
            st.error("Failed to load model. Please check the files in the 'models' directory.")
        else:
            try:
                # model data prep
                feature_columns = ['WETH_value', 'USDC_value', 'gas_price', 'total_gas_cost', 'price_ratio',
                                   'WETH_value_lag_1', 'WETH_value_lag_2', 'WETH_value_lag_3', 'WETH_value_lag_4', 'WETH_value_lag_5']
                X = processed_df[feature_columns].values
                
                # make preds
                with st.spinner("Running simulation..."):
                    predictions = model.predict(X)
                    if predictions.ndim > 1:
                        predictions = predictions.flatten()
                
                actual = processed_df['WETH_value'].values
                time_series = processed_df['time']
                
                # plot preds vs actual
                chart_data = pd.DataFrame({
                    'Time': time_series,
                    'Actual': actual,
                    'Predicted': predictions
                })
                st.line_chart(chart_data.set_index('Time'))
                
                # calc metrics
                mse = np.mean((actual - predictions)**2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(actual - predictions))
                
                st.subheader("Model Performance Metrics")
                st.write(f"Mean Squared Error: {mse:.4f}")
                st.write(f"Root Mean Squared Error: {rmse:.4f}")
                st.write(f"Mean Absolute Error: {mae:.4f}")
            except Exception as e:
                st.error(f"An error occurred during simulation: {str(e)}")
                st.write("Error details:", e)

else:
    st.write("Click 'Run Simulation' to start.")
