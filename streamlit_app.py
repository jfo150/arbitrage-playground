import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle
import requests
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

st.title("Arbitrage Playground")
st.write(
    "Use this app to experiment with different cross-liquidity pool arbitrage scenarios in WETH/USDC liquidity pools. Select a model and click run to simulate performance."
)

# fetch data from Etherscan API
@st.cache_data
def etherscan_request(action, api_key, address, startblock=0, endblock=99999999, sort='desc'):
    base_url = 'https://api.etherscan.io/api'
    params = {
        'module': 'account',
        'action': action,
        'address': address,
        'startblock': startblock,
        'endblock': endblock,
        'sort': sort,
        'apikey': api_key
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        st.error(f"API request failed with status code {response.status_code}")
        return None, None
    
    data = response.json()
    if data['status'] != '1':
        st.error(f"API returned an error: {data['result']}")
        return None, None
    
    df = pd.DataFrame(data['result'])
    return process_etherscan_data(df), df

def process_etherscan_data(df):
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
    
    return consolidated

@st.cache_data
def merge_pool_data(p0, p1):
    # Format P0 and P1 variables of interest
    p0['time'] = p0['timeStamp'].apply(lambda x: datetime.fromtimestamp(x))
    p0['p0.weth_to_usd_ratio'] = p0['WETH_value'] / p0['USDC_value']
    p0['gasPrice'] = p0['gasPrice'].astype(float)
    p0['gasUsed'] = p0['gasUsed'].astype(float)
    p0['p0.gas_fees_usd'] = (p0['gasPrice']/1e9) * (p0['gasUsed']/1e9) * p0['p0.weth_to_usd_ratio']
    
    p1['time'] = p1['timeStamp'].apply(lambda x: datetime.fromtimestamp(x))
    p1['p1.weth_to_usd_ratio'] = p1['WETH_value'] / p1['USDC_value']
    p1['gasPrice'] = p1['gasPrice'].astype(float)
    p1['gasUsed'] = p1['gasUsed'].astype(float)
    p1['p1.gas_fees_usd'] = (p1['gasPrice']/1e9) * (p1['gasUsed']/1e9) * p1['p1.weth_to_usd_ratio']

    # Merge Pool data
    both_pools = pd.merge(p0[['time', 'timeStamp', 'p0.weth_to_usd_ratio', 'p0.gas_fees_usd']],
                          p1[['time', 'timeStamp', 'p1.weth_to_usd_ratio', 'p1.gas_fees_usd']],
                          on=['time', 'timeStamp'], how='outer').sort_values(by='timeStamp')
    both_pools = both_pools.ffill().reset_index(drop=True)
    both_pools = both_pools.dropna()
    both_pools['percent_change'] = (both_pools['p0.weth_to_usd_ratio'] - both_pools['p1.weth_to_usd_ratio']) / both_pools[['p0.weth_to_usd_ratio', 'p1.weth_to_usd_ratio']].min(axis=1)
    both_pools['total_gas_fees_usd'] = both_pools['p0.gas_fees_usd'] + both_pools['p1.gas_fees_usd']

    # Replace inf with NaN
    both_pools.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN values (which were originally inf)
    both_pools.dropna(inplace=True)
    return both_pools

@st.cache_data
def LSTM_preprocessing(both_pools):
    int_df = both_pools.select_dtypes(include=['datetime64[ns]', 'int64', 'float64'])
    int_df = int_df[['time', 'total_gas_fees_usd', 'percent_change']]

    int_df = int_df.sort_values(by='time', ascending=True)
    int_df = int_df.reset_index(drop=True)
    int_df.index = int_df.pop('time')

    columns_selected = ['percent_change']

    windowed_df = df_to_windowed_df(int_df,
                                    str(int_df.index[500]),
                                    str(int_df.index[-1]),
                                    n=50, columns=columns_selected, t_lag=5)

    dates, X, y = windowed_df_to_date_X_y(windowed_df, number_of_columns=len(columns_selected))
    return dates, X, y

@st.cache_data
def XGB_preprocessing(both_pools):
    int_df = both_pools.select_dtypes(include=['datetime64[ns]', 'int64', 'float64'])
    int_df = int_df[['time', 'total_gas_fees_usd']]
    df_3M = shift_column_by_time(int_df, 'time', 'total_gas_fees_usd', 5)
    df_3M.index = df_3M.pop('time')
    df_3M.index = pd.to_datetime(df_3M.index)

    num_lags = 9  # Number of lags to create
    for i in range(1, num_lags + 1):
        df_3M[f'lag_{i}'] = df_3M['total_gas_fees_usd'].shift(i)

    df_3M['rolling_mean_3'] = df_3M['total_gas_fees_usd'].rolling(window=3).mean()
    df_3M['rolling_mean_6'] = df_3M['total_gas_fees_usd'].rolling(window=6).mean()

    df_3M.dropna(inplace=True)
    lag_features = [f'lag_{i}' for i in range(1, num_lags + 1)]
    X_gas_test = df_3M[lag_features + ['rolling_mean_3', 'rolling_mean_6']]
    y_gas_test = df_3M['total_gas_fees_usd_label']

    return X_gas_test, y_gas_test

def Final_results_processing(dates_test, y_test, test_predictions, y_gas_test, y_gas_pred):
    df_percent_change = pd.DataFrame({
        'time': dates_test,
        'percent_change_actual': y_test,
        'percent_change_prediction': test_predictions
    })

    df_gas = y_gas_test.to_frame()
    df_gas = df_gas.reset_index()
    df_gas['gas_fees_prediction'] = y_gas_pred

    df_final = pd.merge(df_percent_change, df_gas, how='left', on='time')
    df_final = df_final.dropna()
    df_final['min_amount_to invest_prediction'] = df_final['gas_fees_prediction'] / (abs(df_final['percent_change_prediction']) - (0.003 + 0.0005))
    df_final['min_amount_to_invest_prediction_2'] = df_final.apply(
        lambda row: row['gas_fees_prediction'] /
                    (
                        (1 + abs(row['percent_change_prediction'])) * (1 - 0.003 if row['percent_change_prediction'] < 0 else 1 - 0.0005) -
                        (1 - 0.0005 if row['percent_change_prediction'] < 0 else 1 - 0.003)
                    ),
        axis=1
    )

    df_final['Profit'] = df_final.apply(
        lambda row: (row['min_amount_to_invest_prediction_2'] *
                     (1 + row['percent_change_actual']) * (1 - 0.003 if row['percent_change_prediction'] < 0 else 1 - 0.0005) -
                     row['min_amount_to_invest_prediction_2'] * (1 - 0.0005 if row['percent_change_prediction'] < 0 else 1 - 0.003) -
                     row['total_gas_fees_usd_label']),
        axis=1
    )

    df_final['Double_Check'] = df_final.apply(
        lambda row: (row['min_amount_to_invest_prediction_2'] *
                     (1 + abs(row['percent_change_prediction'])) * (1 - 0.003 if row['percent_change_prediction'] < 0 else 1 - 0.0005) -
                     row['min_amount_to_invest_prediction_2'] * (1 - 0.0005 if row['percent_change_prediction'] < 0 else 1 - 0.003) -
                     row['gas_fees_prediction']),
        axis=1
    )
    return df_final

def plot_net_gain_vs_threshold(df_final, ax):
    thresholds = list(range(1000, 40000, 1))
    net_gains = []

    for threshold in thresholds:
        df_gain = df_final[(df_final['Profit'] > 0) & 
                           (df_final['min_amount_to_invest_prediction_2'] > 0) & 
                           (df_final['min_amount_to_invest_prediction_2'] < threshold)]
        total_gain = df_gain['Profit'].sum()
        
        df_loss = df_final[(df_final['Profit'] < 0) & 
                           (df_final['min_amount_to_invest_prediction_2'] > 0) & 
                           (df_final['min_amount_to_invest_prediction_2'] < threshold)]
        total_loss = df_loss['Profit'].sum()
        
        net_gain = total_gain + total_loss
        net_gains.append(net_gain)

    ax.plot(thresholds, net_gains, marker='o')
    ax.set_title('Net Gain vs. Minimum Amount to Invest')
    ax.set_xlabel('Minimum Amount to Invest')
    ax.set_ylabel('Net Gain')
    ax.grid(True)

def display_current_arbitrage(df_final):
    now = datetime.now()
    time_difference = now - df_final['time'].iloc[-1]
    is_less_than_five_minutes = time_difference < timedelta(minutes=5)

    if is_less_than_five_minutes:
        if df_final['min_amount_to_invest_prediction_2'].iloc[-1] < 0:
            st.write(f"Arbitrage Opportunity does not exist five minutes after {df_final['time'].iloc[-1]}")
        else:
            st.write(f"Minimum amount to invest ${df_final['min_amount_to_invest_prediction_2'].iloc[-1]:.2f} five minutes after {df_final['time'].iloc[-1]}")
    else:
        st.write(f"Last Data point received from query was at {df_final['time'].iloc[-1]}")
        st.write("Data queried is greater than five minutes old, unable to provide minimum amount to invest")

def df_to_windowed_df(dataframe, first_datetime_str, last_datetime_str, n=3, columns=[], t_lag=0.3):
    first_date = datetime.strptime(first_datetime_str, '%Y-%m-%d %H:%M:%S')
    last_date = datetime.strptime(last_datetime_str, '%Y-%m-%d %H:%M:%S')

    target_datetime = first_date

    dates = []
    X, Y = [], []

    last_time = False
    while True:
        deltas = [timedelta(minutes=t_lag + i * 1) for i in range(n)]
        timestamps = [target_datetime - delta for delta in deltas]

        timestamp_df = pd.DataFrame({'Timestamp': timestamps})
        dataframe.index = pd.to_datetime(dataframe.index)
        timestamp_df = timestamp_df.sort_values(by='Timestamp')
        dataframe = dataframe.sort_index()
        df_subset = pd.merge_asof(timestamp_df, dataframe, left_on='Timestamp', right_index=True, direction='backward')

        if df_subset.isnull().values.any():
            print(f'Error: Could not find required timestamps for datetime {target_datetime}')
            return

        values = df_subset[columns].to_numpy()
        x = values.flatten()  # Flatten to create a single array of features

        target_df = pd.DataFrame({'Timestamp': [target_datetime]})
        target_df = target_df.sort_values(by='Timestamp')
        df_merge = dataframe.reset_index()
        target_value = pd.merge_asof(target_df, df_merge, left_on='Timestamp', right_on='time', direction='backward')
        if target_value.isnull().values.any():
            print(f'Error: No target value for datetime {target_datetime}')
            return

        y = target_value['percent_change'].values[0]

        dates.append(target_value['time'][0])
        X.append(x)
        Y.append(y)

        next_delta = timedelta(minutes=1)
        next_datetime = target_datetime + next_delta

        if last_time:
            break

        target_datetime = next_datetime

        if target_datetime > last_date:
            last_time = True

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates

    X = np.array(X)  # Convert X to numpy array
    for i in range(n):
        for j, feature in enumerate(columns):
            ret_df[f'Target-{n-i}-{feature}'] = X[:, i]

    ret_df['Target'] = Y

    return ret_df

def windowed_df_to_date_X_y(windowed_dataframe, number_of_columns=1):
    df_as_np = windowed_dataframe.to_numpy()

    dates = df_as_np[:, 0]

    middle_matrix = df_as_np[:, 1:-1].reshape((len(dates), -1, number_of_columns))

    Y = df_as_np[:, -1]  # Assuming last column is the target 'percent_change'

    return dates, middle_matrix.astype(np.float32), Y.astype(np.float32)

def shift_column_by_time(df, time_col, value_col, shift_minutes):
    # Ensure 'time_col' is in datetime format
    df[time_col] = pd.to_datetime(df[time_col])

    # Sort the DataFrame by time
    df = df.sort_values(by=time_col).reset_index(drop=True)

    # Create an empty column for the shifted values
    df[f'{value_col}_label'] = None

    # Iterate over each row and find the appropriate value at least 5 minutes later
    for i in range(len(df)):
        current_time = df.loc[i, time_col]
        future_time = current_time + pd.Timedelta(minutes=shift_minutes)

        # Find the first row where the time is greater than or equal to the future_time
        future_row = df[df[time_col] >= future_time]
        if not future_row.empty:
            df.at[i, f'{value_col}_label'] = future_row.iloc[0][value_col]

    return df

# Update the load_model function
@st.cache_resource
def load_model(model_name):
    models_dir = os.path.join(os.getcwd(), 'models')
    model_path = os.path.join(models_dir, f'{model_name}_final.pkl')
    
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

# Main Streamlit app
st.title("Arbitrage Playground")
st.write(
    "Use this app to analyze cross-liquidity pool arbitrage scenarios in WETH/USDC liquidity pools."
)

# Sidebar
st.sidebar.header("API Configuration")

# API key input
api_key = st.sidebar.text_input("Etherscan API Key", "YOUR_API_KEY_HERE")
address = st.sidebar.text_input("Contract Address", "0x7bea39867e4169dbe237d55c8242a8f2fcdcc387")

if st.button("Run Analysis"):
    with st.spinner("Fetching and processing data..."):
        # Fetch and process data for both pools
        p0, transactions = etherscan_request('tokentx', api_key, address='0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8')
        p1, transactions1 = etherscan_request('tokentx', api_key, address='0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640')
        
        both_pools = merge_pool_data(p0, p1)
        
        # LSTM Preprocessing
        dates_test, X_test, y_test = LSTM_preprocessing(both_pools)
        
        # XGB Preprocessing
        X_gas_test, y_gas_test = XGB_preprocessing(both_pools)
        
        # Run LSTM model
        with st.spinner("Running LSTM model..."):
            LSTM = load_model("LSTM")
            test_predictions = LSTM.predict(X_test).flatten()
            mse = mean_squared_error(y_test, test_predictions, squared=False)
            r2 = r2_score(y_test, test_predictions)
            
            st.subheader("LSTM Model Results")
            st.write(f"Mean Squared Error: {mse:.4f}")
            st.write(f"R² Score: {r2:.4f}")
        
        # Run XGB model
        with st.spinner("Running XGB model..."):
            XGB = load_model("XGB")
            y_gas_pred = XGB.predict(X_gas_test)
            mse_gas = mean_squared_error(y_gas_test, y_gas_pred, squared=False)
            r2_gas = r2_score(y_gas_test, y_gas_pred)
            
            st.subheader("XGB Model Results")
            st.write(f"Mean Squared Error: {mse_gas:.4f}")
            st.write(f"R² Score: {r2_gas:.4f}")
        
        # process final results
        df_final = Final_results_processing(dates_test, y_test, test_predictions, y_gas_test, y_gas_pred)
        
        # display results
        st.subheader("Arbitrage Opportunities")
        df_gain = df_final[((df_final['Profit'] > 0) & 
                            (df_final['min_amount_to_invest_prediction_2'] > 0) & 
                            (df_final['min_amount_to_invest_prediction_2'] < 5000))]
        
        st.write(df_gain)
        
        total_gain = df_gain['Profit'].sum()
        st.write(f"Total Potential Gain: ${total_gain:.2f}")
        
        # plot Net Gain vs. Minimum Amount to Invest
        st.subheader("Net Gain vs. Minimum Amount to Invest")
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_net_gain_vs_threshold(df_final, ax)
        st.pyplot(fig)
        
        # display current arbitrage opportunity
        display_current_arbitrage(df_final)
