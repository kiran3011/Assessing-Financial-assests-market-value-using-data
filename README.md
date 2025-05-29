# Assessing-Financial-assests-market-value-using-data
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Forecasting models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import MeanAbsoluteError
import scipy.stats as stats

# --- Reproducibility ---
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

st.set_page_config(layout="wide")
st.title("📈 Stock Analysis Dashboard")

# --- Sidebar UI ---
st.sidebar.header("📁 Upload & Navigation")
uploaded_file = st.sidebar.file_uploader("Upload a stock CSV file", type="csv")

page_choice = st.sidebar.radio("Select Dashboard Section", ["Forecasting", "Behavioral Dashboard"])

if page_choice == "Forecasting":
    model_choice = st.sidebar.selectbox("Select a Forecasting Model", ["SARIMAX", "LSTM"])

# --- Behavioral Indicator Functions ---
def compute_rsi(series, period=14, epsilon=1e-8):
    """
    Computes the Relative Strength Index (RSI) of a given series.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + epsilon)
    return 100 - (100 / (1 + rs))

def get_signals(df):
    """
    Identifies buy and sell signals based on both RSI and Stochastic Oscillator.
    
    Args:
        df (pd.DataFrame): DataFrame containing RSI, %K and %D columns
        
    Returns:
        list: List of tuples containing (date, RSI_signal, Stochastic_signal, Combined_signal)
    """
    signals = []
    for i in range(1, len(df)):
        rsi = df['RSI'].iloc[i]
        rsi_prev = df['RSI'].iloc[i - 1]
        k = df['%K'].iloc[i]
        d = df['%D'].iloc[i]

        rsi_signal = 'Buy' if rsi_prev < 30 and rsi >= 30 else 'Sell' if rsi_prev > 70 and rsi <= 70 else None
        stoch_signal = 'Buy' if k < 20 and d < 20 else 'Sell' if k > 80 and d > 80 else None

        if rsi_signal == stoch_signal and rsi_signal is not None:
            combined_signal = rsi_signal
        else:
            combined_signal = None

        if combined_signal:
            signals.append((df.index[i], rsi_signal or '-', stoch_signal or '-', combined_signal))
    return signals

def compute_stochastic_oscillator(df, k_period=14, d_period=3):
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    df['%K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['%D'] = df['%K'].rolling(window=d_period).mean()
    return df

def add_behavioral_indicators(df):
    """
    Adds several technical analysis indicators to the DataFrame.
    """
    df = df.copy()
    if 'Close' in df.columns and 'Volume' in df.columns and 'High' in df.columns and 'Low' in df.columns:
        df['RSI'] = compute_rsi(df['Close'])
        df = compute_stochastic_oscillator(df)
        return df
    else:
        st.error("Error: 'Close', 'Volume', 'High', and 'Low' columns are required in the CSV file.")
        return None

# --- Main App Logic ---
if uploaded_file is not None:
    stock_name = os.path.splitext(uploaded_file.name)[0].capitalize()
    try:
        df_raw = pd.read_csv(uploaded_file)
        if 'Date' not in df_raw.columns:
            st.error("Error: 'Date' column is required in the CSV file.")
            st.stop()
        df_raw['Date'] = pd.to_datetime(df_raw['Date'], dayfirst=True, errors='coerce')
        if df_raw['Date'].isnull().any():
            st.error("Error: 'Date' column contains invalid date values.")
            st.stop()
        df_raw = df_raw.sort_values('Date').set_index('Date').dropna()
        df_raw = add_behavioral_indicators(df_raw)
        if df_raw is None:
            st.stop()
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    if df_raw is not None:
        if page_choice == "Forecasting":
            st.subheader(f"{stock_name} Forecasting — {model_choice}")

            if model_choice == "SARIMAX":
                df = df_raw.copy()
                if 'Close' in df.columns:
                    exog_vars = df[['RSI', '%K', '%D']].fillna(method='bfill').fillna(method='ffill')
                    model = SARIMAX(df['Close'], exog=exog_vars, order=(0, 1, 0), seasonal_order=(10, 1, 0, 7),
                                     enforce_stationarity=False, enforce_invertibility=False)
                    try:
                        sarimax_result = model.fit(disp=False)

                        in_sample_pred = sarimax_result.predict(start=1, end=len(df)-1, exog=exog_vars[1:])
                        actual = df['Close'][1:]
                        in_sample_pred.index = df.index[1:]

                        mse = mean_squared_error(actual, in_sample_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(actual, in_sample_pred)
                        last_date = df.index[-1]
                        future_exog = pd.DataFrame(
                          np.tile(exog_vars.iloc[-1].values, (30, 1)),
                          columns=exog_vars.columns,
                          index=pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=30)
                        )

                        forecast_30 = sarimax_result.get_forecast(steps=30, exog=future_exog)
                        forecast_mean = forecast_30.predicted_mean
                        conf_int = forecast_30.conf_int()
                        last_date = df.index[-1]
                        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=30)

                        forecast_df = pd.DataFrame({
                            'Forecast': forecast_mean.values,
                            'Lower_CI': conf_int.iloc[:, 0].values,
                            'Upper_CI': conf_int.iloc[:, 1].values
                        }, index=future_dates)

                        residuals = actual - in_sample_pred

                        tab1, tab2, tab3 = st.tabs(["📊 Forecast", "📉 Residuals", "📋 Metrics"])

                        with tab1:
                            df_last_year = df[df.index >= df.index[-1] - pd.DateOffset(years=1)]
                            in_sample_pred_last_year = in_sample_pred[in_sample_pred.index >= df_last_year.index[0]]

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_last_year.index, y=df_last_year['Close'], name='Actual'))
                            fig.add_trace(go.Scatter(x=in_sample_pred_last_year.index, y=in_sample_pred_last_year,
                                                    name='Fitted Prediction', line=dict(dash='dash')))
                            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'],
                                                    name='30-Day Forecast', line=dict(dash='dot', color='red')))
                            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Upper_CI'], line=dict(width=0), showlegend=False))
                            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Lower_CI'],
                                                    fill='tonexty', name='Confidence Interval', line=dict(width=0), fillcolor='rgba(255,182,193,0.3)'))
                            fig.update_layout(height=600,title=f"{stock_name} — SARIMAX Forecast", hovermode="x unified")

                            min_date = df_last_year.index.min().date()
                            max_date = forecast_df.index.max().date()
                            selected_range = st.slider("Select Date Range", min_date, max_date, (min_date, max_date), format="YYYY-MM-DD")
                            fig.update_xaxes(range=[pd.to_datetime(selected_range[0]), pd.to_datetime(selected_range[1])])

                            st.plotly_chart(fig, use_container_width=True)

                        with tab2:
                            fig_resid = px.line(x=residuals.index, y=residuals.values, title=f"{stock_name} SARIMAX Residuals")
                            fig_resid.add_hline(y=0, line_dash="dash", line_color="black")
                            st.plotly_chart(fig_resid, use_container_width=True)

                        with tab3:
                            epsilon = 1e-8  # Avoid division by zero
                            mape = np.mean(np.abs((actual - in_sample_pred) / (actual + epsilon))) * 100
                            st.metric("MSE", f"{mse:.4f}")
                            st.metric("MAPE", f"{mape:.2f}%")
                            st.metric("R²", f"{r2:.4f}")
                            st.dataframe(forecast_df)

                    except Exception as e:
                        st.error(f"Error fitting SARIMAX model: {e}")
                else:
                    st.error("Error: 'Close' column not found for SARIMAX.")

            elif model_choice == "LSTM":
                df = df_raw.copy().dropna()
                if 'Close' in df.columns:
                    scaler = MinMaxScaler()
                    scaled = scaler.fit_transform(df)

                    close_col_index = df.columns.get_loc('Close')

                    def create_sequences(data, seq_len=90, target_col_index=close_col_index):
                        X, y = [], []
                        for i in range(seq_len, len(data)):
                            X.append(data[i-seq_len:i])
                            y.append(data[i, target_col_index])
                        return np.array(X), np.array(y)

                    X, y = create_sequences(scaled)
                    split = int(len(X) * 0.8)
                    X_train, X_test = X[:split], X[split:]
                    y_train, y_test = y[:split], y[split:]

                    model = Sequential([
                        LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                        Dropout(0.3),
                        LSTM(64, return_sequences=True),
                        Dropout(0.3),
                        LSTM(32),
                        Dropout(0.3),
                        Dense(1)
                    ])
                    model.compile(optimizer='adam', loss=MeanAbsoluteError())
                    try:
                        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

                        pred = model.predict(X_test)
                        dummy = np.zeros((len(pred), scaled.shape[1]))
                        dummy[:, close_col_index] = pred[:, 0]
                        predicted = scaler.inverse_transform(dummy)[:, close_col_index]

                        dummy_actual = np.zeros((len(y_test), scaled.shape[1]))
                        dummy_actual[:, close_col_index] = y_test
                        actual = scaler.inverse_transform(dummy_actual)[:, close_col_index]

                        mse = mean_squared_error(actual, predicted)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(actual, predicted)

                        future_steps = 30
                        last_sequence = scaled[-90:]
                        future_preds = []

                        for _ in range(future_steps):
                            seq_input = np.expand_dims(last_sequence, axis=0)
                            next_pred = model.predict(seq_input)[0, 0]
                            future_preds.append(next_pred)
                            next_input = last_sequence[-1].copy()
                            next_input[close_col_index] = next_pred
                            last_sequence = np.append(last_sequence[1:], [next_input], axis=0)

                        dummy_future = np.zeros((future_steps, scaled.shape[1]))
                        dummy_future[:, close_col_index] = future_preds
                        future_forecast = scaler.inverse_transform(dummy_future)[:, close_col_index]

                        residuals = actual - predicted
                        residual_std = np.std(residuals)
                        confidence_interval = 1.96  # 95% confidence interval
                        lower_bound = future_forecast - confidence_interval * residual_std
                        upper_bound = future_forecast + confidence_interval * residual_std
                        last_date = df.index[-1]
                        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)
                        future_df = pd.DataFrame({'Forecast_Close': future_forecast, 'Lower_CI': lower_bound, 'Upper_CI': upper_bound}, index=future_dates)
                        dates = df.index[-len(actual):]

                        tab1, tab2, tab3 = st.tabs(["📊 Forecast", "📉 Residuals", "📋 Metrics"])

                        with tab1:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=dates, y=actual, name='Actual'))
                            fig.add_trace(go.Scatter(x=dates, y=predicted, name='Predicted', line=dict(color='green')))
                            fig.add_trace(go.Scatter(x=future_dates, y=future_forecast, name='30-Day Forecast', line=dict(dash='dot', color='red')))
                            fig.add_trace(go.Scatter(x=future_dates, y=upper_bound, showlegend=False, line=dict(width=0)))
                            fig.add_trace(go.Scatter(x=future_dates, y=lower_bound, fill='tonexty', name='Confidence Interval', line=dict(width=0), fillcolor='rgba(255,182,193,0.3)'))
                            fig.update_layout(height=600,title=f"{stock_name} — LSTM Forecast", hovermode="x unified")

                            min_date = dates.min().date() if len(dates) > 0 else future_dates.min().date()
                            max_date = future_dates.max().date()
                            selected_range = st.slider("Select Date Range", min_date, max_date, (min_date, max_date), format="YYYY-MM-DD")
                            fig.update_xaxes(range=[pd.to_datetime(selected_range[0]), pd.to_datetime(selected_range[1])])

                            st.plotly_chart(fig, use_container_width=True)

                        with tab2:
                            fig_resid = px.line(x=dates, y=residuals, title=f"{stock_name} LSTM Residuals")
                            fig_resid.add_hline(y=0, line_dash="dash", line_color="black")
                            st.plotly_chart(fig_resid, use_container_width=True)

                        with tab3:
                            epsilon = 1e-8  # Avoid division by zero
                            mape = np.mean(np.abs((actual - predicted) / (actual + epsilon))) * 100
                            st.metric("MSE", f"{mse:.4f}")
                            st.metric("MAPE", f"{mape:.2f}%")
                            st.metric("R²", f"{r2:.4f}")
                            st.dataframe(future_df)

                    except Exception as e:
                        st.error(f"Error training or predicting with LSTM model: {e}")
                else:
                    st.error("Error: 'Close' column not found for LSTM.")

        elif page_choice == "Behavioral Dashboard":
            st.subheader(f"{stock_name} — Behavioral Indicators")

            min_date = df_raw.index.min().date()
            max_date = df_raw.index.max().date()
            selected_range = st.slider("Select Date Range", min_value=min_date, max_value=max_date,
                                        value=(min_date, max_date), format="YYYY-MM-DD")

            df_filtered = df_raw.loc[
                (df_raw.index >= pd.to_datetime(selected_range[0])) &
                (df_raw.index <= pd.to_datetime(selected_range[1]))
            ].copy()

            signals = get_signals(df_filtered)

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=("RSI (14)", "Stochastic Oscillator (%K, %D)"),
                                vertical_spacing=0.1)

            if 'RSI' in df_filtered.columns:
                fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['RSI'], name='RSI', line=dict(color='orange')), row=1, col=1)
                fig.add_hline(y=70, line_dash='dash', line_color='red', row=1, col=1)
                fig.add_hline(y=30, line_dash='dash', line_color='green', row=1, col=1)

            if '%K' in df_filtered.columns and '%D' in df_filtered.columns:
                fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['%K'], name='%K', line=dict(color='blue')), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['%D'], name='%D', line=dict(color='red')), row=2, col=1)
                fig.add_hline(y=80, line_dash='dash', line_color='red', row=2, col=1)
                fig.add_hline(y=20, line_dash='dash', line_color='green', row=2, col=1)

            # Add signals to the plot
            for date, rsi_signal, stoch_signal, combined_signal in signals:
                val_rsi = df_filtered.loc[date, 'RSI']
                val_k = df_filtered.loc[date, '%K']
                color = 'green' if combined_signal == 'Buy' else 'red' if combined_signal == 'Sell' else 'gray'
                arrow = 'arrow-up' if combined_signal == 'Buy' else 'arrow-down' if combined_signal == 'Sell' else 'circle'

                fig.add_trace(go.Scatter(x=[date], y=[val_rsi], mode='markers+text',
                                         marker=dict(symbol=arrow, color=color, size=12),
                                         text=[combined_signal], textposition="top center",
                                         name=f"{combined_signal} Signal"), row=1, col=1)

                fig.add_trace(go.Scatter(x=[date], y=[val_k], mode='markers+text',
                                         marker=dict(symbol=arrow, color=color, size=12),
                                         text=[combined_signal], textposition="top center",
                                         name=f"{combined_signal} Signal"), row=2, col=1)

            # --- Event Annotations ---
            event_annotations = {
                "Sainsburys": [
                    ("2020-03-01", "COVID-19 panic buying"),
                    ("2020-04-01", "Delivery slots ramped"),
                    ("2020-07-01", "Q1 results: Grocery +10.5%, Digital sales doubled"),
                    ("2021-07-01", "Fuel shortage logistics strain"),
                    ("2021-10-01", "Supply chain crisis"),
                    ("2022-02-01", "Ukraine conflict"),
                    ("2023-03-01", "Strong FY results"),
                    ("2024-10-01", "Digital transformation met"),
                    ("2025-04-01", "Profit tops £1bn"),
                ],
                "Tesco": [
                    ("2020-05-01", "Q1 lockdown demand spike"),
                    ("2020-07-01", "20000 employee, Mature sstock scheme"),
                    ("2020-11-01", "COVID staff bonus boost"),
                    ("2021-03-01", "Interim profits up 30%"),
                    ("2022-01-01", "Inflation concerns rise"),
                    ("2022-04-01", "Cost of living crisis"),
                    ("2023-02-01", "Strong Christmas results"),
                    ("2024-08-01", "Automation savings plan"),
                    ("2025-01-01", "Aldi price match expanded"),
                    ("2025-04-01", "Store expansion strategy")
                ],
                "Ocado": [
                    ("2020-05-01", "Online grocery surge"),
                    ("2020-09-01", "Joins FTSE 100"),
                    ("2021-02-01", "Warehouse fire"),
                    ("2022-01-01", "Auchan partnership"),
                    ("2022-07-01", "Losses widen"),
                    ("2023-11-01", "Holiday demand boost"),
                    ("2024-03-01", "AI robotic upgrade"),
                    ("2025-04-01", "AI warehouse plans")
                ]
            }

            if stock_name in event_annotations:
                for date_str, label in event_annotations[stock_name]:
                    try:
                        date_obj = pd.to_datetime(date_str)
                        if df_filtered.index.min() <= date_obj <= df_filtered.index.max():
                            fig.add_vline(
                                x=date_obj,
                                line_width=1.5,
                                line_dash="dash",
                                line_color="cyan",
                            )
                            fig.add_annotation(
                                x=date_obj,
                                yref="paper",
                                y=1.0,
                                text=label,
                                showarrow=True,
                                arrowhead=1,
                                ax=0,
                                ay=-40,
                                font=dict(size=10, color="cyan"),
                                bgcolor="rgba(0,255,255,0.1)"
                            )
                    except Exception as e:
                        st.warning(f"⚠️ Could not annotate event on {date_str}: {e}")

            fig.update_layout(height=600, title=f"{stock_name} Behavioral Indicators", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # Signal Table
            st.markdown("### 📋 Signal Table (Grouped by RSI & Stochastic)")
            signal_df = pd.DataFrame(signals, columns=["Date", "RSI_Signal", "Stochastic_Signal", "Combined_Signal"])
            st.dataframe(signal_df)

            csv = signal_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Signals as CSV", csv, f"{stock_name.lower()}_signals.csv", "text/csv")

else:
    st.info("Please upload a stock CSV file to begin analysis.")
