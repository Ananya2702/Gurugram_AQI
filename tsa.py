import subprocess
import sys
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

@st.cache_resource
def install_requirements():
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Call the function once at the start of the app
install_requirements()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from statsmodels.tsa.stattools import adfuller
import math

# Load AQI data
st.title("AQI Forecasting for Gurugram Sector 51")

@st.cache_data
def load_data():
    data= pd.read_csv("sector_51_daily_aqi.csv", index_col=0)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

data = load_data()  

tab1, tab2, tab3, tab4, tab5  = st.tabs(["Exploratorary Data Analysis", "Stationarity Tests","ACF and PACF plots","Exponential Smoothing","Forecasting"])


with tab1:
    # Data Preview
    st.subheader("AQI Data Preview")
    st.write(data.head())

    # Fill missing values
    data['AQI'] = data['AQI'].fillna(method='ffill')
    data['Date'] = pd.to_datetime(data['Date'])


    # Set the default start and end dates for the slider (converting to date format)
    start_date = data['Date'].min().date()
    end_date = data['Date'].max().date()

    # Add a date range slider with dates in 'date' format
    date_range = st.slider(
        "Select Date Range",
        min_value=start_date,
        max_value=end_date,
        value=(start_date, end_date)  # default to full range
    )

    # Filter the data based on the selected date range
    filtered_data = data[(data['Date'] >= pd.to_datetime(date_range[0])) & (data['Date'] <= pd.to_datetime(date_range[1]))]

    # Plot using Plotly
    st.subheader("Interactive Daily AQI Time Plot")
    fig = px.line(
        filtered_data,
        x='Date',
        y='AQI',
        title="AQI of Gurugram Sector 51",
        labels={'AQI': 'AQI Value'},
    )

    # Customize the layout for better readability
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="AQI value",
        hovermode="x unified",  # Hover information shown along the x-axis
        template="plotly_white"
    )

    st.plotly_chart(fig)

    # Monthly AQI plot
    st.subheader("Monthly Average AQI")
    monthly_avg = data.resample('M', on='Date').mean()['AQI']

    # Convert the resampled data to a DataFrame for Plotly compatibility
    monthly_avg_df = monthly_avg.reset_index()

    # Plot using Plotly
    fig = px.line(
        monthly_avg_df,
        x='Date',
        y='AQI',
        title="Monthly Average AQI of Gurugram Sector 51",
        labels={'AQI': 'Average AQI'}
    )

    # Customize the layout for better readability
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Average AQI",
        hovermode="x unified",
        template="plotly_white"
    )

    st.plotly_chart(fig)


    def AQ(x):
        if 0 <= x <= 50:
            return 'Good'
        if 50 < x <= 100:
            return 'Satisfactory'
        if 100 < x <= 200:
            return 'Moderate'
        if 200 < x <= 300:
            return 'Poor'
        if 300 < x <= 400:
            return 'Very Poor'
        if x > 400:
            return 'Severe'
        
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Air Quality Type'] = data['AQI'].apply(AQ)

    # Define the order for the air quality types
    air_quality_order = ['Good', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    color_map = {
        'Good': 'green',
        'Moderate': 'yellow',
        'Poor': 'orange',
        'Very Poor': 'brown',
        'Severe': 'red'
    }

    # Calculate the count of each air quality type
    air_quality_counts = data['Air Quality Type'].value_counts().reindex(air_quality_order).fillna(0)

    # Convert to a DataFrame for Plotly compatibility
    air_quality_df = air_quality_counts.reset_index()
    air_quality_df.columns = ['Air Quality Type', 'Count']

    # Create an interactive bar chart using Plotly with custom colors
    st.subheader("Air Quality Type Distribution")
    fig = px.bar(
        air_quality_df,
        x='Air Quality Type',
        y='Count',
        title="Distribution of Air Quality Types in Gurugram Sector 51",
        labels={'Count': 'Number of Days'},
        color='Air Quality Type',  # Map the color to the Air Quality Type
        color_discrete_map=color_map  # Use the custom color mapping
    )

    # Customize layout
    fig.update_layout(
        xaxis=dict(categoryorder='array', categoryarray=air_quality_order),
        yaxis_title="Number of Days",
        template="plotly_white"
    )

    st.plotly_chart(fig)

    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month

    # Group by Year and Month to get the average AQI
    monthly_avg_aqi = data.groupby(['Year', 'Month'])['AQI'].mean().unstack(level=0)

    # Prepare the text labels as strings formatted to one decimal place
    text_labels = monthly_avg_aqi.applymap(lambda x: f"{x:.1f}" if pd.notnull(x) else "")

    # Plotting the heatmap using Plotly
    st.subheader("Average Monthly AQI by Year")
    fig = go.Figure(data=go.Heatmap(
        z=monthly_avg_aqi.values,
        x=monthly_avg_aqi.columns,  # Years
        y=monthly_avg_aqi.index,     # Months
        colorscale="YlOrRd",
        colorbar=dict(title="AQI"),
        hoverongaps=False,
        text=text_labels.values,  # Adding the text labels
        texttemplate="%{text}"    # Display the text on each cell
    ))

    # Update layout for better readability
    fig.update_layout(
        title="Average Monthly AQI by Year in Gurugram (2020-2024)",
        xaxis_title="Year",
        yaxis_title="Month",
        yaxis=dict(tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    )

    st.plotly_chart(fig)
    st.write('''The plot of monthly mean AQI reveals a cyclical pattern, with certain months consistently experiencing higher AQI levels, which may be influenced by seasonal factors like weather, agricultural activities, or traffic patterns.

    We know that burning of crops in Punjab during Diwali greatly worsens the Air Quality in Delhi, Delhi NCR and neighbouring areas.

    The above plot clearly validates this by showing a spike around the months of October/November''')


    # Extract the month from the date and calculate the monthly average AQI
    data['Month'] = data['Date'].dt.month
    monthly_avg = data.groupby('Month')['AQI'].mean().reset_index()

    # Map numeric months to month names for better readability
    monthly_avg['Month'] = monthly_avg['Month'].apply(lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1])

    # Plotting with Plotly
    st.subheader("Average AQI by Month")
    fig = px.line(
        monthly_avg,
        x='Month',
        y='AQI',
        title="Average AQI by Month",
        markers=True,
        labels={'AQI': 'Average AQI'},
        line_shape='spline'  # Smooth the line for better visualization
    )

    # Customize the layout
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Average AQI",
        template="plotly_white"
    )

    st.plotly_chart(fig)


    # Decompose the time series
    decomposition = seasonal_decompose(data['AQI'], model='additive', period=365)

    # Plot each component in a separate Plotly figure
    st.subheader("Time Series Decomposition Components")

    # Original Series Plot
    fig_original = go.Figure()
    fig_original.add_trace(go.Scatter(x=data['Date'], y=data['AQI'], mode='lines', name='Original', line=dict(color='blue')))
    fig_original.update_layout(title="Original Time Series", xaxis_title="Date", yaxis_title="AQI", template="plotly_white")
    st.plotly_chart(fig_original, use_container_width=True, key="original_series")

    # Trend Component Plot
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=data['Date'], y=decomposition.trend, mode='lines', name='Trend', line=dict(color='orange')))
    fig_trend.update_layout(title="Trend Component", xaxis_title="Date", yaxis_title="Trend", template="plotly_white")
    st.plotly_chart(fig_trend, use_container_width=True, key="trend_component")

    # Seasonal Component Plot
    fig_seasonal = go.Figure()
    fig_seasonal.add_trace(go.Scatter(x=data['Date'], y=decomposition.seasonal, mode='lines', name='Seasonal', line=dict(color='green')))
    fig_seasonal.update_layout(title="Seasonal Component", xaxis_title="Date", yaxis_title="Seasonality", template="plotly_white")
    st.plotly_chart(fig_seasonal, use_container_width=True, key="seasonal_component")

    # Residuals Component Plot
    fig_residual = go.Figure()
    fig_residual.add_trace(go.Scatter(x=data['Date'], y=decomposition.resid, mode='lines', name='Residuals', line=dict(color='red')))
    fig_residual.update_layout(title="Residuals", xaxis_title="Date", yaxis_title="Residuals", template="plotly_white")
    st.plotly_chart(fig_residual, use_container_width=True, key="residual_component")


    data1 = data.set_index('Date')
    monthly_aqi = data1['AQI'].resample('M').mean()
    decomposition = seasonal_decompose(monthly_aqi, model='additive')

    # Plot each component in a separate Plotly figure
    st.subheader("Monthly AQI Time Series Decomposition")

    # Observed Series Plot
    fig_observed = go.Figure()
    fig_observed.add_trace(go.Scatter(x=monthly_aqi.index, y=decomposition.observed, mode='lines', name='Observed', line=dict(color='blue')))
    fig_observed.update_layout(title="Observed Monthly AQI", xaxis_title="Date", yaxis_title="AQI", template="plotly_white")
    st.plotly_chart(fig_observed, use_container_width=True, key="observed_monthly")

    # Trend Component Plot
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=monthly_aqi.index, y=decomposition.trend, mode='lines', name='Trend', line=dict(color='orange')))
    fig_trend.update_layout(title="Trend Component", xaxis_title="Date", yaxis_title="Trend", template="plotly_white")
    st.plotly_chart(fig_trend, use_container_width=True, key="trend_monthly")

    # Seasonal Component Plot
    fig_seasonal = go.Figure()
    fig_seasonal.add_trace(go.Scatter(x=monthly_aqi.index, y=decomposition.seasonal, mode='lines', name='Seasonal', line=dict(color='green')))
    fig_seasonal.update_layout(title="Seasonal Component", xaxis_title="Date", yaxis_title="Seasonality", template="plotly_white")
    st.plotly_chart(fig_seasonal, use_container_width=True, key="seasonal_monthly")

    # Residuals Component Plot
    fig_residual = go.Figure()
    fig_residual.add_trace(go.Scatter(x=monthly_aqi.index, y=decomposition.resid, mode='lines', name='Residuals', line=dict(color='red')))
    fig_residual.update_layout(title="Residual Component", xaxis_title="Date", yaxis_title="Residuals", template="plotly_white")
    st.plotly_chart(fig_residual, use_container_width=True, key="residual_monthly")

    st.write('''The decomposition plot reveals the following components:

    Observed: Reflects the original monthly mean AQI series.

    Trend: Shows an underlying trend in AQI levels over time, indicating potential long-term shifts in air quality.

    Seasonal: Exhibits a clear seasonal pattern, with recurring peaks and troughs, likely due to seasonal factors impacting AQI.

    Residual: Captures random fluctuations not explained by the trend or seasonal components.

    This decomposition confirms the presence of seasonal and trend components, both essential for building an accurate forecasting model.''')


with tab2:
    # Prepare the daily AQI data by setting 'Date' as the index
    daily_aqi = data[['Date', 'AQI']]
    daily_aqi.set_index('Date', inplace=True)

    # Calculate the monthly average AQI for the ADF test
    monthly_aqi = daily_aqi['AQI'].resample('M').mean()

    # Perform the Augmented Dickey-Fuller (ADF) test on the monthly mean AQI data
    adf_test = adfuller(monthly_aqi.dropna())
    adf_test_results = {
        "ADF Statistic": [adf_test[0]],
        "p-value": [adf_test[1]],
        "Critical Value (1%)": [adf_test[4]['1%']],
        "Critical Value (5%)": [adf_test[4]['5%']],
        "Critical Value (10%)": [adf_test[4]['10%']]
    }

    # Display the ADF test results as a DataFrame
    st.subheader("Augmented Dickey-Fuller (ADF) Test Results")
    st.markdown("### Monthly")
    adf_results_df = pd.DataFrame(adf_test_results)
    st.write(adf_results_df)

    # Interpret the result based on the p-value
    if adf_test[1] < 0.05:
        st.write("The time series is **stationary** (reject the null hypothesis of the ADF test).")
    else:
        st.write("The time series is **non-stationary** (fail to reject the null hypothesis of the ADF test).")

    # Perform the Augmented Dickey-Fuller (ADF) test on the monthly mean AQI data
    adf_test = adfuller(daily_aqi.dropna())
    adf_test_results = {
        "ADF Statistic": [adf_test[0]],
        "p-value": [adf_test[1]],
        "Critical Value (1%)": [adf_test[4]['1%']],
        "Critical Value (5%)": [adf_test[4]['5%']],
        "Critical Value (10%)": [adf_test[4]['10%']]
    }
    
    # Display the ADF test results as a DataFrame
    st.markdown("### Daily")
    adf_results_df = pd.DataFrame(adf_test_results)
    st.write(adf_results_df)

    # Interpret the result based on the p-value
    if adf_test[1] < 0.05:
        st.write("The time series is **stationary** (reject the null hypothesis of the ADF test).")
    else:
        st.write("The time series is **non-stationary** (fail to reject the null hypothesis of the ADF test).")
    
    
    monthly_aqi_diff = monthly_aqi.diff().dropna()
    st.markdown("### Differenced Monthly")
    # Perform the Augmented Dickey-Fuller (ADF) test on the monthly mean AQI data
    adf_test = adfuller(monthly_aqi_diff.dropna())
    adf_test_results = {
        "ADF Statistic": [adf_test[0]],
        "p-value": [adf_test[1]],
        "Critical Value (1%)": [adf_test[4]['1%']],
        "Critical Value (5%)": [adf_test[4]['5%']],
        "Critical Value (10%)": [adf_test[4]['10%']]
    }
    adf_results_df = pd.DataFrame(adf_test_results)
    st.write(adf_results_df)

    # Interpret the result based on the p-value
    if adf_test[1] < 0.05:
        st.write("The time series is **stationary** (reject the null hypothesis of the ADF test).")
    else:
        st.write("The time series is **non-stationary** (fail to reject the null hypothesis of the ADF test).")
  
        
with tab3:
    # Prepare the daily AQI data
    data['Date'] = pd.to_datetime(data['Date'])
    daily_aqi = data.set_index('Date')['AQI']

    # Resample monthly
    monthly_aqi = daily_aqi.resample('M').mean()

    # ACF/PACF plot settings
    st.subheader("ACF and PACF Plots")

    # Frequency selection: Monthly or Daily
    frequency = st.selectbox("Select Frequency", options=["Daily", "Monthly"])

    # Lags selection
    num_lags = st.slider("Select Number of Lags", min_value=1, max_value=50, value=20)

    # Select the appropriate data based on the frequency choice
    selected_data = monthly_aqi if frequency == "Monthly" else daily_aqi

    # Plot ACF and PACF using Plotly
    st.write(f"ACF and PACF for {frequency} AQI Data with {num_lags} Lags")

    # Matplotlib to Plotly Conversion for ACF and PACF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(selected_data.dropna(), ax=ax1, lags=num_lags, title="ACF")
    plot_pacf(selected_data.dropna(), ax=ax2, lags=num_lags, title="PACF")
    st.pyplot(fig)


with tab4:    
    st.subheader("Dynamic Moving Average and Exponential Smoothing")

    # Slider for moving average window size (daily)
    daily_window = st.slider("Select window size for daily moving average", min_value=1, max_value=30, value=7, step=1)
    data['AQI_MA'] = data['AQI'].rolling(window=daily_window).mean()

    # Plot the Moving Average for Daily AQI
    st.write(f"Moving Average for Daily AQI with {daily_window}-Day Window")
    fig_ma_daily = go.Figure()
    fig_ma_daily.add_trace(go.Scatter(x=data.index, y=data['AQI'], mode='lines', name='Original AQI', line=dict(color='blue')))
    fig_ma_daily.add_trace(go.Scatter(x=data.index, y=data['AQI_MA'], mode='lines', name=f'{daily_window}-Day Moving Average', line=dict(color='red')))
    fig_ma_daily.update_layout(title="AQI with Daily Moving Average", xaxis_title="Date", yaxis_title="AQI", template="plotly_white")
    st.plotly_chart(fig_ma_daily, use_container_width=True)

    # Exponential Smoothing - Dynamic seasonal period for daily AQI
    seasonal_period = st.slider("Select seasonal period for Exponential Smoothing", min_value=1, max_value=1000, value=365, step=1)
    exp_smooth = ExponentialSmoothing(data['AQI'], trend='add', seasonal=None, seasonal_periods=seasonal_period).fit()
    data['AQI_ES'] = exp_smooth.fittedvalues

    # Plot Exponential Smoothing for Daily AQI
    st.write(f"Exponential Smoothing for Daily AQI with Seasonal Period of {seasonal_period}")
    fig_es = go.Figure()
    fig_es.add_trace(go.Scatter(x=data.index, y=data['AQI'], mode='lines', name='Original AQI', line=dict(color='blue')))
    fig_es.add_trace(go.Scatter(x=data.index, y=data['AQI_ES'], mode='lines', name='Exponential Smoothing', line=dict(color='red')))
    fig_es.update_layout(title="AQI with Exponential Smoothing", xaxis_title="Date", yaxis_title="AQI", template="plotly_white")
    st.plotly_chart(fig_es, use_container_width=True)

    # Monthly AQI data for 4-month moving average
    monthly = monthly_aqi.reset_index()
    
    # Slider for moving average window size (monthly)
    monthly_window = st.slider("Select window size for monthly moving average", min_value=1, max_value=12, value=4, step=1)
    monthly['AQI_MA'] = monthly['AQI'].rolling(window=monthly_window).mean()

    # Plot the Moving Average for Monthly AQI
    st.write(f"Moving Average for Monthly AQI with {monthly_window}-Month Window")
    fig_ma_monthly = go.Figure()
    fig_ma_monthly.add_trace(go.Scatter(x=monthly['Date'], y=monthly['AQI'], mode='lines', name='Original AQI', line=dict(color='blue')))
    fig_ma_monthly.add_trace(go.Scatter(x=monthly['Date'], y=monthly['AQI_MA'], mode='lines', name=f'{monthly_window}-Month Moving Average', line=dict(color='red')))
    fig_ma_monthly.update_layout(title="AQI with Monthly Moving Average", xaxis_title="Date", yaxis_title="AQI", template="plotly_white")
    st.plotly_chart(fig_ma_monthly, use_container_width=True)

      
with tab5:
    
    # Function to scale data
    @st.cache_data
    def scale_data(data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[['AQI']])
        return scaled_data, scaler

    scaled_data, scaler = scale_data(data)

    # Split data into train and test sets
    @st.cache_data
    def split_data(scaled_data, train_size_ratio=0.8):
        train_size = int(len(scaled_data) * train_size_ratio)
        return scaled_data[:train_size], scaled_data[train_size:]

    train, test = split_data(scaled_data)

    # Function to calculate and display error metrics
    def display_error_metrics(actual, predicted, label="Model"):
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = math.sqrt(mse)
        st.write(f'**{label} - Root Mean Squared Error (RMSE):** {rmse:.2f}')
        st.write(f'**{label} - Mean Absolute Error (MAE):** {mae:.2f}')

    # Plotting function
    def plot_results(train_index, test_index, test, test_predictions, title="Model Prediction", test_color='orange', pred_color='red'):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['AQI'], mode='lines', name='Actual AQI', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=test_index, y=test.flatten(), mode='lines', name='Actual Test AQI', line=dict(color=test_color)))
        fig.add_trace(go.Scatter(x=test_index, y=test_predictions.flatten(), mode='lines', name='Test Predictions', line=dict(color=pred_color)))
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="AQI",
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Auto ARIMA for Daily Data
    def arima_forecast_daily(data, seasonal=True, d=0, stepwise=True):
        train_size = int(len(data) * 0.8)
        train, test = data['AQI'][:train_size], data['AQI'][train_size:]

        auto_arima_model = pm.auto_arima(train, seasonal=seasonal, d=d, stepwise=stepwise, suppress_warnings=True, error_action='ignore')
        forecast, conf_int = auto_arima_model.predict(n_periods=len(test), return_conf_int=True)

        test_df = pd.DataFrame({'AQI': test, 'Forecast': forecast, 'Lower Bound': conf_int[:, 0], 'Upper Bound': conf_int[:, 1]}, index=test.index)
        
        # Plot with confidence intervals
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Train', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=test_df.index, y=test_df['AQI'], mode='lines', name='Test', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Forecast'], mode='lines', name='Forecast', line=dict(color='red')))
        fig.add_trace(go.Scatter(
            x=test_df.index.tolist() + test_df.index[::-1].tolist(),
            y=np.concatenate([test_df['Upper Bound'], test_df['Lower Bound'][::-1]]),
            fill='toself',
            fillcolor='rgba(255,182,193,0.3)',
            line=dict(color='rgba(255,182,193,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Confidence Interval'
        ))
        fig.update_layout(
            title='Auto ARIMA Model Forecast with Confidence Intervals (Daily Data)',
            xaxis_title='Date',
            yaxis_title='AQI',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

        display_error_metrics(test_df['AQI'], test_df['Forecast'], "Auto ARIMA Model (Daily)")

    with st.expander("Auto ARIMA Model Forecast (Daily Data)"):
        seasonal = st.checkbox('Seasonal', value=True)
        d = st.number_input('Order of Differencing (d)', min_value=0, max_value=2, value=0)
        stepwise = st.checkbox('Stepwise', value=True)
        arima_forecast_daily(data, seasonal, d, stepwise)

    # Auto ARIMA for Monthly Data
    def arima_forecast_monthly(monthly_aqi):
        train_size = int(len(monthly_aqi) * 0.8)
        train, test = monthly_aqi.iloc[:train_size], monthly_aqi.iloc[train_size:]

        auto_arima_model = pm.auto_arima(
            train['AQI'], start_p=1, start_q=1, max_p=5, max_q=5, d=1, start_P=0, start_Q=0,
            max_P=3, max_Q=3, D=1, m=12, seasonal=True, stepwise=True, suppress_warnings=True, error_action='ignore'
        )
        forecast, conf_int = auto_arima_model.predict(n_periods=len(test), return_conf_int=True)

        test_df = pd.DataFrame({
            'Date': test['Date'], 'AQI': test['AQI'].values, 'Forecast': forecast,
            'Lower Bound': conf_int[:, 0], 'Upper Bound': conf_int[:, 1]
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train['Date'], y=train['AQI'], mode='lines', name='Train', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=test_df['Date'], y=test_df['AQI'], mode='lines', name='Test', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=test_df['Date'], y=test_df['Forecast'], mode='lines', name='Forecast', line=dict(color='red')))
        fig.add_trace(go.Scatter(
            x=np.concatenate([test_df['Date'], test_df['Date'][::-1]]),
            y=np.concatenate([test_df['Upper Bound'], test_df['Lower Bound'][::-1]]),
            fill='toself', fillcolor='rgba(255,182,193,0.3)', line=dict(color='rgba(255,182,193,0)'),
            hoverinfo="skip", showlegend=True, name='Confidence Interval'
        ))
        fig.update_layout(
            title='Auto ARIMA Model Forecast with Confidence Intervals (Monthly Data)',
            xaxis_title='Date',
            yaxis_title='AQI',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

        display_error_metrics(test_df['AQI'], test_df['Forecast'], "Auto ARIMA Model (Monthly)")

    with st.expander("Auto ARIMA Model Forecast (Monthly Data)"):
        monthly_aqi = monthly_aqi.reset_index()
        arima_forecast_monthly(monthly_aqi)

    # Holt-Winters Model Forecast
    @st.cache_resource
    def fit_holt_winters(train_data, seasonal_periods):
        model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=seasonal_periods)
        return model.fit()

    def holt_winters_forecast(train, test, seasonal_periods):
        model = fit_holt_winters(train, seasonal_periods)
        test_predictions = model.forecast(len(test)).reshape(-1, 1)
        test = scaler.inverse_transform(test)
        test_predictions = scaler.inverse_transform(test_predictions)

        plot_results(data.index[len(train):], data.index[len(train):], test, test_predictions, title="Holt-Winters Exponential Smoothing Model Prediction")
        display_error_metrics(test, test_predictions, "Holt-Winters Model")

    with st.expander("Holt-Winters Model Forecast"):
        seasonal_periods = st.slider("Select Seasonal Periods", min_value=7, max_value=365, value=365)
        holt_winters_forecast(train, test, seasonal_periods)
        
    # Scale data
    @st.cache_data
    def scale_data(data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[['AQI']])
        return scaled_data, scaler

    scaled_data, scaler = scale_data(data)

    # Prepare sequences for LSTM
    @st.cache_data
    def create_sequences(data, look_back):
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:i + look_back, 0])
            y.append(data[i + look_back, 0])
        X = np.array(X)
        y = np.array(y)
        # Reshape data for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y

    look_back = 30  # Define look-back period
    X, y = create_sequences(scaled_data, look_back)

    # Split into train and test sets
    @st.cache_data
    def split_data(X, y, train_size_ratio=0.8):
        train_size = int(len(X) * train_size_ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = split_data(X, y)

    # Build and compile LSTM model
    @st.cache_resource
    def build_lstm_model(input_shape):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    model = build_lstm_model((X_train.shape[1], 1))

    # Train the model
    @st.cache_resource
    def train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
        return model, history

    with st.spinner("Training LSTM model..."):
        model, history = train_model(model, X_train, y_train, X_test, y_test)

    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse scale the predictions and actual values
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
    y_train = scaler.inverse_transform([y_train])[0]
    y_test = scaler.inverse_transform([y_test])[0]

    # Prepare data for plotting
    train_plot = np.empty_like(scaled_data)
    train_plot[:, :] = np.nan
    train_plot[look_back:len(train_predict) + look_back, :] = train_predict

    test_plot = np.empty_like(scaled_data)
    test_plot[:, :] = np.nan
    start_index_test = len(scaled_data) - len(test_predict)
    test_plot[start_index_test:, :] = test_predict

    # Plotting with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=scaler.inverse_transform(scaled_data).flatten(), mode='lines', name='Actual AQI', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=train_plot.flatten(), mode='lines', name='Train Predictions', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=data.index, y=test_plot.flatten(), mode='lines', name='Test Predictions', line=dict(color='red')))
    fig.update_layout(
        title="LSTM Model Prediction",
        xaxis_title="Date",
        yaxis_title="AQI",
        template="plotly_white"
    )

    # Display the plot
    with st.expander("LSTM Model Forecast"):
        st.plotly_chart(fig, use_container_width=True)

    # Calculate and display error metrics
    mse = mean_squared_error(y_test, test_predict)
    mae = mean_absolute_error(y_test, test_predict)
    rmse = math.sqrt(mse)

    st.write(f'**LSTM Model - Root Mean Squared Error (RMSE):** {rmse:.2f}')
    st.write(f'**LSTM Model - Mean Absolute Error (MAE):** {mae:.2f}')


    
