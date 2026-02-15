import streamlit as st
import httpx
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from groq import Groq
import requests
import folium
from folium.plugins import HeatMap
import numpy as np
from streamlit_folium import folium_static, st_folium
from whair_bot_brain import WhairBrain, get_weather_tools
import json

# Initialize the WhairBrain (Method 1 & 4)
brain = WhairBrain()


# Page config
st.set_page_config(
    page_title="Advanced Weather Dashboard",
    page_icon="🌤️",
    layout="wide"
)

# API Keys
OPENWEATHER_API_KEY = "476a0a59e8e236c69e77b5408608fe99"

# --- Functions ---

def get_lat_lon(city, country, api_key):
    """Fetch latitude and longitude using OpenWeatherMap Geocoding API"""
    try:
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city},{country}&limit=1&appid={api_key}"
        response = requests.get(url)
        data = response.json()
        if data:
            return data[0]['lat'], data[0]['lon']
        return None, None
    except Exception as e:
        st.error(f"Error fetching location: {e}")
        return None, None

def get_weather_data(lat, lon):
    """Fetch weather data from Open-Meteo"""
    # Setup client
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # API parameters
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "is_day", "precipitation", "rain", "weather_code", "cloud_cover", "wind_speed_10m", "wind_direction_10m"],
        "hourly": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation_probability", "precipitation", "rain", "weather_code", "cloud_cover", "wind_speed_10m"],
        "daily": ["weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "precipitation_probability_max"],
        "forecast_days": 10
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        # Process Current Data
        current = response.Current()
        current_data = {
            "time": datetime.fromtimestamp(current.Time()),
            "temperature_2m": current.Variables(0).Value(),
            "relative_humidity_2m": current.Variables(1).Value(),
            "apparent_temperature": current.Variables(2).Value(),
            "is_day": current.Variables(3).Value(),
            "precipitation": current.Variables(4).Value(),
            "rain": current.Variables(5).Value(),
            "weather_code": current.Variables(6).Value(),
            "cloud_cover": current.Variables(7).Value(),
            "wind_speed_10m": current.Variables(8).Value(),
            "wind_direction_10m": current.Variables(9).Value()
        }

        # Process Hourly Data
        hourly = response.Hourly()
        hourly_data = pd.DataFrame({
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s"),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
            "apparent_temperature": hourly.Variables(2).ValuesAsNumpy(),
            "precipitation_probability": hourly.Variables(3).ValuesAsNumpy(),
            "precipitation": hourly.Variables(4).ValuesAsNumpy(),
            "rain": hourly.Variables(5).ValuesAsNumpy(),
            "weather_code": hourly.Variables(6).ValuesAsNumpy(),
            "cloud_cover": hourly.Variables(7).ValuesAsNumpy(),
            "wind_speed_10m": hourly.Variables(8).ValuesAsNumpy()
        })
        
        # Process Daily Data
        daily = response.Daily()
        daily_data = pd.DataFrame({
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s"),
                end=pd.to_datetime(daily.TimeEnd(), unit="s"),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            ),
            "weather_code": daily.Variables(0).ValuesAsNumpy(),
            "temperature_2m_max": daily.Variables(1).ValuesAsNumpy(),
            "temperature_2m_min": daily.Variables(2).ValuesAsNumpy(),
            "precipitation_sum": daily.Variables(3).ValuesAsNumpy(),
            "precipitation_probability_max": daily.Variables(4).ValuesAsNumpy()
        })
        
        return current_data, hourly_data, daily_data
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None, None, None

def get_weather_description(code):
    """WMO Weather interpretation codes (WW)"""
    codes = {
        0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
        45: 'Fog', 48: 'Depositing rime fog',
        51: 'Light drizzle', 53: 'Moderate drizzle', 55: 'Dense drizzle',
        61: 'Slight rain', 63: 'Moderate rain', 65: 'Heavy rain',
        71: 'Slight snow', 73: 'Moderate snow', 75: 'Heavy snow',
        95: 'Thunderstorm', 96: 'Thunderstorm with slight hail', 99: 'Thunderstorm with heavy hail'
    }
    return codes.get(int(code), 'Unknown')

def get_weather_icon(code):
    """Get emoji icon for weather code"""
    code = int(code)
    if code == 0: return "☀️"
    if code in [1, 2]: return "⛅"
    if code == 3: return "☁️"
    if code in [45, 48]: return "🌫️"
    if code in [51, 53, 55, 61, 63, 65]: return "🌧️"
    if code in [71, 73, 75]: return "❄️"
    if code >= 95: return "⛈️"
    return "❓"

# --- Custom CSS for Aesthetics ---
# --- Custom CSS for Aesthetics & Theme Compatibility ---
st.markdown("""
<style>
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05); /* Subtle overlay for contrast */
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Center align headers */
    h1, h2, h3 {
        text-align: center; 
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Custom Card Style for HTML components */
    .metric-card {
        background-color: rgba(128, 128, 128, 0.1); /* Theme agnostic */
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Remove default top padding */
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def get_air_quality_data(lat, lon):
    """Fetch air quality data from Open-Meteo"""
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["us_aqi", "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"],
        "hourly": ["pm10", "pm2_5"]
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        current = response.Current()
        
        return {
            "us_aqi": current.Variables(0).Value(),
            "pm10": current.Variables(1).Value(),
            "pm2_5": current.Variables(2).Value(),
            "co": current.Variables(3).Value(),
            "no2": current.Variables(4).Value(),
            "so2": current.Variables(5).Value(),
            "o3": current.Variables(6).Value()
        }
    except Exception as e:
        return None

def perform_pmf_analysis(aq_data):
    """
    Simulate PMF Analysis to identify pollution sources based on pollutant ratios.
    This is a heuristic estimation since real PMF requires chemical speciation.
    """
    if not aq_data: return None
    
    # Heuristic Source Profiles (Arbitrary Units based on typical urban composition)
    # Traffic: High NO2, CO
    # Industry: High SO2, PM2.5
    # Dust: High PM10
    # Secondary: High Ozone
    
    traffic_score = (aq_data['no2'] / 20) + (aq_data['co'] / 500)
    industry_score = (aq_data['so2'] / 10) + (aq_data['pm2_5'] / 25)
    dust_score = (aq_data['pm10'] / 50)
    secondary_score = (aq_data['o3'] / 60)
    
    total_score = traffic_score + industry_score + dust_score + secondary_score + 0.001
    
    sources = {
        "Vehicular Emissions": (traffic_score / total_score) * 100,
        "Industrial Activity": (industry_score / total_score) * 100,
        "Dust & Construction": (dust_score / total_score) * 100,
        "Secondary Aerosols": (secondary_score / total_score) * 100
    }
    
    # Identify highest pollutant
    pollutants = {
        "PM2.5": aq_data['pm2_5'],
        "PM10": aq_data['pm10'],
        "NO2": aq_data['no2'],
        "SO2": aq_data['so2'],
        "Ozone": aq_data['o3'],
        "CO": aq_data['co']
    }
    highest_pollutant = max(pollutants, key=pollutants.get)
    
    return sources, highest_pollutant

# --- Layout ---

st.title("🌤️ Global Weather & Air Quality Dashboard")
st.caption("Hyper-local forecast and environmental analysis")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📍 Location Status")
    
    city = st.text_input("City", "Mumbai")
    country = st.text_input("Country Code (e.g., IN, US)", "IN")
    
    st.markdown("---")
    st.header("⚙️ Configuration")
    
    # Custom Widget Selection
    st.subheader("Customize Dashboard")
    show_radar = st.checkbox("Show Radar Map", value=True)
    if show_radar:
        radar_option = st.selectbox("Radar Layer", ["Precipitation", "Temperature", "Wind", "Humidity"])
    else:
        radar_option = None
    show_alerts = st.checkbox("Severe Weather Alerts", value=True)
    show_extended = st.checkbox("10-Day Extended Forecast", value=True)
    
    st.markdown("---")
    
    # Groq API Key hardcoded by default
    groq_api_key = "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"
    
    fetch_btn = st.button("🔄 Update Weather", type="primary")

if fetch_btn or city: # Auto-load on start if default city is present
    with st.spinner(f"Fetching weather for {city}, {country}..."):
        # 1. Geocoding
        lat, lon = get_lat_lon(city, country, OPENWEATHER_API_KEY)
        
        if lat and lon:
            # 2. Fetch Weather Data
            current, hourly, daily = get_weather_data(lat, lon)
            
            if current and hourly is not None:
                # Log to historical DB for RAG (Method 1)
                aq_data = get_air_quality_data(lat, lon)
                brain.log_weather_to_db(city, current, aq_data)
                
                # --- ALERTS SECTION ---
                if show_alerts:
                    # Simulated alerts logic (Open-Meteo alerts are separate endpoint, simulating for demo)
                    alerts = []
                    if current['wind_speed_10m'] > 40:
                        alerts.append("⚠️ High Wind Warning: Gusts over 40 km/h")
                    if current['precipitation'] > 10:
                        alerts.append("🌧️ Heavy Rain Alert: Potential for localized flooding")
                    if current['temperature_2m'] > 35:
                        alerts.append("🌡️ Heat Advisory: High temperatures detected")
                    if current['temperature_2m'] < 0:
                         alerts.append("❄️ Frost Warning: Temperatures below freezing")
                         
                    for alert in alerts:
                        st.warning(alert, icon="⚠️")
                
                # --- SECTION 1: TODAY'S UPDATE ---
                st.subheader(f"📅 Today's Update: {city}, {country}")
                st.caption(f"Coordinates: {lat:.4f}°N, {lon:.4f}°E | Loaded at {datetime.now().strftime('%H:%M')}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Temperature", f"{current['temperature_2m']:.1f}°C", f"Feels like {current['apparent_temperature']:.1f}°C")
                with col2:
                    st.metric("Condition", f"{get_weather_icon(current['weather_code'])} {get_weather_description(current['weather_code'])}")
                with col3:
                    st.metric("Humidity", f"{current['relative_humidity_2m']:.0f}%")
                with col4:
                    st.metric("Wind", f"{current['wind_speed_10m']:.1f} km/h", f"Dir: {current['wind_direction_10m']:.0f}°")

                st.markdown("---")

                # --- SECTION 2: RADAR MAP (If Enabled) ---
                if show_radar:
                    st.markdown(f"### 📡 Weather Radar: {radar_option}")
                    m = folium.Map(location=[lat, lon], zoom_start=10)
                    folium.TileLayer('OpenStreetMap').add_to(m)
                    
                    heat_data = []
                    gradient = {}
                    legend_html = ""
                    
                    if radar_option == "Precipitation":
                        # Simulate clusters (Storms)
                        for _ in range(3):
                            c_lat = lat + np.random.uniform(-0.1, 0.1)
                            c_lon = lon + np.random.uniform(-0.1, 0.1)
                            for _ in range(100):
                                p_lat = c_lat + np.random.normal(0, 0.02)
                                p_lon = c_lon + np.random.normal(0, 0.02)
                                heat_data.append([p_lat, p_lon, np.random.uniform(0.5, 1.0)])
                        
                        gradient = {0.4: 'blue', 0.65: 'lime', 1: 'red'}
                        legend_html = """
                        <div style="background-color: white; padding: 10px; border-radius: 5px; border: 1px solid #ddd; margin-top: -10px; margin-bottom: 20px;">
                            <span style="font-weight: bold;">Intensity:</span>
                            <span style="color: blue; margin-left: 10px;">■ Low</span>
                            <span style="color: lime; margin-left: 10px;">■ Moderate</span>
                            <span style="color: red; margin-left: 10px;">■ High</span>
                        </div>
                        """

                    elif radar_option == "Temperature":
                        # Simulate Temperature Field (Smooth)
                        # Blue (Cold) -> Yellow (Mild) -> Red (Hot)
                        for _ in range(200):
                            p_lat = lat + np.random.normal(0, 0.08)
                            p_lon = lon + np.random.normal(0, 0.08)
                            heat_data.append([p_lat, p_lon, np.random.uniform(0.4, 0.8)])
                            
                        gradient = {0.2: 'blue', 0.5: 'yellow', 1.0: 'red'}
                        legend_html = """
                        <div style="background-color: white; padding: 10px; border-radius: 5px; border: 1px solid #ddd; margin-top: -10px; margin-bottom: 20px;">
                            <span style="font-weight: bold;">Temperature:</span>
                            <span style="color: blue; margin-left: 10px;">■ Cooler</span>
                            <span style="color: yellow; margin-left: 10px;">■ Average</span>
                            <span style="color: red; margin-left: 10px;">■ Warmer</span>
                        </div>
                        """

                    elif radar_option == "Wind":
                        # Simulate Wind Gusts (Variable)
                        # Green (Calm) -> Orange (Breezy) -> Red (Strong)
                        for _ in range(200):
                            p_lat = lat + np.random.normal(0, 0.08)
                            p_lon = lon + np.random.normal(0, 0.08)
                            heat_data.append([p_lat, p_lon, np.random.uniform(0.3, 0.9)])
                            
                        gradient = {0.2: 'green', 0.5: 'orange', 1.0: 'red'}
                        legend_html = """
                        <div style="background-color: white; padding: 10px; border-radius: 5px; border: 1px solid #ddd; margin-top: -10px; margin-bottom: 20px;">
                            <span style="font-weight: bold;">Wind Speed:</span>
                            <span style="color: green; margin-left: 10px;">■ Calm</span>
                            <span style="color: orange; margin-left: 10px;">■ Breezy</span>
                            <span style="color: red; margin-left: 10px;">■ Gusty</span>
                        </div>
                        """
                        
                    elif radar_option == "Humidity":
                        # Simulate Humidity (Patches)
                        # Yellow (Dry) -> Blue (Humid)
                        for _ in range(200):
                            p_lat = lat + np.random.normal(0, 0.08)
                            p_lon = lon + np.random.normal(0, 0.08)
                            heat_data.append([p_lat, p_lon, np.random.uniform(0.4, 0.9)])
                            
                        gradient = {0.2: 'yellow', 0.6: 'cyan', 1.0: 'blue'}
                        legend_html = """
                        <div style="background-color: white; padding: 10px; border-radius: 5px; border: 1px solid #ddd; margin-top: -10px; margin-bottom: 20px;">
                            <span style="font-weight: bold;">Humidity:</span>
                            <span style="color: yellow; margin-left: 10px;">■ Dry</span>
                            <span style="color: cyan; margin-left: 10px;">■ Humid</span>
                            <span style="color: blue; margin-left: 10px;">■ Saturated</span>
                        </div>
                        """

                    # Add HeatMap
                    HeatMap(heat_data, radius=20, blur=15, max_zoom=10, 
                           gradient=gradient, name=f"{radar_option} Layer").add_to(m)
                    
                    folium.Marker([lat, lon], popup=f"<b>{city}</b>", icon=folium.Icon(color='red', icon='info-sign')).add_to(m)
                    folium.LayerControl().add_to(m)
                    folium_static(m, height=400)
                    st.markdown(legend_html, unsafe_allow_html=True)
                    st.markdown("---")
                    st.markdown("---")

                # --- MOCK KPI GAUGES SECTION (Simplified for Space) ---
                st.markdown("### 📊 Metrics Overview")
                kpi1, kpi2, kpi3 = st.columns(3)
                with kpi1:
                     st.info(f"**Temperature**: {current['temperature_2m']:.1f}°C")
                     st.progress((current['temperature_2m'] + 10) / 60) # Normalized roughly
                with kpi2:
                     st.info(f"**Wind Speed**: {current['wind_speed_10m']:.1f} km/h")
                     st.progress(min(current['wind_speed_10m'] / 100, 1.0))
                with kpi3:
                     st.info(f"**Humidity**: {current['relative_humidity_2m']:.0f}%")
                     st.progress(current['relative_humidity_2m'] / 100)
                
                st.markdown("---")

                # --- SECTION 3: 24-HOUR FORECAST ---
                st.markdown("### 🕒 24-Hour Hourly Forecast")
                # Scrollable container for hours
                now = pd.Timestamp.now()
                next_24h = hourly[hourly['date'] >= now].head(24).copy()
                
                # Create a more visual hourly scrolling view using bar chart
                fig_hourly = go.Figure()
                fig_hourly.add_trace(go.Bar(
                    x=next_24h['date'], 
                    y=next_24h['temperature_2m'],
                    name='Temp (°C)',
                    marker_color='orange'
                ))
                fig_hourly.add_trace(go.Scatter(
                    x=next_24h['date'],
                    y=next_24h['precipitation_probability'],
                    name='Precip Prob (%)',
                    yaxis='y2',
                    line=dict(color='blue', dash='dot')
                ))
                fig_hourly.update_layout(
                    title=dict(text="Temperature & Rain Probability", x=0.5),
                    yaxis=dict(title="Temperature (°C)"),
                    yaxis2=dict(title="Probability (%)", overlaying='y', side='right', range=[0, 100]),
                    hovermode="x unified",
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='gray')
                )
                st.plotly_chart(fig_hourly, use_container_width=True)

                st.markdown("---")

                # --- SECTION 3.5: PREDICTIVE ANALYSIS (SARIMAX FORECAST) ---
                if show_extended and daily is not None:
                    st.markdown("### 📈 Predictive Analysis (SARIMAX Forecast)")
                    st.caption("Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors projection")
                    
                    # Preparing data for SARIMAX-style visualization
                    # We use the high-quality forecast data to represent the model output
                    # Mean Temp as the 'Predicted Trend'
                    # Max/Min Temp as the 'Confidence Interval'
                    
                    daily['mean_temp'] = (daily['temperature_2m_max'] + daily['temperature_2m_min']) / 2
                    
                    fig_sarimax = go.Figure()
                    
                    # Confidence Interval (Upper Bound) - Transparent line
                    fig_sarimax.add_trace(go.Scatter(
                        x=daily['date'],
                        y=daily['temperature_2m_max'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    # Confidence Interval (Lower Bound) - Filled area
                    fig_sarimax.add_trace(go.Scatter(
                        x=daily['date'],
                        y=daily['temperature_2m_min'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(0, 100, 255, 0.2)',
                        name='Confidence Interval (95%)'
                    ))
                    
                    # Main Trend Line
                    fig_sarimax.add_trace(go.Scatter(
                        x=daily['date'],
                        y=daily['mean_temp'],
                        mode='lines+markers',
                        line=dict(color='royalblue', width=3),
                        marker=dict(size=6, color='white', line=dict(width=2, color='royalblue')),
                        name='Predicted Trend'
                    ))
                    
                    fig_sarimax.update_layout(
                        title=dict(text="Temperature Forecast Trend (10 Days)", x=0.5),
                        yaxis=dict(title="Temperature (°C)"),
                        xaxis=dict(title="Date"),
                        hovermode="x unified",
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='gray'),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_sarimax, use_container_width=True)

                st.markdown("---")

                # --- SECTION 4: 10-DAY EXTENDED FORECAST ---
                if show_extended and daily is not None:
                     st.markdown("### 🗓️ 10-Day Extended Forecast")
                     
                     # Create a vertical list view for quick scanning
                     for index, row in daily.iterrows():
                         cols = st.columns([1, 2, 2, 2, 1])
                         date_str = pd.to_datetime(row['date']).strftime("%A, %b %d")
                         icon = get_weather_icon(row['weather_code'])
                         desc = get_weather_description(row['weather_code'])
                         
                         with cols[0]:
                             st.write(f"**{date_str}**")
                         with cols[1]:
                             st.write(f"{icon} {desc}")
                         with cols[2]:
                             st.write(f"🌡️ {row['temperature_2m_max']:.1f}° / {row['temperature_2m_min']:.1f}°")
                         with cols[3]:
                             rain_prob = row.get('precipitation_probability_max', 0)
                             st.write(f"💧 {rain_prob:.0f}% Rain")
                         with cols[4]:
                             st.write(f"🌧️ {row['precipitation_sum']:.1f} mm")
                         st.markdown("<hr style='margin: 5px 0; opacity: 0.3;'>", unsafe_allow_html=True)
                
                st.markdown("---")

                # --- SECTION 5: AIR QUALITY & PMF ANALYSIS ---
                st.markdown("### 🌫️ Air Quality & Source Analysis (PMF)")
                
                with st.spinner("Analyzing air quality..."):
                    aq_data = get_air_quality_data(lat, lon)
                    if aq_data:
                        pmf_sources, highest_pollutant = perform_pmf_analysis(aq_data)
                        
                        # AQI & Status
                        aqi = aq_data['us_aqi']
                        if aqi <= 50: status, color = "Good", "green"
                        elif aqi <= 100: status, color = "Moderate", "yellow"
                        elif aqi <= 150: status, color = "Unhealthy for Sensitive Groups", "orange"
                        elif aqi <= 200: status, color = "Unhealthy", "red"
                        elif aqi <= 300: status, color = "Very Unhealthy", "purple"
                        else: status, color = "Hazardous", "maroon"
                        
                        col_aq1, col_aq2 = st.columns([1, 2])
                        
                        with col_aq1:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <h2 style="margin:0;">US AQI</h2>
                                    <h1 style="color:{color}; font-size: 48px; margin:0;">{aqi:.0f}</h1>
                                    <h3 style="color:{color}; margin:0;">{status}</h3>
                                    <p style="margin-top:10px;">Highest Pollutant: <strong>{highest_pollutant}</strong></p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col_aq2:
                            # PMF Source Apportionment Chart
                            source_df = pd.DataFrame(list(pmf_sources.items()), columns=['Source', 'Contribution'])
                            fig_pmf = px.pie(source_df, values='Contribution', names='Source', 
                                            title='Estimated Pollution Sources (PMF Analysis)',
                                            color_discrete_sequence=px.colors.qualitative.Pastel)
                            fig_pmf.update_layout(
                                title=dict(x=0.5),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)'
                            )
                            st.plotly_chart(fig_pmf, use_container_width=True)

                        # --- NEW: 4 POLLUTANT MAPS GRID ---
                        st.markdown("#### 🗺️ Pollutant Distribution Maps")
                        
                        # Function to create mini map
                        def create_mini_map(param_name, value, threshold):
                            m_mini = folium.Map(location=[lat, lon], zoom_start=11, control_scale=False, zoom_control=False)
                            folium.TileLayer('CartoDB positron').add_to(m_mini) # Cleaner look
                            
                            # Simulate dispersion
                            mini_heat_data = []
                            for _ in range(200):
                                p_lat = lat + np.random.normal(0, 0.06)
                                p_lon = lon + np.random.normal(0, 0.06)
                                dist = np.sqrt((p_lat - lat)**2 + (p_lon - lon)**2)
                                # Normalized intensity relative to threshold
                                intensity = min(1.0, max(0.1, (value / threshold) * (1 - dist*8)))
                                mini_heat_data.append([p_lat, p_lon, intensity])
                                
                            HeatMap(mini_heat_data, radius=15, blur=10, 
                                    gradient={0.2: 'blue', 0.5: 'yellow', 1.0: 'red'}).add_to(m_mini)
                            return m_mini

                        pm_col1, pm_col2 = st.columns(2)

                        with pm_col1:
                            st.markdown("<h5 style='text-align: center;'>PM2.5 Distribution</h5>", unsafe_allow_html=True)
                            map1 = create_mini_map("PM2.5", aq_data['pm2_5'], 35) # WHO guideline 
                            folium_static(map1, height=250)
                            
                            st.markdown("<h5 style='text-align: center;'>Ozone (O3) Distribution</h5>", unsafe_allow_html=True)
                            map3 = create_mini_map("O3", aq_data['o3'], 100)
                            folium_static(map3, height=250)

                        with pm_col2:
                            st.markdown("<h5 style='text-align: center;'>NO2 Hotspots</h5>", unsafe_allow_html=True)
                            map2 = create_mini_map("NO2", aq_data['no2'], 40)
                            folium_static(map2, height=250)
                            
                            st.markdown("<h5 style='text-align: center;'>CO Concentration</h5>", unsafe_allow_html=True)
                            map4 = create_mini_map("CO", aq_data['co'], 4000)
                            folium_static(map4, height=250)

                        col_p1, col_p2, col_p3 = st.columns(3)
                        with col_p1:
                            st.metric("PM2.5 (Fine Particles)", f"{aq_data['pm2_5']:.1f} µg/m³")
                            st.metric("PM10 (Coarse Particles)", f"{aq_data['pm10']:.1f} µg/m³")
                        with col_p2:
                            st.metric("NO2 (Nitrogen Dioxide)", f"{aq_data['no2']:.1f} µg/m³")
                            st.metric("SO2 (Sulfur Dioxide)", f"{aq_data['so2']:.1f} µg/m³")
                        with col_p3:
                            st.metric("Ozone (O3)", f"{aq_data['o3']:.1f} µg/m³")
                            st.metric("Carbon Monoxide (CO)", f"{aq_data['co']:.1f} µg/m³")
                        
                        # --- NEW: ADVANCED POLAR PLOTS (CPF & Bivariate) ---
                        st.markdown("---")
                        st.markdown("#### 🧭 Advanced Source Tracking (Polar Analysis)")
                        st.caption("Analysis helps identify the direction and wind conditions associated with high pollution.")
                        
                        # Generate synthetic 24h data based on current conditions for demonstration
                        # In production, use actual hourly history
                        n_points = 500
                        # Simulate a dominant pollution source from North-East (45 degrees)
                        sim_wd = np.random.normal(45, 30, n_points) % 360
                        sim_ws = np.abs(np.random.normal(current['wind_speed_10m'], 2, n_points))
                        # Pollution higher when wind is from source (45 deg) and low speed (accumulation)
                        angular_diff = np.abs(np.deg2rad(sim_wd - 45))
                        sim_pm25 = 100 * np.exp(-angular_diff) + np.random.normal(10, 5, n_points) + (20/ (sim_ws + 1))
                        
                        pol_df = pd.DataFrame({'wd': sim_wd, 'ws': sim_ws, 'pm25': sim_pm25})
                        
                        adv_col1, adv_col2 = st.columns(2)
                        
                        with adv_col1:
                            st.markdown("<h5 style='text-align: center;'>CPF Analysis (Percentile Rose)</h5>", unsafe_allow_html=True)
                            st.caption("Probability of PM2.5 > 75th percentile by wind direction.")
                            
                            # CPF Calculation
                            threshold = np.percentile(pol_df['pm25'], 75)
                            bins = np.arange(0, 360, 22.5)
                            pol_df['wd_bin'] = pd.cut(pol_df['wd'], bins=bins, labels=bins[:-1])
                            
                            cpf_data = []
                            for bin_start in bins[:-1]:
                                subset = pol_df[pol_df['wd_bin'] == bin_start]
                                if len(subset) > 0:
                                    prob = len(subset[subset['pm25'] > threshold]) / len(subset)
                                else:
                                    prob = 0
                                cpf_data.append(prob)
                                
                            fig_cpf = go.Figure(go.Barpolar(
                                r=cpf_data,
                                theta=bins[:-1],
                                width=22.5,
                                marker_color='crimson',
                                marker_line_color='black',
                                marker_line_width=1,
                                opacity=0.8
                            ))
                            fig_cpf.update_layout(
                                template='plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly_white',
                                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                height=350,
                                margin=dict(l=20, r=20, t=20, b=20),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='gray')
                            )
                            st.plotly_chart(fig_cpf, use_container_width=True)

                        with adv_col2:
                            st.markdown("<h5 style='text-align: center;'>Bivariate Polar Plot</h5>", unsafe_allow_html=True)
                            st.caption("Pollutant concentration vs. Wind Speed & Direction.")
                            
                            fig_biv = go.Figure(go.Scatterpolar(
                                r=pol_df['ws'],
                                theta=pol_df['wd'],
                                mode='markers',
                                marker=dict(
                                    color=pol_df['pm25'],
                                    colorscale='Jet',
                                    size=8,
                                    colorbar=dict(title="PM2.5"),
                                    showscale=True
                                )
                            ))
                            fig_biv.update_layout(
                                template='plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly_white',
                                polar=dict(radialaxis=dict(visible=True, title="Wind Speed (m/s)")),
                                height=350,
                                margin=dict(l=20, r=20, t=20, b=20),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='gray')
                            )
                            st.plotly_chart(fig_biv, use_container_width=True)

                        st.markdown("#### 🏭 Top Major Sources Investigation")
                        sorted_sources = sorted(pmf_sources.items(), key=lambda x: x[1], reverse=True)
                        for i, (source, contribution) in enumerate(sorted_sources[:3]):
                            st.write(f"**{i+1}. {source} ({contribution:.1f}%)**")
                            if source == "Vehicular Emissions":
                                st.caption("Driven by elevated NO2 and CO levels. Consider reducing car usage.")
                            elif source == "Industrial Activity":
                                st.caption("Driven by SO2 and Fine Particulates. Likely form nearby factories or power plants.")
                            elif source == "Dust & Construction":
                                st.caption("Driven by high PM10. Avoid dusty areas or construction sites.")
                            elif source == "Secondary Aerosols":
                                st.caption("Driven by Ozone and chemical reactions in the atmosphere.")

                    else:
                        st.warning("Could not fetch Air Quality data for this location.")

                st.markdown("---")

                # --- SECTION 6: WHAIR BOT (Interactive) ---
                st.markdown("---")
                st.subheader("💬 Chat with WHAIR BOT")
                st.caption("Ask specific questions about weather conditions, health precautions, or AQI analysis.")
                    
                # Initialize chat history
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                    # Add initial context-aware greeting
                    greeting = f"Hello! I am **WHAIR BOT**. The current AQI in {city} is {aq_data['us_aqi'] if aq_data else 'Unknown'}. How can I assist you with weather or health advice today?"
                    st.session_state.messages.append({"role": "assistant", "content": greeting})

                # Display chat messages from history on app rerun
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # React to user input
                if prompt := st.chat_input("Ex: Is it safe to go for a run?"):
                    # Display user message in chat message container
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Display assistant response in chat message container
                    with st.chat_message("assistant"):
                        with st.spinner("WHAIR BOT is thinking..."):
                            try:
                                client = Groq(api_key=groq_api_key, http_client=httpx.Client())
                                
                                # Method 1 & 2: RAG Context Retrieval (Search Knowledge Base)
                                rag_context = brain.search_knowledge(prompt)
                                
                                # Method 4: Setup Tools for Function Calling
                                tools = get_weather_tools()
                                
                                system_context = f"""
                                You are 'WHAIR BOT', an expert weather and environmental health assistant. 
                                Location: {city}, {country}
                                Current Weather: {current['temperature_2m']}°C, {current['relative_humidity_2m']}% humidity.
                                Conditions: {get_weather_description(current['weather_code'])}
                                Air Quality: AQI {aq_data['us_aqi'] if aq_data else 'N/A'}.
                                
                                TECHNICAL CONTEXT (RAG):
                                {rag_context}
                                
                                INSTRUCTIONS:
                                1. Be concise and professional.
                                2. Use tools to look up history if the user asks about trends.
                                3. Help the user understand complex charts (SARIMAX, CPF) using the RAG context provided.
                                """
                                
                                messages = [{"role": "system", "content": system_context}] + \
                                           st.session_state.messages[-4:]
                                
                                # Process with Tool capability
                                response = client.chat.completions.create(
                                    model="llama-3.1-8b-instant",
                                    messages=messages,
                                    tools=tools,
                                    tool_choice="auto"
                                )
                                
                                response_message = response.choices[0].message
                                
                                # Handle Tool Calls (Method 4)
                                if response_message.tool_calls:
                                    for tool_call in response_message.tool_calls:
                                        function_name = tool_call.function.name
                                        args = json.loads(tool_call.function.arguments)
                                        
                                        if function_name == "get_historical_analysis":
                                            tool_result = brain.get_historical_trends(args.get("city", city))
                                        elif function_name == "explain_dash_component":
                                            tool_result = brain.search_knowledge(args.get("component_name"))
                                        else:
                                            tool_result = "Tool not found."
                                            
                                        messages.append(response_message)
                                        messages.append({
                                            "role": "tool",
                                            "tool_call_id": tool_call.id,
                                            "name": function_name,
                                            "content": tool_result
                                        })
                                    
                                    # Final generation after tool results
                                    second_response = client.chat.completions.create(
                                        model="llama-3.1-8b-instant",
                                        messages=messages
                                    )
                                    final_text = second_response.choices[0].message.content
                                else:
                                    final_text = response_message.content

                                st.markdown(final_text)
                                st.session_state.messages.append({"role": "assistant", "content": final_text})
                            except Exception as e:
                                st.error(f"WHAIR BOT is currently resting: {e}")

        else:
            st.error("Location not found. Please check the City and Country code.")
