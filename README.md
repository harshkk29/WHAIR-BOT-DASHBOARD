# 🌤️ Weather Smart Dashboard & WHAIR BOT

A premium, interactive weather and air quality monitoring system built with Streamlit, Plotly, and Folium. Featuring **WHAIR BOT**, an AI-powered atmospheric assistant.

## ✨ Key Features
- **Hyper-Local Forecast**: Real-time weather data from Open-Meteo API.
- **Interactive Radar Map**: Multi-layer GIS integration for Precipitation, Temperature, Wind, and Humidity.
- **Advanced Air Quality Analysis**:
    - **Pollutant Distribution Grid**: 4 separate heatmaps for PM2.5, NO2, O3, and CO.
    - **SARIMAX Predictive Trends**: 10-day predictive time series using seasonal ARIMA logic.
    - **Polar Analysis**: CPF (Percentile Rose) and Bivariate Polar Plots for source tracking.
- **💬 WHAIR BOT**: Context-aware AI chatbot that answers questions about weather conditions and health precautions using data-driven insights (Powered by Groq).
- **Theme Adaptive**: Premium UI compatible with both Dark and Light modes.

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/harshkk29/WEATHER-SMART-DASH-BOARD.git
   cd WEATHER-SMART-DASH-BOARD
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run weather_dashboard.py
   ```

## 🛠️ Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Visualizations**: [Plotly](https://plotly.com/), [Folium](https://python-visualization.github.io/folium/)
- **Data APIs**: [Open-Meteo](https://open-meteo.com/), [OpenWeatherMap](https://openweathermap.org/)
- **AI/LLM**: [Groq Cloud](https://groq.com/) (Llama 3.1)
- **Time Series**: [Statsmodels](https://www.statsmodels.org/)

## 📜 License
This project is open-source and available under the MIT License.
