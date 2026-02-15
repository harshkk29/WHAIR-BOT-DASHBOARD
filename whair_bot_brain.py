import pandas as pd
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from groq import Groq
import httpx

import logging
# Suppress transformers and sentence_transformers spam
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# --- CONFIG ---
VERSION = "1.0.2" # Force update with silent loading
DOCS_FILE = "dashboard_docs.json"
HISTORY_FILE = "weather_history_db.csv"
MODEL_NAME = 'all-MiniLM-L6-v2'

class WhairBrain:
    def __init__(self):
        self.encoder = SentenceTransformer(MODEL_NAME)
        self.docs = []
        self.doc_embeddings = None
        self.load_knowledge()

    def load_knowledge(self):
        """Load static documentation and create embeddings using Numpy (No FAISS to avoid crashes)"""
        if os.path.exists(DOCS_FILE):
            try:
                with open(DOCS_FILE, 'r') as f:
                    self.docs = json.load(f)
                
                if self.docs:
                    texts = [f"{d['topic']}: {d['description']}" for d in self.docs]
                    # Pre-calculate embeddings for the whole knowledge base
                    self.doc_embeddings = self.encoder.encode(texts)
                    # Normalize for cosine similarity
                    norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
                    self.doc_embeddings = self.doc_embeddings / (norms + 1e-9)
            except Exception as e:
                print(f"Knowledge Load Error: {e}")

    def search_knowledge(self, query, k=2):
        """Retrieve most relevant context using pure Numpy (Cosine Similarity)"""
        if self.doc_embeddings is None or not self.docs:
            return ""
        
        try:
            # Encode and normalize query
            query_vector = self.encoder.encode([query])
            query_norm = np.linalg.norm(query_vector)
            query_vector = query_vector / (query_norm + 1e-9)
            
            # Calculate cosine similarity (dot product on normalized vectors)
            similarities = np.dot(self.doc_embeddings, query_vector.T).flatten()
            
            # Get top K indices
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            results = []
            for idx in top_indices:
                if idx < len(self.docs):
                    results.append(self.docs[idx]['description'])
            
            return "\n".join(results)
        except Exception as e:
            return ""

    def log_weather_to_db(self, city, weather_data, aq_data):
        """Method 1: RAG - Build a historical library of data with rich features"""
        try:
            # PARANOID SAFETY CHECK
            # We never use direct indexing on input dicts anymore
            w_safe = weather_data if isinstance(weather_data, dict) else {}
            a_safe = aq_data if isinstance(aq_data, dict) else {}

            new_entry = {
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'city': str(city),
                'temp': w_safe.get('temperature_2m'),
                'humidity': w_safe.get('relative_humidity_2m'),
                'wind_speed': w_safe.get('wind_speed_10m'),
                'weather_code': w_safe.get('weather_code'),
                'aqi': a_safe.get('us_aqi'),
                'source': a_safe.get('highest_pollutant') or a_safe.get('source')
            }
            
            # Fallback for source if missing
            if not new_entry['source'] and a_safe:
                pollutants = {k: a_safe.get(k, 0) for k in ['pm2_5', 'pm10', 'no2', 'so2', 'o3', 'co']}
                if any(v > 0 for v in pollutants.values()):
                    valid_pollutants = {k: v for k, v in pollutants.items() if v is not None}
                    if valid_pollutants:
                        new_entry['source'] = max(valid_pollutants, key=valid_pollutants.get).upper()
            
            df = pd.DataFrame([new_entry])
            
            # Ensure file existence and column match
            if not os.path.exists(HISTORY_FILE):
                df.to_csv(HISTORY_FILE, index=False)
            else:
                df.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
        except Exception as e:
            # ABSOLUTE SILENCE: Do not let logging errors reach the main app
            pass

    def get_historical_trends(self, city):
        """Method 4: Tool for the agent to look up past data"""
        if not os.path.exists(HISTORY_FILE):
            return "No historical data recorded yet."
        
        try:
            df = pd.read_csv(HISTORY_FILE)
            if 'city' not in df.columns:
                return "History database format is outdated."
            
            city_df = df[df['city'].astype(str).str.lower() == str(city).lower()].tail(5)
            if city_df.empty:
                return f"No history for {city}."
            return city_df.to_string()
        except Exception as e:
            return f"Error reading history: {e}"

# --- TOOL DEFINITIONS ---
def get_weather_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_historical_analysis",
                "description": "Get last 5 recorded weather/AQI snapshots for a specific city to identify trends.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The name of the city"}
                    },
                    "required": ["city"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "explain_dash_component",
                "description": "Get deep technical explanation of dashboard charts (SARIMAX, CPF, Bivariate).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "component_name": {"type": "string", "description": "The name of the chart"}
                    },
                    "required": ["component_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_forecast_analysis",
                "description": "Analyze the upcoming 7-day and 24-hour forecast data to provide future-dated recommendations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scope": {"type": "string", "enum": ["hourly", "daily"], "description": "The timeframe to analyze"}
                    },
                    "required": ["scope"]
                }
            }
        }
    ]
