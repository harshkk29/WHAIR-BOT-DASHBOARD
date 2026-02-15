import pandas as pd
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from groq import Groq
import httpx

# --- CONFIG ---
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
            with open(DOCS_FILE, 'r') as f:
                self.docs = json.load(f)
            
            if self.docs:
                texts = [f"{d['topic']}: {d['description']}" for d in self.docs]
                # Pre-calculate embeddings for the whole knowledge base
                self.doc_embeddings = self.encoder.encode(texts)
                # Normalize for cosine similarity
                norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
                self.doc_embeddings = self.doc_embeddings / (norms + 1e-9)

    def search_knowledge(self, query, k=2):
        """Retrieve most relevant context using pure Numpy (Cosine Similarity)"""
        if self.doc_embeddings is None or not self.docs:
            return ""
        
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

    def log_weather_to_db(self, city, weather_data, aq_data):
        """Method 1: RAG - Build a historical library of data"""
        if not aq_data:
            aq_data = {}

        new_entry = {
            'timestamp': pd.Timestamp.now(),
            'city': city,
            'temp': weather_data.get('temperature_2m'),
            'aqi': aq_data.get('us_aqi'),
            'source': aq_data.get('highest_pollutant')
        }
        
        # If source is missing, find the highest value among common pollutants
        if aq_data and not new_entry['source']:
            pollutants = {k: aq_data[k] for k in ['pm2_5', 'pm10', 'no2', 'so2', 'o3', 'co'] if k in aq_data}
            if pollutants:
                new_entry['source'] = max(pollutants, key=pollutants.get).upper()
        
        df = pd.DataFrame([new_entry])
        if not os.path.exists(HISTORY_FILE):
            df.to_csv(HISTORY_FILE, index=False)
        else:
            df.to_csv(HISTORY_FILE, mode='a', header=False, index=False)

    def get_historical_trends(self, city):
        """Method 4: Tool for the agent to look up past data"""
        if not os.path.exists(HISTORY_FILE):
            return "No historical data recorded yet."
        
        try:
            df = pd.read_csv(HISTORY_FILE)
            city_df = df[df['city'].str.lower() == city.lower()].tail(5)
            if city_df.empty:
                return f"No history for {city}."
            return city_df.to_string()
        except Exception:
            return "Error reading history database."

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
        }
    ]
