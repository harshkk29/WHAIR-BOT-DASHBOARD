import pandas as pd
import numpy as np
import json
import os
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import httpx

# --- CONFIG ---
DOCS_FILE = "dashboard_docs.json"
HISTORY_FILE = "weather_history_db.csv"
INDEX_FILE = "whair_rag.index"
MODEL_NAME = 'all-MiniLM-L6-v2'

class WhairBrain:
    def __init__(self):
        self.encoder = SentenceTransformer(MODEL_NAME)
        self.docs = []
        self.index = None
        self.load_knowledge()

    def load_knowledge(self):
        """Load static documentation into vector store"""
        if os.path.exists(DOCS_FILE):
            with open(DOCS_FILE, 'r') as f:
                self.docs = json.load(f)
            
            # Create embeddings
            texts = [f"{d['topic']}: {d['description']}" for d in self.docs]
            embeddings = self.encoder.encode(texts)
            
            # Setup FAISS
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(embeddings).astype('float32'))

    def search_knowledge(self, query, k=2):
        """Retrieve most relevant context for a query"""
        if not self.index:
            return ""
        
        query_vector = self.encoder.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.docs):
                results.append(self.docs[idx]['description'])
        
        return "\n".join(results)

    def log_weather_to_db(self, city, weather_data, aq_data):
        """Method 1: RAG - Build a historical library of data"""
        new_entry = {
            'timestamp': pd.Timestamp.now(),
            'city': city,
            'temp': weather_data['temperature_2m'],
            'aqi': aq_data['us_aqi'] if aq_data else None,
            'source': aq_data['highest_pollutant'] if aq_data else None
        }
        
        df = pd.DataFrame([new_entry])
        if not os.path.exists(HISTORY_FILE):
            df.to_csv(HISTORY_FILE, index=False)
        else:
            df.to_csv(HISTORY_FILE, mode='a', header=False, index=False)

    def get_historical_trends(self, city):
        """Method 4: Tool for the agent to look up past data"""
        if not os.path.exists(HISTORY_FILE):
            return "No historical data recorded yet."
        
        df = pd.read_csv(HISTORY_FILE)
        city_df = df[df['city'].str.lower() == city.lower()].tail(5)
        if city_df.empty:
            return f"No history for {city}."
        
        return city_df.to_string()

# --- TOOL DEFINITIONS FOR FUNCTION CALLING ---
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
                "description": "Get deep technical explanation of dashboard charts like SARIMAX or CPF plots.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "component_name": {"type": "string", "description": "The name of the chart (e.g. SARIMAX, CPF, Bivariate)"}
                    },
                    "required": ["component_name"]
                }
            }
        }
    ]
