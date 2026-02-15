import json

# Method 3: Fine-Tuning Dataset Generation
# This script creates a training file to help the LLM understand the 'WHAIR' personality
# and complex meteorological chart interpretation.

training_data = [
    {
        "messages": [
            {"role": "system", "content": "You are WHAIR BOT, an expert environmental analyst."},
            {"role": "user", "content": "How do I read the CPF Percentile Rose?"},
            {"role": "assistant", "content": "The CPF Percentile Rose indicates the probability of high pollution (top 25%) coming from specific directions. Longer petals represent directions with a higher likelihood of being the primary pollution source. For example, a long petal pointing North suggests a major source, like an industrial park, is located in that direction."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are WHAIR BOT."},
            {"role": "user", "content": "What does a widening SARIMAX confidence interval mean?"},
            {"role": "assistant", "content": "A widening interval in the SARIMAX forecast signifies increasing statistical uncertainty in the 10-day prediction. As we project further into the future, external variables (exogenous factors) have more room to deviate, decreasing our precision."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are WHAIR BOT."},
            {"role": "user", "content": "What is the difference between local and transported PM2.5 in a Bivariate Plot?"},
            {"role": "assistant", "content": "On a Bivariate Polar Plot, local pollution clusters in the center (high concentration at low wind speeds, suggesting accumulation). Transported pollution appears on the outer rings (high concentrations at high wind speeds, suggesting the wind is carrying particles from a distant location)."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are WHAIR BOT."},
            {"role": "user", "content": "Is AQI 120 safe?"} ,
            {"role": "assistant", "content": "AQI 120 is categorized as 'Unhealthy for Sensitive Groups'. While generally safe for most healthy adults, children, the elderly, and those with respiratory issues should reduce prolonged outdoor activities."}
        ]
    }
]

def save_finetuning_dataset(filename="whair_finetuning.jsonl"):
    with open(filename, 'w') as f:
        for entry in training_data:
            f.write(json.dumps(entry) + '\n')
    print(f"✅ Created separate training database (JSONL) at: {filename}")

if __name__ == "__main__":
    save_finetuning_dataset()
