import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError


MODEL_DIR = "cnn_lstm_model"   
print("Loading model and scalers...")
model = load_model(f"{MODEL_DIR}/final_cnn_lstm_model.h5", compile=False)
model.compile(optimizer="adam", loss=MeanSquaredError(), metrics=["mae"])

x_scaler = joblib.load(f"{MODEL_DIR}/x_scaler.pkl")
y_scaler = joblib.load(f"{MODEL_DIR}/y_scaler.pkl")

with open(f"{MODEL_DIR}/meta.json", "r") as f:
    meta = json.load(f)

FEATURES = meta["features"]
WINDOW_SIZE = meta["window_size"]

custom_input = {
    "open": 9422.849081,
    "high": 9428.490628,
    "low": 9422.849081,
    "volume": 2450000,
    "market_cap": 4500000000,
    "url_shares": 500,
    "unique_url_shares": 480,
    "reddit_posts": 60,
    "reddit_posts_score": 700,
    "reddit_comments": 120,
    "reddit_comments_score": 500,
    "tweets": 900,
    "tweet_spam": 10,
    "tweet_followers": 12000,
    "tweet_quotes": 25,
    "tweet_retweets": 50,
    "tweet_replies": 40,
    "tweet_favorites": 400,
    "tweet_sentiment1": 0.1,
    "tweet_sentiment2": 0.15,
    "tweet_sentiment3": 0.4,
    "tweet_sentiment4": 0.25,
    "tweet_sentiment5": 0.1,
    "tweet_sentiment_impact1": 0.12,
    "tweet_sentiment_impact2": 0.18,
    "tweet_sentiment_impact3": 0.35,
    "tweet_sentiment_impact4": 0.25,
    "tweet_sentiment_impact5": 0.1,
    "social_score": 5000,
    "average_sentiment": 0.45,
    "news": 25,
    "price_score": 0.75,
    "social_impact_score": 65,
    "correlation_rank": 30,
    "galaxy_score": 65,
    "volatility": 0.2,
    "market_cap_rank": 50,
    "percent_change_24h_rank": 40,
    "volume_24h_rank": 45,
    "social_volume_24h_rank": 35,
    "social_score_24h_rank": 38,
    "medium": 0,
    "youtube": 1,
    "social_volume": 15000,
    "percent_change_24h": 2.3,
    "market_cap_global": 1200000000000
}

input_vector = np.array([[custom_input[f] for f in FEATURES]], dtype=float)
input_scaled = x_scaler.transform(input_vector)
input_window = np.repeat(input_scaled, WINDOW_SIZE, axis=0).reshape(1, WINDOW_SIZE, len(FEATURES))
pred_scaled = model.predict(input_window)
predicted_value = y_scaler.inverse_transform(pred_scaled)[0, 0]
print("\n Predicted 'close' value:")
print(f"Predicted Close Price: {predicted_value:.4f}")
