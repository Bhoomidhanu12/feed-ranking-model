from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load model & encoders
model = joblib.load("feed_ranking_model.pkl")
encoders = joblib.load("encoders.pkl")

# =========================
# DEFINE INPUT STRUCTURE
# =========================
class Post(BaseModel):
    post_type: str
    media_type: str
    post_age: float
    global_likes: float
    user_activity_score: float
    history_watched_count: float
    creator_affinity_score: float

# =========================
# FEATURES (IMPORTANT)
# =========================
features = [
    'post_type',
    'media_type',
    'post_age',
    'global_likes',
    'user_activity_score',
    'history_watched_count',
    'creator_affinity_score'
]

# =========================
# API
# =========================
@app.post("/rank-feed")
def rank_feed(posts: List[Post]):
    df = pd.DataFrame([p.dict() for p in posts])

    # 🔥 Encode categorical features
    for col in ['post_type', 'media_type']:
        df[col] = encoders[col].transform(df[col].astype(str))

    # 🔥 Predict engagement score
    df["score"] = model.predict_proba(df[features])[:, 1]

    # 🔥 Decode back to original values (IMPORTANT)
    for col in ['post_type', 'media_type']:
        df[col] = encoders[col].inverse_transform(df[col])

    # 🔥 Sort by ranking score
    df = df.sort_values(by="score", ascending=False)

    return df.to_dict(orient="records")    