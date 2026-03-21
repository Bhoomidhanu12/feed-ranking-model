import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

model = joblib.load("feed_ranking_model.pkl")
encoders = joblib.load("encoders.pkl")

@app.get("/")
def home():
    return {"message": "Feed Ranking API Running 🚀"}

@app.post("/rank-feed")
def rank_feed(data: list):
    df = pd.DataFrame(data)

    # encode categorical
    for col in ['post_type', 'media_type']:
        df[col] = encoders[col].transform(df[col].astype(str))

    # prediction
    df['score'] = model.predict_proba(df[[
        'post_type',
        'media_type',
        'post_age',
        'global_likes',
        'user_activity_score',
        'history_watched_count',
        'creator_affinity_score'
    ]])[:, 1]

    df = df.sort_values(by="score", ascending=False)

    return df.to_dict(orient="records")