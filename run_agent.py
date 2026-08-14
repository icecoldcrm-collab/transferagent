import os
import json
import feedparser
import joblib
import pandas as pd
import numpy as np
from groq import Groq
from pydantic import BaseModel, Field

# -------------------------------------------------------------------
# 1. Configuration & Premier League Tracking Targets
# -------------------------------------------------------------------
PREMIER_LEAGUE_CLUBS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich Town",
    "Leicester City", "Liverpool", "Manchester City", "Manchester United",
    "Newcastle United", "Nottingham Forest", "Southampton", "Tottenham",
    "West Ham", "Wolves"
]

# Ground Truth Verification DB (Cross-referenced against official transfers)
COMPLETED_TRANSFERS = {
    ("Ousmane Diomande", "Nottingham Forest"): True,
    ("Riccardo Calafiori", "Arsenal"): True,
    ("Savinho", "Manchester City"): True
}

MODEL_FILE = "transfer_model.pkl"

# -------------------------------------------------------------------
# 2. Pydantic Schema for Feature Extraction
# -------------------------------------------------------------------
class TransferFeatures(BaseModel):
    player: str = Field(description="Name of the player linked, or 'None' if unavailable.")
    buying_club: str = Field(description="Premier League team rumored to sign the player.")
    source_tier: int = Field(description="1 for top-tier (Athletic, Sky, Romano), 2 for mid-tier, 3 for low/tabloid.")
    urgency_level: int = Field(description="1 (interest/monitoring), 2 (negotiations), 3 (advanced), 4 (medical/done).")
    mention_frequency: int = Field(description="Estimated number of separate outlets covering this specific rumor (1 to 5).")
    status: str = Field(description="Short status string e.g., Speculation, Advanced Talks, Medical Scheduled.")
    justification: str = Field(description="One concise English sentence explaining the extracted tier and urgency.")

# -------------------------------------------------------------------
# 3. Data Scraper (Google RSS)
# -------------------------------------------------------------------
def fetch_club_news(club_name: str, max_articles: int = 2) -> list:
    query = f"{club_name}+transfer+rumors".replace(" ", "+")
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    
    articles = []
    for entry in feed.entries[:max_articles]:
        articles.append({
            "title": entry.title,
            "source": entry.source.title if hasattr(entry, "source") else "Unknown Media",
            "link": entry.link,
            "club": club_name
        })
    return articles

# -------------------------------------------------------------------
# 4. XGBoost Model Engine (Loads or creates fallback classifier)
# -------------------------------------------------------------------
def get_xgboost_model():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception as e:
            print(f"⚠️ Could not load {MODEL_FILE}: {e}. Initializing inline fallback.")

    # Fallback inline model training if no pretrained weight file exists
    from xgboost import XGBClassifier
    
    # Synthetic training setup matching Llama feature dimensions
    X_dummy = np.array([
        [1, 4, 5], [1, 3, 4], [2, 2, 2], 
        [3, 1, 1], [3, 2, 1], [1, 1, 2]
    ])
    y_dummy = np.array([1, 1, 0, 0, 0, 0])
    
    clf = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1)
    clf.fit(X_dummy, y_dummy)
    return clf

# -------------------------------------------------------------------
# 5. Main Execution Pipeline
# -------------------------------------------------------------------
def main():
    api_key = (
        os.environ.get("GROQ_API_KEY2") 
        or os.environ.get("GROQ_API_KEY") 
        or os.environ.get("GROQ_API_KEY_2") 
        or ""
    ).strip()

    if not api_key:
        print("❌ CRITICAL ERROR: GROQ_API_KEY is missing from environment variables.")
        return

    client = Groq(api_key=api_key)
    xgb_model = get_xgboost_model()

    print(f"--- Fetching News for {len(PREMIER_LEAGUE_CLUBS)} Premier League Clubs ---")
    all_articles = []
    for club in PREMIER_LEAGUE_CLUBS:
        all_articles.extend(fetch_club_news(club, max_articles=2))

    dataset = []

    for art in all_articles:
        prompt = f"""
        Analyze this transfer headline:
        Source: {art['source']}
        Headline: {art['title']}
        Target Club Context: {art['club']}

        Extract numerical features according to the JSON schema:
        - source_tier: 1 (Reliable), 2 (Regional/Mid), 3 (Tabloid/Clickbait)
        - urgency_level: 1 (Interest), 2 (Talks), 3 (Advanced), 4 (Imminent/Done)
        - mention_frequency: Estimated market repetition (1-5)
        """

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a sports structured data extraction model. Output ONLY valid JSON matching the schema."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )

            res = json.loads(response.choices[0].message.content)
            player = res.get("player", "None")

            # Filter invalid or missing player extractions
            if not player or player.lower() in ["none", "nan", "unknown", "n/a"]:
                continue

            buying_club = res.get("buying_club", art['club'])
            
            # --- XGBoost Inference Execution ---
            s_tier = int(res.get("source_tier", 3))
            u_level = int(res.get("urgency_level", 1))
            m_freq = int(res.get("mention_frequency", 1))

            # Reshape features into 2D array for predict_proba
            features = np.array([[s_tier, u_level, m_freq]])
            
            # Extract probability of class 1 (Transfer Completed Probability)
            xgb_prob = xgb_model.predict_proba(features)[0][1]
            calibrated_score = int(xgb_prob * 100)

            # --- Ground Truth Cross-Check ---
            is_done = COMPLETED_TRANSFERS.get((player, buying_club), False)

            dataset.append({
                "Player": player,
                "Buying Club": buying_club,
                "Source Media": art['source'],
                "XGBoost Prediction": f"{calibrated_score}%",
                "Status": "Done Deal" if is_done else res.get("status", "Rumor"),
                "Verified Completed": "✅ Yes" if is_done else "❌ Pending",
                "Justification": res.get("justification", ""),
                "Source": f"[Article Link]({art['link']})"
            })

        except Exception as e:
            print(f"Skipping article due to error: {e}")
            continue

    if not dataset:
        print("No valid player transfer rumors extracted.")
        return

    # Build Pandas DataFrame & Export to Markdown Matrix
    df = pd.DataFrame(dataset)
    
    markdown_report = "# ⚽ Premier League Transfer Reliability Matrix (Hybrid LLM + XGBoost)\n\n"
    markdown_report += "*Updated daily via GitHub Actions | Feature Extraction: Llama 3.3 70B | Probabilities: XGBoost Engine*\n\n"
    markdown_report += df.to_markdown(index=False)

    with open("TRANSFER_REPORT.md", "w") as f:
        f.write(markdown_report)

    print("TRANSFER_REPORT.md generated successfully!")

if __name__ == "__main__":
    main()
