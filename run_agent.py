import os
import json
import feedparser
import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# 1. Graceful Imports & Dependency Check
# -------------------------------------------------------------------
try:
    import joblib
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "❌ Missing package 'joblib'. Install it via 'pip install joblib'"
    )

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "❌ Missing package 'xgboost'. Install it via 'pip install xgboost'"
    )

try:
    from groq import Groq
    from pydantic import BaseModel, Field
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "❌ Missing package 'groq' or 'pydantic'. Install them via 'pip install groq pydantic'"
    )

# -------------------------------------------------------------------
# 2. Configuration & Premier League Tracking Targets
# -------------------------------------------------------------------
PREMIER_LEAGUE_CLUBS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich Town",
    "Leicester City", "Liverpool", "Manchester City", "Manchester United",
    "Newcastle United", "Nottingham Forest", "Southampton", "Tottenham",
    "West Ham", "Wolves"
]

COMPLETED_TRANSFERS = {
    ("Ousmane Diomande", "Nottingham Forest"): True,
    ("Bruno Guimaraes", "Arsenal"): True,
    ("Morgan Rogers", "Chelsea"): True,
    ("Pep Chavarria", "Chelsea"): True,
    ("Brennan Johnson", "Everton"): True,
    ("Dwight McNeil", "Crystal Palace"): True,
    ("Shea Charles", "Fulham"): True,
    ("Ronald Araujo", "Liverpool"): True,
    ("Geronimo Rulli", "Manchester City"): True
}

MODEL_FILE = "transfer_model.pkl"

# -------------------------------------------------------------------
# 3. Pydantic Schema for Feature Extraction
# -------------------------------------------------------------------
class TransferFeatures(BaseModel):
    player: str = Field(description="Name of the player linked, or 'None' if unavailable.")
    buying_club: str = Field(description="Premier League team rumored to sign the player.")
    source_tier: int = Field(description="1 for top-tier, 2 for mid-tier, 3 for low/tabloid.")
    urgency_level: int = Field(description="1 (interest), 2 (negotiations), 3 (advanced), 4 (medical/done).")
    mention_frequency: int = Field(description="Estimated number of outlets covering this rumor (1 to 5).")
    status: str = Field(description="Short status string e.g., Speculation, Advanced Talks.")
    justification: str = Field(description="One concise English sentence explaining the extracted tier and urgency.")

# -------------------------------------------------------------------
# 4. Data Scraper (Google RSS Engine)
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
# 5. XGBoost Inference Engine
# -------------------------------------------------------------------
def get_xgboost_model():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            pass

    X_train = np.array([
        [1, 4, 5], [1, 3, 4], [2, 3, 3], [1, 2, 3],
        [2, 2, 2], [3, 1, 1], [3, 2, 1], [1, 1, 2]
    ])
    y_train = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss"
    )
    clf.fit(X_train, y_train)
    return clf

# -------------------------------------------------------------------
# 6. Main Pipeline Execution
# -------------------------------------------------------------------
def main():
    # Retrieve and thoroughly sanitize secret key variable including GROQ_API_KEY3
    raw_key = (
        os.environ.get("GROQ_API_KEY3")
        or os.environ.get("GROQ_API_KEY2") 
        or os.environ.get("GROQ_API_KEY") 
        or os.environ.get("GROQ_API_KEY_2") 
        or ""
    )
    
    api_key = raw_key.strip().strip("'").strip('"')

    if not api_key:
        print("❌ CRITICAL ERROR: GROQ_API_KEY3 or valid API key is missing from environment variables.")
        return

    client = Groq(api_key=api_key)
    xgb_model = get_xgboost_model()

    print(f"🔍 Fetching news feeds for {len(PREMIER_LEAGUE_CLUBS)} Premier League clubs...")
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

        Extract numerical features matching this JSON schema:
        - player: Full player name or "None"
        - buying_club: Premier League club name
        - source_tier: 1 (Reliable), 2 (Regional/Mid), 3 (Tabloid/Clickbait)
        - urgency_level: 1 (Interest), 2 (Talks), 3 (Advanced), 4 (Imminent/Done)
        - mention_frequency: Market repetition score (1-5)
        - status: Short string (e.g., Speculation, Advanced Talks)
        - justification: One concise explanation sentence.
        """

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a sports structured data extraction engine. Output ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )

            res = json.loads(response.choices[0].message.content)
            player = res.get("player", "None")

            if not player or player.lower() in ["none", "nan", "unknown", "n/a"]:
                continue

            buying_club = res.get("buying_club", art['club'])
            
            s_tier = int(res.get("source_tier", 3))
            u_level = int(res.get("urgency_level", 1))
            m_freq = int(res.get("mention_frequency", 1))

            features = np.array([[s_tier, u_level, m_freq]])
            xgb_prob = xgb_model.predict_proba(features)[0][1]
            calibrated_score = int(xgb_prob * 100)

            is_done = COMPLETED_TRANSFERS.get((player, buying_club), False)

            dataset.append({
                "Player": player,
                "Buying Club": buying_club,
                "Source Media": art['source'],
                "XGBoost Score": f"{calibrated_score}%",
                "Status": "Done Deal" if is_done else res.get("status", "Rumor"),
                "Verified Completed": "✅ Yes" if is_done else "❌ Pending",
                "Justification": res.get("justification", ""),
                "Source": f"[Article Link]({art['link']})"
            })

        except Exception as e:
            continue

    if not dataset:
        print("⚠️ No valid transfer rumors were extracted from current feeds.")
        return

    df = pd.DataFrame(dataset)
    
    markdown_report = "# ⚽ Premier League Transfer Reliability Matrix (LLM + XGBoost)\n\n"
    markdown_report += "*Updated via GitHub Actions | Model: Llama 3.3 70B + XGBoost Classifier*\n\n"
    markdown_report += df.to_markdown(index=False)

    with open("TRANSFER_REPORT.md", "w", encoding="utf-8") as f:
        f.write(markdown_report)

    print("✅ TRANSFER_REPORT.md has been generated successfully!")

if __name__ == "__main__":
    main()
