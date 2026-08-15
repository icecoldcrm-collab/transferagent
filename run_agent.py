import os
import json
import feedparser
import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# 1. Dependency Checks & Exception Handling
# -------------------------------------------------------------------
try:
    import joblib
except ModuleNotFoundError:
    raise ModuleNotFoundError("Missing package 'joblib'. Install via 'pip install joblib'")

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    raise ModuleNotFoundError("Missing package 'xgboost'. Install via 'pip install xgboost'")

try:
    from groq import Groq
except ModuleNotFoundError:
    raise ModuleNotFoundError("Missing package 'groq'. Install via 'pip install groq'")

# -------------------------------------------------------------------
# 2. Target Clubs Configuration & Ground Truth Database
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
    ("Savinho", "Manchester City"): True
}

MODEL_FILE = "transfer_model.pkl"

# -------------------------------------------------------------------
# 3. Multinational & Domestic RSS Scraper Engine
# -------------------------------------------------------------------
def fetch_multinational_news(club_name: str) -> list:
    """
    Scrapes feeds across multiple regional parameters (US, Spain, Italy, Germany) 
    to capture foreign media breaks and direct domestic reports.
    """
    regions = [
        {"hl": "en-US", "gl": "US", "ceid": "US:en", "suffix": "transfer rumors"},
        {"hl": "es", "gl": "ES", "ceid": "ES:es", "suffix": "fichajes rumores"},
        {"hl": "it", "gl": "IT", "ceid": "IT:it", "suffix": "calciomercato news"},
        {"hl": "de", "gl": "DE", "ceid": "DE:de", "suffix": "transfergerüchte"}
    ]
    
    articles = []
    seen_links = set()

    for reg in regions:
        query = f"{club_name}+{reg['suffix']}".replace(" ", "+")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl={reg['hl']}&gl={reg['gl']}&ceid={reg['ceid']}"
        feed = feedparser.parse(rss_url)
        
        # Take top 1 article per region to ensure diverse global coverage
        for entry in feed.entries[:1]:
            if entry.link not in seen_links:
                seen_links.add(entry.link)
                articles.append({
                    "title": entry.title,
                    "source": entry.source.title if hasattr(entry, "source") else f"Global Media ({reg['gl']})",
                    "link": entry.link,
                    "club": club_name
                })
    return articles

# -------------------------------------------------------------------
# 4. XGBoost Model Setup
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
    
    clf = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric="logloss")
    clf.fit(X_train, y_train)
    return clf

# -------------------------------------------------------------------
# 5. Main Execution Pipeline
# -------------------------------------------------------------------
def main():
    raw_key = (
        os.environ.get("GROQ_API_KEY3")
        or os.environ.get("GROQ_API_KEY2") 
        or os.environ.get("GROQ_API_KEY") 
        or ""
    )
    api_key = raw_key.strip().strip("'").strip('"')

    if not api_key:
        print("❌ CRITICAL ERROR: API Key missing from environment variables.")
        return

    client = Groq(api_key=api_key)
    xgb_model = get_xgboost_model()

    print(f"🌍 Fetching international & domestic feeds for all {len(PREMIER_LEAGUE_CLUBS)} clubs...")
    
    dataset = []

    for club in PREMIER_LEAGUE_CLUBS:
        articles = fetch_multinational_news(club)
        for art in articles:
            prompt = f"""
            Analyze this football transfer headline (which may originate from English, Spanish, Italian, or German media):
            Target Club Context: {art['club']}
            Source Media: {art['source']}
            Headline: {art['title']}

            Translate or interpret context accurately and return a valid JSON object with keys:
            - "player": Exact player name or "None"
            - "buying_club": Target Premier League club name
            - "source_tier": integer 1 (tier-1 global/local reliable source like Marca, Di Marzio, Sky), 2 (mid), 3 (tabloid)
            - "urgency_level": integer 1 (rumor) to 4 (done/medical)
            - "mention_frequency": integer 1 to 5
            - "status": short description string (e.g. "Advanced Talks", "Completed", "Rumor")
            - "justification": single sentence explanation written in English
            """

            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are a precise multilingual data extraction engine. Output ONLY valid JSON."},
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

                xgb_prob = xgb_model.predict_proba(np.array([[s_tier, u_level, m_freq]]))[0][1]
                likelihood_val = int(xgb_prob * 100)
                is_done = COMPLETED_TRANSFERS.get((player, buying_club), False)

                dataset.append({
                    "Club": buying_club,
                    "Player": player,
                    "Source": art['source'],
                    "_Likelihood_Raw": likelihood_val,  # Used for accurate numeric sorting
                    "Likelihood": f"{likelihood_val}%",
                    "Status": "Completed" if is_done else res.get("status", "Rumor"),
                    "Justification": res.get("justification", ""),
                    "Link": f"[Source]({art['link']})"
                })

            except Exception:
                continue

    if not dataset:
        print("⚠️ No valid transfer records compiled.")
        return

    df = pd.DataFrame(dataset)
    
    # Sort matrix entries from highest likelihood to lowest
    df = df.sort_values(by="_Likelihood_Raw", ascending=False).drop(columns=["_Likelihood_Raw"])
    
    report = "# ⚽ Comprehensive Premier League Transfer Intelligence Matrix\n\n"
    report += f"*Sorted by AI Likelihood Score — Multilingual engine covering international & domestic media for {len(PREMIER_LEAGUE_CLUBS)} clubs.*\n\n"
    report += df.to_markdown(index=False)

    with open("TRANSFER_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("✅ TRANSFER_REPORT.md updated, sorted by likelihood, and infused with global media sources!")

if __name__ == "__main__":
    main()
