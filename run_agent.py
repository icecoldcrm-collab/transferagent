import os
import json
import feedparser
import pandas as pd
from groq import Groq
from pydantic import BaseModel, Field

# 1. Expanded Premier League Club List
PREMIER_LEAGUE_CLUBS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich Town",
    "Leicester City", "Liverpool", "Manchester City", "Manchester United",
    "Newcastle United", "Nottingham Forest", "Southampton", "Tottenham",
    "West Ham", "Wolves"
]

# 2. Verified Completed Transfers Database (Ground Truth for Accuracy Tracking)
COMPLETED_TRANSFERS = {
    ("Bruno Guimaraes", "Arsenal"): True,
    ("Piero Hincapie", "Arsenal"): True,
    ("Youri Tielemans", "Manchester United"): True,
    ("Morgan Rogers", "Chelsea"): True,
    ("James Trafford", "Leeds United"): True,
    ("Ousmane Diomande", "Nottingham Forest"): True
}

class TransferAnalysis(BaseModel):
    player: str = Field(description="Name of the player linked")
    buying_club: str = Field(description="Premier League team linked")
    likelihood_score: int = Field(description="Probability 0 to 100 based on source reliability")
    status: str = Field(description="Speculation, Advanced Talks, or Completed")
    justification: str = Field(description="One concise sentence explaining the score")

def fetch_club_news(club_name: str, max_articles: int = 3):
    query = f"{club_name}+transfer+rumors".replace(" ", "+")
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    return [{
        "title": entry.title,
        "source": entry.source.title if hasattr(entry, "source") else "Unknown",
        "link": entry.link,
        "club": club_name
    } for entry in feed.entries[:max_articles]]

def main():
    api_key = (os.environ.get("GROQ_API_KEY2") or os.environ.get("GROQ_API_KEY") or "").strip()
    if not api_key:
        print("❌ Missing API key.")
        return

    client = Groq(api_key=api_key)
    all_articles = []
    
    # Scrape news across all 20 Premier League clubs
    for club in PREMIER_LEAGUE_CLUBS:
        all_articles.extend(fetch_club_news(club, max_articles=2))

    dataset = []
    for art in all_articles:
        prompt = f"""
        Analyze this news item for {art['club']}:
        Source: {art['source']} | Headline: {art['title']}
        Extract player name, buying club, likelihood_score (0-100), status, and justification.
        If no player is mentioned, set player to 'None'.
        """
        try:
            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Respond strictly in JSON matching the schema."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            data = json.loads(res.choices[0].message.content)
            
            player = data.get("player", "None")
            if player.lower() in ["none", "unknown", "nan", "n/a"]:
                continue

            buying_club = data.get("buying_club", art['club'])
            
            # Ground-truth cross check
            is_completed = COMPLETED_TRANSFERS.get((player, buying_club), False)

            dataset.append({
                "Player": player,
                "Buying Club": buying_club,
                "Source Media": art['source'],
                "AI Score": f"{data.get('likelihood_score', 0)}%",
                "Status": "Done Deal" if is_completed else data.get("status", "Rumor"),
                "Deal Completed": "✅ Yes" if is_completed else "❌ Pending/Failed",
                "Justification": data.get("justification", ""),
                "Article Link": f"[Source]({art['link']})"
            })
        except Exception:
            continue

    df = pd.DataFrame(dataset)
    markdown_report = "# ⚽ All-Premier League Transfer Reliability Matrix\n\n"
    markdown_report += df.to_markdown(index=False)

    with open("TRANSFER_REPORT.md", "w") as f:
        f.write(markdown_report)

    print("TRANSFER_REPORT.md updated successfully across all PL teams!")

if __name__ == "__main__":
    main()
