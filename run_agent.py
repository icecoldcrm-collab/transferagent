import os
import json
import feedparser
import pandas as pd
from groq import Groq
from pydantic import BaseModel, Field

# Target club to track (Change this to any team)
TARGET_CLUB = "Nottingham Forest"

# 1. Define Output Schema
class TransferAnalysis(BaseModel):
    player: str = Field(description="Name of the player linked")
    linked_club: str = Field(description="Club the player is rumored to join")
    current_club: str = Field(description="Player's current club if mentioned, else Unknown")
    likelihood_score: int = Field(description="Probability 0 to 100 based on source reliability and language")
    status: str = Field(description="e.g. Speculation, Advanced Talks, Medical Booked, Done Deal")
    justification: str = Field(description="1-2 sentences explaining why the score was given")

# 2. Fetch Google News RSS
def fetch_transfer_news(club_name: str, max_articles: int = 5):
    encoded_query = f"{club_name}+transfer+rumors".replace(" ", "+")
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    
    feed = feedparser.parse(rss_url)
    articles = []
    
    for entry in feed.entries[:max_articles]:
        articles.append({
            "title": entry.title,
            "source": entry.source.title if hasattr(entry, "source") else "Unknown",
            "link": entry.link
        })
    return articles

# 3. Main Runner Function
def main():
    api_key = os.environ.get("GROQ_API_KEY_2")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing!")

    client = Groq(api_key=api_key)
    articles = fetch_transfer_news(TARGET_CLUB, max_articles=5)

    if not articles:
        print("No news articles found.")
        return

    data_rows = []

    for art in articles:
        prompt = f"""
        Analyze the following football news headline for potential transfers involving {TARGET_CLUB}.
        
        Source: {art['source']}
        Headline: {art['title']}
        
        Evaluate source reliability and wording urgency. Output a JSON object with player, linked_club, current_club, likelihood_score (0-100), status, and justification.
        """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an elite football transfer credibility analyst. Respond ONLY with valid JSON matching the schema."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        res = json.loads(response.choices[0].message.content)
        data_rows.append({
            "Player": res.get("player", "N/A"),
            "Source Media": art['source'],
            "Likelihood": f"{res.get('likelihood_score', 0)}%",
            "Status": res.get("status", "N/A"),
            "Justification": res.get("justification", "N/A")
        })

    df = pd.DataFrame(data_rows)

    # Export markdown table for GitHub display
    markdown_report = f"# ⚽ Daily Transfer Identification Matrix for {TARGET_CLUB}\n\n"
    markdown_report += f"*Last Updated: Automated via GitHub Actions*\n\n"
    markdown_report += df.to_markdown(index=False)

    with open("TRANSFER_REPORT.md", "w") as f:
        f.write(markdown_report)

    print("TRANSFER_REPORT.md generated successfully!")

if __name__ == "__main__":
    main()
