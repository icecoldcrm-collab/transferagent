import os
import json
import feedparser
import pandas as pd
from groq import Groq
from pydantic import BaseModel, Field

TARGET_CLUB = "Nottingham Forest"

# 1. Define Output Schema
class TransferAnalysis(BaseModel):
    player: str = Field(description="Name of the player linked. Return 'None' if no specific player is mentioned.")
    linked_club: str = Field(description="Club the player is rumored to join")
    likelihood_score: int = Field(description="Probability 0 to 100 based on source reliability and language")
    status: str = Field(description="Short status e.g. Speculation, Advanced Talks, Done Deal, No Transfer")
    justification: str = Field(description="A single clear English sentence explaining the score. DO NOT output JSON or dicts here.")

# 2. Fetch News via Google RSS
def fetch_transfer_news(club_name: str, max_articles: int = 8):
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

# 3. Main Script
def main():
    api_key = (
        os.environ.get("GROQ_API_KEY2") 
        or os.environ.get("GROQ_API_KEY") 
        or os.environ.get("GROQ_API_KEY_2") 
        or ""
    ).strip()

    if not api_key:
        print("❌ CRITICAL ERROR: GROQ_API_KEY is missing.")
        return

    client = Groq(api_key=api_key)
    articles = fetch_transfer_news(TARGET_CLUB, max_articles=8)

    if not articles:
        print("No articles found.")
        return

    data_rows = []

    for art in articles:
        prompt = f"""
        Analyze this football news headline for potential transfers involving {TARGET_CLUB}:
        
        Source: {art['source']}
        Headline: {art['title']}
        
        Extract information matching the JSON schema. Ensure 'justification' is strictly a plain English sentence, NOT a JSON or key-value object.
        """

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a football transfer analyst. Respond ONLY with valid JSON matching the specified schema."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )

            res = json.loads(response.choices[0].message.content)
            player_name = res.get("player", "None")

            # Filter out non-transfer news or unidentifiable players
            if not player_name or player_name.lower() in ["none", "nan", "unknown", "n/a"]:
                continue

            # Ensure justification is a string if the model returned a dict
            justification_val = res.get("justification", "")
            if isinstance(justification_val, dict):
                justification_val = ". ".join([f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in justification_val.items()])

            data_rows.append({
                "Player": player_name,
                "Source Media": art['source'],
                "Likelihood": f"{res.get('likelihood_score', 0)}%",
                "Status": res.get("status", "N/A"),
                "Justification": str(justification_val)
            })

        except Exception as e:
            print(f"Skipping article due to parsing error: {e}")
            continue

    if not data_rows:
        print("No valid player transfer rumors detected.")
        return

    df = pd.DataFrame(data_rows)

    markdown_report = f"# ⚽ Daily Transfer Identification Matrix for {TARGET_CLUB}\n\n"
    markdown_report += f"*Last Updated: Automated via GitHub Actions*\n\n"
    markdown_report += df.to_markdown(index=False)

    with open("TRANSFER_REPORT.md", "w") as f:
        f.write(markdown_report)

    print("TRANSFER_REPORT.md updated successfully!")

if __name__ == "__main__":
    main()
