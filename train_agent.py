import os
import json
import time
import pandas as pd
from groq import Groq
from pydantic import BaseModel, Field

# 1. Expand to Multiple Clubs
CLUBS_TO_TRACK = [
    "Nottingham Forest", "Arsenal", "Chelsea", 
    "Liverpool", "Manchester United", "Real Madrid"
]

# Mock Ground-Truth Completed Transfers Database (In production, scrape Transfermarkt)
COMPLETED_TRANSFERS = {
    ("Ousmane Diomande", "Nottingham Forest"): 1,
    ("Curtis Jones", "Nottingham Forest"): 0,
    ("Martin Zubimendi", "Arsenal"): 0,
    ("Riccardo Calafiori", "Arsenal"): 1
}

class HistoricalRumorSchema(BaseModel):
    player: str = Field(description="Name of the player linked")
    buying_club: str = Field(description="Club linked with buying the player")
    source_tier: int = Field(description="1 for top-tier journos/outlets, 2 for mid-tier, 3 for clickbait")
    urgency_level: int = Field(description="1 (low interest) to 5 (imminent/medical booked)")
    justification: str = Field(description="Brief reason for the extracted values")

def extract_rumor_features(headline: str, source: str, client: Groq) -> dict:
    prompt = f"""
    Analyze this historical transfer headline:
    Source: {source}
    Headline: {headline}
    
    Extract structured features matching the JSON schema.
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a sports data extraction engine. Output strictly valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return None

def main():
    api_key = os.environ.get("GROQ_API_KEY2") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("❌ Missing API key.")
        return

    client = Groq(api_key=api_key)
    dataset = []

    print(f"--- Processing Historical Rumors across {len(CLUBS_TO_TRACK)} Clubs ---")

    # In production, replace this sample loop with historical API calls (e.g., NewsAPI / SerpAPI)
    sample_historical_news = [
        {"title": "Nottingham Forest complete signing of Ousmane Diomande for €40m", "source": "The Athletic", "club": "Nottingham Forest"},
        {"title": "Arsenal in advanced talks for Riccardo Calafiori as fee agreed", "source": "Fabrizio Romano", "club": "Arsenal"},
        {"title": "Nottingham Forest eyeing ambitious swoop for Curtis Jones", "source": "Daily Mail", "club": "Nottingham Forest"},
        {"title": "Arsenal monitor Zubimendi situation ahead of deadline day", "source": "Transfer Tavern", "club": "Arsenal"}
    ]

    for item in sample_historical_news:
        extracted = extract_rumor_features(item["title"], item["source"], client)
        if not extracted or extracted.get("player") == "None":
            continue

        player = extracted.get("player")
        buying_club = item["club"]

        # Label ground truth: Did this transfer actually happen?
        actual_outcome = COMPLETED_TRANSFERS.get((player, buying_club), 0)

        dataset.append({
            "Player": player,
            "Buying Club": buying_club,
            "Source": item["source"],
            "Source Tier": extracted.get("source_tier", 3),
            "Urgency Level": extracted.get("urgency_level", 1),
            "Landed Deal (Ground Truth)": actual_outcome
        })
        time.sleep(0.5)

    df = pd.DataFrame(dataset)
    df.to_csv("historical_transfer_dataset.csv", index=False)
    print("\nSuccessfully generated training dataset: 'historical_transfer_dataset.csv'")
    print(df.to_string())

if __name__ == "__main__":
    main()
