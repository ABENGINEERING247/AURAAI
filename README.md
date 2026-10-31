# AURA Sports Intelligence

AURA is a Streamlit AI sports assistant for cricket and football. It includes:
- OpenAI-powered conversational assistant
- Cricket and football knowledge assistant
- Player data/ranking page with provider adapters
- Editable tabular report builder
- PDF, DOCX and CSV export
- Twitch community-mode concept and suggested commands

## 1. Run locally (Windows)

```powershell
cd AURA_CHATBOT
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## 2. Configure API keys

Create `.streamlit/secrets.toml` (do not commit this file):

```toml
OPENAI_API_KEY = "your-key"
OPENAI_MODEL = "gpt-4o-mini"
API_FOOTBALL_KEY = "your-api-sports-key"
CRICAPI_KEY = "your-cricapi-key"
TWITCH_CLIENT_ID = "optional"
TWITCH_CLIENT_SECRET = "optional"
```

The app also reads environment variables. The included `secrets.toml.example` is a template only.

## 3. Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and `README.md`.
2. Open Streamlit Community Cloud and deploy the repository with `app.py` as the entry point.
3. In the deployed app's **Settings → Secrets**, paste the TOML secrets.
4. Reboot the app after adding or changing secrets.

Never upload real API keys or `.streamlit/secrets.toml` to GitHub.

## 4. Live sports data and ranking caveats

- Football adapter uses API-Sports football `/players` statistics, sorted by goals and assists for the selected league and season. This is a computed leaderboard, not an official FIFA ranking.
- Cricket adapter currently reads CricAPI's player directory. A player directory is **not** an official ICC ranking. To show official ICC batting, bowling and all-rounder rankings, connect a licensed/provider endpoint that supplies those ranking categories and rank positions. Do not label sample data as live.
- Provider subscriptions, endpoint access, request quotas and season availability vary. Check the provider's current documentation and account plan before launch.
- For truly live scores, fixtures, standings and match events, add provider endpoints and cache responses with a short TTL. Respect provider terms and attribution requirements.

## 5. Twitch hackathon extension

The current app includes a streamer control panel concept and command design. For a production Twitch integration:
- Register a Twitch application and configure OAuth securely.
- Run a separate TwitchIO bot or Twitch EventSub/IRC service.
- Send allowed chat questions to the AURA backend and moderate/rate-limit replies.
- Add streamer controls for enable/disable, allowed commands, cooldowns and safety filters.
- Keep Twitch client secrets and tokens in deployment secrets, never in source control.

Suggested commands: `!aura`, `!cricketquiz`, `!footballquiz`, `!compare`, `!matchreport`, `!rules`.

## 6. Important product boundary

AURA can explain rules, tactics, formats, history and statistics. AI-generated analysis is not an official ruling or source of truth. Real-time scores, player availability, rankings and transfers must come from a connected and authorized data provider.
