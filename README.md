# Chad - Discord AI Bot

## Overview

Chad is a feature‑rich Discord bot powered by Grok AI. It provides intelligent slash‑command interactions, robust input validation, rate limiting, token budgeting, and a web‑based admin dashboard. Designed to be offensive.

## Features

- **Slash Commands** – Native `/ask` command for AI queries.
- **Smart Validation** – Filters spam, gibberish, duplicates, and trivial inputs.
- **Rate Limiting** – Configurable per‑user and per‑guild limits.
- **Daily Budgets** – Token‑based budgeting to control API costs.
- **Auto‑Approve Workflow** – Optional admin approval queue for all requests.
- **Customizable Personality** – Editable system prompt and bot responses.
- **Admin Dashboard** – Web UI for configuration, message templates, approval queue, analytics, and admin management.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ashwnn/chad.git
   cd chad
   ```
2. **Set up a virtual environment**
   ```bash
   python -m venv .venv
   # On Windows
   .venv\\Scripts\\activate
   # On Unix/macOS
   source .venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Create a `.env` file** with the required variables (see Configuration).

## Usage

Run the bot and the dashboard in separate terminals:

```bash
# Terminal 1 – Discord bot
python -m chad_bot.bot
```

```bash
# Terminal 2 – Web dashboard (FastAPI)
python -m chad_bot.web
```

The dashboard will be available at `http://localhost:8000`.

## Configuration

All settings are supplied via environment variables or the `config/config.yaml` file.

### Environment Variables (`.env` example)
```
DISCORD_BOT_TOKEN=your_discord_bot_token_here
GROK_API_KEY=your_grok_api_key_here
GROK_API_BASE=https://api.x.ai/v1
GROK_CHAT_MODEL=grok-beta
DATABASE_PATH=data/chad.sqlite3
WEB_HOST=0.0.0.0
WEB_PORT=8000
MAX_PROMPT_CHARS=4000
```

### YAML Configuration
The `config/config.yaml` file lets you customise bot messages, system prompt, rate‑limit messages, and other user‑facing texts via the dashboard or directly editing the file.

## Project Structure
```
chad/
├── src/
│   └── chad_bot/          # Main package
│       ├── bot.py         # Discord bot implementation
│       ├── web.py         # FastAPI dashboard
│       ├── service.py     # Core request processing
│       ├── database.py    # SQLite data layer
│       ├── grok_client.py # Grok API client
│       ├── yaml_config.py # YAML config manager
│       ├── spam.py        # Input validation logic
│       ├── rate_limits.py # Rate‑limiting utilities
│       ├── config.py      # Settings handling
│       └── discord_api.py # Discord interaction helpers
├── config/
│   └── config.yaml        # Default bot messages & settings
├── templates/             # HTML templates for the dashboard
├── static/                # CSS and static assets
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Compose file for bot + DB
├── docs/                  # Additional documentation
├── data/                  # Runtime SQLite database (created on start)
├── requirements.txt       # Python dependencies
└── README.md
```

## Examples

- **Ask the bot**: Use the `/ask` slash command in any Discord channel where the bot is present.
- **Approve a request**: Open the dashboard, navigate to the *Approval Queue*, and click *Approve* or *Reject*.
- **View analytics**: Visit `http://localhost:8000/analytics` to see usage statistics and token consumption.

## Roadmap

- Add support for additional LLM providers.
- Implement persistent queue storage.
- Expand admin permissions granularity.
- Provide Docker‑Swarm/Kubernetes deployment templates.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Make your changes and ensure existing functionality remains intact.
4. Open a pull request describing the changes.

## Testing

Run the test suite with:
```bash
pytest
```
(Ensure any required test dependencies are installed.)

## License

Licensed under the [CC BY‑NC‑SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/) license.

## Acknowledgments

Thanks to the developers of:
- **Grok AI** for the language model API.
- **FastAPI** for the web framework.
- **Discord.py** for the Discord integration library.
- The open‑source community for numerous utilities and inspiration.
