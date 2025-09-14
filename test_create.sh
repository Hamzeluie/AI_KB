#!/bin/bash

# =========================
# Configuration
# =========================
OWNER_ID="123"
KB_ID="test_kb"
API_KEY="test_api_key"                # Must match config.yaml: API_KEY

# Orchestrator (Dashboard) Details
DASHBOARD_HOST="${DASHBOARD_HOST:-0.0.0.0}" 
DASHBOARD_PORT="${DASHBOARD_PORT:-8000}"   


URL="http://$DASHBOARD_HOST:$DASHBOARD_PORT/db/$OWNER_ID"

# =========================
# Validate Environment
# =========================
if [ -z "$OPENAI_API_BASE_URL_DASHBOARD" ]; then
  echo "⚠️  Warning: OPENAI_API_BASE_URL_DASHBOARD is not set!"
  echo "   Set it before starting the orchestrator:"
  echo "   export OPENAI_API_BASE_URL_DASHBOARD=$URL"
fi

# =========================
# Echo Info
# =========================
echo "🚀 Sending document creation request to orchestrator"
echo "📍 URL: $URL"
echo "🔑 API Key: $API_KEY"
echo "📦 KB_ID: $KB_ID | OWNER_ID: $OWNER_ID"
echo "🔗 Will forward to local_omni at: $OPENAI_API_BASE_URL_DASHBOARD"

# =========================
# Send Request
# =========================
curl -X POST "$URL" \
  -H "Content-Type: application/json" \
  -H "api-key: $API_KEY" \
  -w "\n\n⏱️  Response time: %{time_total}s | Status: %{http_code}\n" \
  -d @- <<EOF
{
  "kb_id": "$KB_ID",
  "owner_id": "$OWNER_ID",
  "document": {
    "1": {
      "soccer_player_name": "Lionel Messi",
      "team": "Inter Miami CF"
    },
    "2": {
      "soccer_player_name": "Cristiano Ronaldo",
      "team": "Al Nassr FC"
    },
    "3": {
      "soccer_player_name": "Kylian Mbappé",
      "team": "Real Madrid CF"
    }
  }
}
EOF