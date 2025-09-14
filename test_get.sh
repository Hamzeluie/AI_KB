#!/bin/bash

# Configuration
OWNER_ID="123"
API_KEY="test_api_key"
QUERY="السعر بلوتوث"
KB_ID="test_kb"
LIMIT=3

DASHBOARD_HOST="${DASHBOARD_HOST:-localhost}" 
DASHBOARD_PORT="${DASHBOARD_PORT:-8000}"   

URL="http://$DASHBOARD_HOST:$DASHBOARD_PORT/db/search/$OWNER_ID"

echo "🔍 Searching for: '$QUERY'"
echo "📍 URL: $URL"
echo "🔐 API Key: $API_KEY"
echo "📦 KB_ID: $KB_ID"

# Send search request
curl -X POST "$URL" \
  -H "Content-Type: application/json" \
  -H "api-key: $API_KEY" \
  -w "\n\n⏱️  Response time: %{time_total}s | Status: %{http_code}\n" \
  -d @- <<EOF
{
  "query_text": "$QUERY",
  "kb_id": ["$KB_ID"],
  "limit": $LIMIT
}
EOF