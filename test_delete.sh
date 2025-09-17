#!/bin/bash

# Configuration
OWNER_ID="123"
KB_ID="test_kb"
API_KEY="test_api_key"

DASHBOARD_HOST="${DASHBOARD_HOST:-localhost}" 
DASHBOARD_PORT="${DASHBOARD_PORT:-5003}"   

URL="http://$DASHBOARD_HOST:$DASHBOARD_PORT/db/$OWNER_ID/$KB_ID"

echo "🗑️  Deleting knowledge base:"
echo "📍 Owner ID: $OWNER_ID"
echo "📚 KB ID: $KB_ID"
echo "🔗 URL: $URL"
echo "🔐 API Key: $API_KEY"

# Send DELETE request
curl -X DELETE "$URL" \
  -H "api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -w "\n\n⏱️  Response time: %{time_total}s | Status: %{http_code}\n"