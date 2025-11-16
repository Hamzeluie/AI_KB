#!/bin/bash

# =========================
# Configuration
# =========================
OWNER_ID="+12345952496"
KB_ID="kb+12345952496_en"
API_KEY="91a64066-74a1-4687-ae2c-122362469cc9"                # Must match config.yaml: API_KEY

# Orchestrator (Dashboard) Details
DASHBOARD_HOST="${DASHBOARD_HOST:-0.0.0.0}" 
DASHBOARD_PORT="${DASHBOARD_PORT:-5003}"   


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
    "101": { "Product name: Vintage Leather-bound Journal": "Price: 25.50$" },
    "102": { "Product name: Hand-poured Soy Candle (Lavender Scent)": "Price: 18.00$" },
    "103": { "Product name: Artisan Coffee Beans (Single Origin)": "Price: 15.75$" },
    "104": { "Product name: Wireless Ergonomic Mouse": "Price: 45.99$" },
    "105": { "Product name: Set of 4 Ceramic Mugs": "Price: 32.50$" },
    "106": { "Product name: Noise-Cancelling Headphones": "Price: 199.99$" },
    "107": { "Product name: Reusable Stainless Steel Water Bottle": "Price: 22.00$" },
    "108": { "Product name: Board Game: 'Catan'": "Price: 40.00$" },
    "109": { "Product name: Succulent Plant in a Geometric Pot": "Price: 14.25$" },
    "110": { "Product name: Digital E-reader": "Price: 129.00$" },
    "111": { "Product name: Wool Blend Scarf": "Price: 35.50$" },
    "112": { "Product name: Portable Bluetooth Speaker": "Price: 65.00$" },
    "113": { "Product name: Set of Gourmet Spices": "Price: 28.75$" },
    "114": { "Product name: Yoga Mat with Carrying Strap": "Price: 29.99$" },
    "115": { "Product name: Framed Abstract Art Print": "Price: 75.00$" },
    "116": { "Product name: Smartwatch": "Price: 249.99$" },
    "117": { "Product name: Box of Assorted Chocolates": "Price: 19.50$" },
    "118": { "Product name: High-speed USB-C Hub": "Price: 55.00$" },
    "119": { "Product name: Gardening Tool Set": "Price: 42.00$" },
    "120": { "Product name: Instant Film Camera": "Price: 89.99$" },
    "121": { "Product name: Memory Foam Pillow": "Price: 50.00$" },
    "122": { "Product name: Subscription Box (3-month)": "Price: 90.00$" },
    "123": { "Product name: Cookbook by a Famous Chef": "Price: 26.50$" },
    "124": { "Product name: Insulated Lunch Bag": "Price: 16.99$" },
    "125": { "Product name: Weighted Blanket": "Price: 79.99$" },
    "126": { "Product name: External Hard Drive (1TB)": "Price: 69.00$" },
    "127": { "Product name: Stainless Steel French Press": "Price: 38.00$" },
    "128": { "Product name: Set of 5 Resistance Bands": "Price: 20.00$" },
    "129": { "Product name: Leather Wallet": "Price: 45.00$" },
    "130": { "Product name: Electric Kettle": "Price: 34.50$" },
    "131": { "Product name: Smart Light Bulbs (Pack of 2)": "Price: 29.99$" },
    "132": { "Product name: Cold Brew Coffee Maker": "Price: 31.00$" },
    "133": { "Product name: Hiking Backpack": "Price: 75.00$" },
    "134": { "Product name: Pair of Blue Light Blocking Glasses": "Price: 23.00$" },
    "135": { "Product name: Vinyl Record Player": "Price: 150.00$" },
    "136": { "Product name: Echo Dot Smart Speaker": "Price: 49.99$" },
    "137": { "Product name: Set of 3 Food Storage Containers": "Price: 18.50$" },
    "138": { "Product name: Wireless Charging Pad": "Price: 25.00$" },
    "139": { "Product name: Movie Theater Gift Card": "Price: 50.00$" },
    "140": { "Product name: Handheld Milk Frother": "Price: 12.99$" },
    "141": { "Product name: Digital Meat Thermometer": "Price: 19.00$" },
    "142": { "Product name: Subscription to a Streaming Service (1 year)": "Price: 120.00$" },
    "143": { "Product name: Portable Power Bank": "Price: 39.99$" },
    "144": { "Product name: Silicone Baking Mat Set": "Price: 17.50$" },
    "145": { "Product name: Fitness Tracker": "Price: 99.00$" },
    "146": { "Product name: Stainless Steel Insulated Tumbler": "Price: 28.00$" },
    "147": { "Product name: Beginner's Ukulele Kit": "Price: 60.00$" },
    "148": { "Product name: Set of 6 Microfiber Cleaning Cloths": "Price: 10.50$" },
    "149": { "Product name: Electric Toothbrush": "Price: 49.00$" },
    "150": { "Product name: Coffee Grinder": "Price: 45.00$" },
    "151": { "Product name: Heated Neck and Shoulder Massager": "Price: 65.00$" },
    "152": { "Product name: Recipe Card Box": "Price: 15.00$" },
    "153": { "Product name: Scented Bath Bombs (Pack of 6)": "Price: 24.00$" },
    "154": { "Product name: Portable Desk Fan": "Price: 21.00$" },
    "155": { "Product name: Travel-sized toiletry set": "Price: 18.00$" },
    "156": { "Product name: Reusable Shopping Bags (Set of 3)": "Price: 12.00$" },
    "157": { "Product name: Desk Organizer": "Price: 27.50$" },
    "158": { "Product name: Set of 4 Coasters": "Price: 14.00$" },
    "159": { "Product name: Wireless Keyboard": "Price: 75.00$" },
    "160": { "Product name: Indoor Herb Garden Kit": "Price: 30.00$" },
    "161": { "Product name: Stainless Steel Utensil Set": "Price: 22.50$" },
    "162": { "Product name: Rechargeable Headlamp": "Price: 25.99$" },
    "163": { "Product name: Digital Photo Frame": "Price: 89.00$" },
    "164": { "Product name: Set of 2 Cocktail Glasses": "Price: 19.50$" },
    "165": { "Product name: Personalized Phone Case": "Price: 35.00$" },
    "166": { "Product name: Electric Wine Opener": "Price: 29.99$" },
    "167": { "Product name: Portable Hammock": "Price: 55.00$" },
    "168": { "Product name: Car Detailing Kit": "Price: 45.00$" },
    "169": { "Product name: Set of 3 Art Pens": "Price: 16.00$" },
    "170": { "Product name: Foldable Laundry Hamper": "Price: 18.00$" },
    "171": { "Product name: Smart Plug (2-Pack)": "Price: 24.99$" },
    "172": { "Product name: Bluetooth Tracker": "Price: 29.00$" },
    "173": { "Product name: Insulated Food Jar": "Price: 19.99$" },
    "174": { "Product name: Desktop Water Fountain": "Price: 40.00$" },
    "175": { "Product name: Aromatherapy Essential Oil Diffuser": "Price: 32.00$" },
    "176": { "Product name: LED Desk Lamp": "Price: 48.00$" },
    "177": { "Product name: Set of 3 Scented Soaps": "Price: 13.50$" },
    "178": { "Product name: Microscope Kit for Kids": "Price: 35.00$" },
    "179": { "Product name: Digital Kitchen Scale": "Price: 22.00$" },
    "180": { "Product name: Personal Blender": "Price: 39.00$" },
    "181": { "Product name: Travel Pillow": "Price: 21.50$" },
    "182": { "Product name: Subscription to a Magazine (1 year)": "Price: 60.00$" },
    "183": { "Product name: Portable Projector": "Price: 250.00$" },
    "184": { "Product name: Back Massager": "Price: 85.00$" },
    "185": { "Product name: Cocktail Shaker Set": "Price: 28.00$" },
    "186": { "Product name: Air Fryer": "Price: 99.00$" },
    "187": { "Product name: Electric Toothbrush Heads (4-pack)": "Price: 25.00$" },
    "188": { "Product name: Smart Water Bottle": "Price: 49.99$" },
    "189": { "Product name: Set of 6 Shot Glasses": "Price: 16.00$" },
    "190": { "Product name: Digital Alarm Clock with USB Port": "Price: 30.00$" },
    "191": { "Product name: Fitness Jump Rope": "Price: 14.50$" },
    "192": { "Product name: Stainless Steel Colander": "Price: 20.00$" },
    "193": { "Product name: Wireless Doorbell": "Price: 45.00$" },
    "194": { "Product name: Cast Iron Skillet": "Price: 55.00$" },
    "195": { "Product name: External Battery Pack for Laptop": "Price: 120.00$" },
    "196": { "Product name: Set of 4 Tea Towels": "Price: 17.00$" },
    "197": { "Product name: Portable Desk Vacuum": "Price: 25.00$" },
    "198": { "Product name: Ceramic Vases (Set of 3)": "Price: 38.00$" },
    "199": { "Product name: Digital Stylus Pen": "Price: 40.00$" },
    "200": { "Product name: Wireless Keyboard and Mouse Combo": "Price: 60.00$" }
  }
}
EOF