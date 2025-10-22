import requests
import json
import time

# The URL where your FastAPI server is running
SERVER_URL = "http://127.0.0.1:8101/batch_generate"

# 1. Define the 4 requests for the batch
# We will use two sessions to test history:
# - session_A: Asks two follow-up questions
# - session_B: Asks one question
# - session_C: Asks one question
# What is the price of Pair of Blue Light Blocking Glasses?
test_requests = [
    # {
    #     "session_id": "session_A_123",
    #     "owner_id": "+12345952496",
    #     "user_prompt": "What is the price of Electric Toothbrush?",
    #     "kb_id": "kb+12345952496_en"
    # },
    # {
    #     "session_id": "session_B_456",
    #     "owner_id": "+12345952496",
    #     "user_prompt": "What is the price of Fitness Tracker?",
    #     "kb_id": "kb+12345952496_en"
    # },
    # {
    #     "session_id": "session_A_123", # <--- SAME session_id as the first request
    #     "owner_id": "+12345952496",
    #     "user_prompt": "what did i ask from you?", # Follow-up question
    #     "kb_id": "kb+12345952496_en"
    # },
    # {
    #     "session_id": "session_C_789",
    #     "owner_id": "+12345952496",
    #     "user_prompt": "Can you tell me about the Set of 2 Cocktail Glasses?",
    #     "kb_id": "kb+12345952496_en"
    # },
    {
        "session_id": "session_C_789",
        "owner_id": "+12345952496",
        "user_prompt": "I am Borhan?",
        "kb_id": "kb+12345952496_en"
    },
    {
        "session_id": "session_B_456",
        "owner_id": "+12345952496",
        "user_prompt": "I am Jack",
        "kb_id": "kb+12345952496_en"
    },
    {
        "session_id": "session_C_789",
        "owner_id": "+12345952496",
        "user_prompt": "What is my name?",
        "kb_id": "kb+12345952496_en"
    },
    {
        "session_id": "session_B_456",
        "owner_id": "+12345952496",
        "user_prompt": "What is my name?",
        "kb_id": "kb+12345952496_en"
    },

    
]

# 2. Format the payload according to the BatchGenerateRequest model
payload = {
    "requests": test_requests
}

# 3. Send the request and print the response
print(f"Sending batch of {len(test_requests)} requests to {SERVER_URL}...")
print("Payload:")
print(json.dumps(payload, indent=2))
print("-" * 30)

try:
    start_time = time.time()
    
    response = requests.post(SERVER_URL, json=payload)
    
    end_time = time.time()
    print(f"Request completed in {end_time - start_time:.2f} seconds.")
    print("-" * 30)

    # 4. Check the response
    if response.status_code == 200:
        print("Success! (HTTP 200)")
        print("Server Response:")
        # Pretty-print the JSON response
        response_data = response.json()
        print(json.dumps(response_data, indent=2))
        
        # Check if we got 4 responses back
        if 'responses' in response_data and len(response_data['responses']) == len(test_requests):
            print(f"\nSuccessfully received {len(response_data['responses'])} responses.")
        else:
            print("\nError: Did not receive the expected number of responses.")
            
    else:
        print(f"Error! (HTTP {response.status_code})")
        print("Response Text:")
        print(response.text)

except requests.exceptions.ConnectionError:
    print(f"Connection Error: Failed to connect to {SERVER_URL}.")
    print("Please make sure the 'model.py' server is running.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")