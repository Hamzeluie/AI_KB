import requests
import json
from typing import List, Dict, Optional

class QwenChatClient:
    def __init__(self, base_url: str = "http://localhost:8100"):
        self.base_url = base_url.rstrip("/")
        self.session_id = None
        self.message_history = []  # Local cache of conversation

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        new_session: bool = False
    ) -> Dict:
        """
        Send a message to the Qwen3-32B-AWQ chat API and return reply + updated history.

        Automatically manages session and message history locally.
        """
        if new_session:
            self.session_id = None
            self.message_history = []
            if system_prompt:
                self.message_history.append({"role": "system", "content": system_prompt})

        # If no session yet, initialize with system prompt (if any) + user prompt
        if self.session_id is None:
            if system_prompt and not self.message_history:
                self.message_history.append({"role": "system", "content": system_prompt})
            self.message_history.append({"role": "user", "content": prompt})
        else:
            # Append new user message to existing history
            self.message_history.append({"role": "user", "content": prompt})

        # Send full history to server
        payload = {
            "session_id": self.session_id,
            "messages": self.message_history,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=120
            )
            response.raise_for_status()
            result = response.json()

            # Update internal state
            self.session_id = result["session_id"]
            self.message_history = result["messages"]  # Sync with server history

            return result

        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")

    def send_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> Dict:
        """
        Send full message history (advanced usage).
        """
        payload = {
            "session_id": self.session_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            self.session_id = result["session_id"]
            self.message_history = result["messages"]
            return result

        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")

    def get_session_id(self) -> Optional[str]:
        return self.session_id

    def get_history(self) -> List[Dict]:
        return self.message_history.copy()

    def clear_session(self):
        if self.session_id:
            try:
                requests.delete(f"{self.base_url}/session/{self.session_id}")
            except:
                pass
        self.session_id = None
        self.message_history = []

    def reset_history(self):
        """Reset local history without deleting server session."""
        self.message_history = []


# ===== Simple Function Interface =====

def send_prompt(
    prompt: str,
    base_url: str = "http://localhost:8100",
    session_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95
) -> Dict:
    """
    Simple function to send a single prompt and get a reply.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "session_id": session_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }

    try:
        response = requests.post(
            f"{base_url}/chat",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to send prompt: {str(e)}")


# ===== Example Usage =====

if __name__ == "__main__":
    print("=== Simple Prompt ===")
    result = send_prompt(
        prompt="Hello, who are you?",
        system_prompt="You are a helpful AI that don't think and answers concisely and very shortly.",
    )
    print("Reply:", result["reply"])
    print("Session ID:", result["session_id"])

    # print("\n=== Follow-up ===")
    # followup = send_prompt(
    #     prompt="What's the speed of light?",
    #     session_id=result["session_id"]
    # )
    # print("Reply:", followup["reply"])

    # print("\n=== Client Class (Multi-turn) ===")
    # client = QwenChatClient()

    # resp1 = client.chat("Hello, introduce yourself briefly.")
    # print("AI:", resp1["reply"])

    # resp2 = client.chat("What can you help me with?")
    # print("AI:", resp2["reply"])

    # resp3 = client.chat("Give me a fun fact about space.")
    # print("AI:", resp3["reply"])

    # print("Session ID:", client.get_session_id())
    # print("Full History:", client.get_history())