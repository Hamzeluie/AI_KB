import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

import requests
from fastmcp import Client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Qwen LLM server URL
QWEN_BASE_URL = "http://localhost:8100"


def send_prompt(
    prompt: str,
    base_url: str = QWEN_BASE_URL,
    session_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_tokens: int = 3000,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> Dict:
    """
    Send a prompt to the Qwen chat API.
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
        "top_p": top_p,
    }

    try:
        response = requests.post(
            f"{base_url}/chat",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to send prompt: {str(e)}")


def parse_llm_output(raw_output: str) -> Dict[str, Any]:
    """
    Extracts ALL valid tool calls from the response text.
    Skips malformed or invalid JSON blocks.
    Returns 'tool_calls' type if at least one valid call found.
    Otherwise, returns 'text' type with error info.
    """
    start_tag = "<tool_call>"
    end_tag = "</tool_call>"

    tool_call_blocks = re.findall(
        rf"{re.escape(start_tag)}\s*(.*?)\s*{re.escape(end_tag)}",
        raw_output,
        re.DOTALL,
    )

    valid_tool_calls = []
    errors = []

    for i, block in enumerate(tool_call_blocks):
        try:
            tool_call_data = json.loads(block)
            if (
                "function_name" not in tool_call_data
                or "arguments" not in tool_call_data
            ):
                raise KeyError("Missing 'function_name' or 'arguments'")
            valid_tool_calls.append(
                {
                    "name": tool_call_data["function_name"],
                    "arguments": tool_call_data["arguments"],
                }
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            errors.append(f"Block {i + 1}: {str(e)}")

    if valid_tool_calls:
        result = {
            "type": "tool_calls",
            "tool_calls": valid_tool_calls,
        }
        if errors:
            result["parsing_errors"] = errors
        return result
    else:
        error_detail = "; ".join(errors) if errors else "No valid tool calls found"
        return {
            "type": "text",
            "content": raw_output,
            "error": error_detail,
        }


async def return_tools():
    """
    Fetch available tools from MCP server.
    """
    async with Client("http://localhost:6000/mcp") as client:
        tools_list = []
        tools = await client.list_tools()
        for tool in tools:
            tool_str = (
                f"Tool(name='{tool.name}', title={tool.title}, "
                f"description='{tool.description}', inputSchema={tool.inputSchema}, "
                f"outputSchema={tool.outputSchema}, annotations={tool.annotations}, "
                f"meta={tool.meta})"
            )
            tools_list.append(f"<tool>\n{tool_str}\n</tool>")
        return tools_list


async def llm(user_input: str, max_tool_iterations: int = 10):
    """
    Main LLM interaction loop using Qwen REST API.
    """
    owner_id = "+12345952496"
    kb_id = "kb+12345952496_en"  # Still used for tool context if needed
    config = {"temperature": 0.7, "max_tokens": 3000}
    system_prompt_template = """
You are a helpful assistant with access to external tools to assist in answering user queries. Use a tool only when strictly necessary to provide an accurate and complete response. If you can answer the query directly with your knowledge, do not use a tool—respond with the final answer in plain text.

### Available Tools:
{{TOOLS}}

Each tool is defined within <tool> tags as a JSON object with:
- **name**: The tool's identifier (e.g., 'code_execution', 'search').
- **description**: What the tool does.
- **inputSchema**: A JSON schema specifying required and optional parameters.
- **outputSchema**: The expected output format (typically a JSON string).

### Tool Call Instructions:
When a tool is necessary:
- Output **only** the tool call(s) using the mandatory <tool_call> and <tool_call> tags, with no additional text, explanations, or partial answers.
- Use the following format for each tool call, ensuring both opening <tool_call> and closing <tool_call> tags are included:
  <tool_call>
  {"function_name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}
  <tool_call>
- **function_name**: Must match the tool's name exactly.
- **arguments**: A JSON object with keys and values strictly adhering to the tool's inputSchema.
- Example for the 'search' tool:
  <tool_call>
  {"function_name": "search", "arguments": {"query": "What is the price of usb hub?", "num_results": 5}}
  </tool_call>
- For multiple tool calls, output consecutive <tool_call> blocks.
- Do not escape values; provide them as plain text within the JSON structure.
- Ensure the JSON is valid and fully complies with the tool's inputSchema.
- Always include both the opening <tool_call> and closing <tool_call> tags.

### Response Instructions:
- **If a tool is needed**: Output only the <tool_call> block(s).
- **If no tool is needed**: Respond directly with the answer in plain text.
- **After tool results**: Incorporate them and provide a final answer in plain text unless further tools are needed.
"""
    
    tools_list = await return_tools()
    system_prompt = system_prompt_template.replace("{{TOOLS}}", "\n\n".join(tools_list))

    session_id = None  # Will be set by first Qwen response
    current_input = user_input
    iteration = 0

    while iteration < max_tool_iterations:
        iteration += 1
        logging.info(f"Iteration {iteration}: Sending to LLM: {current_input[:100]}...")

        try:
            response = send_prompt(
                prompt=current_input,
                session_id=session_id,
                system_prompt=system_prompt,
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
            )
        except Exception as e:
            logging.error(f"Qwen API error: {e}")
            return f"LLM Error: {str(e)}", False

        session_id = response.get("session_id")  # Update for next turn
        message_content = response.get("reply", "").strip()

        if not message_content:
            logging.warning("Empty LLM response")
            return "No response from LLM", False

        parsed_response = parse_llm_output(message_content)
        logging.info(f"Parsed Response Type: {parsed_response.get('type')}")

        if parsed_response["type"] == "text":
            final_text = parsed_response.get("content", "").strip()
            if final_text:
                logging.info("✅ Final Answer from LLM (no tool needed)")
                # Use reflection to check the relevance of the answer with user_input
                
                response = send_prompt(
                    prompt=f'This is the user query: {user_input}\n\nThis is the LLM response: {final_text}\n',
                    session_id=session_id,
                    system_prompt=f'Does the LLM response adequately and accurately answer the user query? Reply with "Yes" or "No" only. You will be penalized for saying something other than "Yes" or "No".',
                    max_tokens=config["max_tokens"],
                    temperature=config["temperature"],
                )
                print("Reflection response:", response.get("reply", "").strip())
                if response.get("reply", "").strip().lower() == "no":
                    logging.info("LLM reflection indicates answer is inadequate, continuing...")
                    current_input = f"""
Previous user query: {user_input}
LLM response: {final_text}
The LLM response does not adequately answer the user query. Please try again, using tools if necessary to provide a complete and accurate answer.
                    """
                    continue  # Restart loop to try again
                return final_text, False
            else:
                error_msg = parsed_response.get("error", "Unknown error")
                logging.warning(f"LLM returned text with error: {error_msg}")
                return f"LLM Error: {error_msg}", False

        elif parsed_response["type"] == "tool_calls":
            tool_calls = parsed_response.get("tool_calls", [])
            tool_results = []

            async with Client("http://localhost:6000/mcp") as client:
                for i, tool_call in enumerate(tool_calls):
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("arguments", {})

                    logging.info(
                        f"➡️  Executing tool {i+1}/{len(tool_calls)}: {tool_name} with args: {tool_args}"
                    )
                    try:
                        result = await client.call_tool(tool_name, tool_args)
                        tool_results.append(
                            {
                                "tool_name": tool_name,
                                "arguments": tool_args,
                                "result": result,
                            }
                        )
                        logging.info(f"✅ Tool {tool_name} executed successfully")
                    except Exception as e:
                        error_detail = f"Tool '{tool_name}' failed: {str(e)}"
                        logging.error(error_detail)
                        tool_results.append(
                            {
                                "tool_name": tool_name,
                                "arguments": tool_args,
                                "error": str(e),
                            }
                        )

            # Format feedback for next LLM turn
            tool_feedback = "\n".join(
                f"Tool '{tr['tool_name']}' called with {tr['arguments']} returned: {tr.get('result', 'ERROR: ' + tr.get('error', 'Unknown'))}"
                for tr in tool_results
            )

            current_input = f"""
Previous user query: {user_input}

Tool execution results:
{tool_feedback}

Based on these results, please provide a final answer to the user's original question.
Do NOT call any more tools unless absolutely necessary.
If you must call another tool, use the <tool_call> tags again.
Otherwise, respond with plain text only.
"""
            logging.info("Sending tool results back to LLM for synthesis...")

        else:
            logging.error(f"Unknown response type: {parsed_response.get('type')}")
            return "Unknown LLM response type", False

    return "Max tool iterations reached. Could not produce final answer.", False


async def main():
    user_input = "What is the price of usb hub? Use the search tool to find the answer."

    print("=" * 60)
    print(f"User: {user_input}")
    print("=" * 60)

    result, tool_used = await llm(user_input)

    print("\n" + "=" * 60)
    if tool_used:
        print("🔧 Final Tool Result:")
    else:
        print("💬 Final LLM Response:")
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())