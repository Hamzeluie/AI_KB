import asyncio
import requests
import websockets
from fastmcp import Client

async def fetch_tools():
    """Fetch and format tools from the MCP server."""
    async with Client("http://127.0.0.1:13080/sse") as client:
        tools = await client.list_tools()  # Should succeed!
        tools_list = []
        for tool in tools:
            # Optionally, format tools for system prompt
            tool_str = (
                f"Tool(name='{tool.name}', title={tool.title}, "
                f"description='{tool.description}', inputSchema={tool.inputSchema}, "
                f"outputSchema={tool.outputSchema}, annotations={tool.annotations}, "
                f"meta={tool.meta})"
            )
            tools_list.append(f"<tool>\n{tool_str}\n</tool>")
        return tools_list

async def main():
    # tools = await fetch_tools()
    # for tool in tools:
    #     print(tool)

    async with Client("http://127.0.0.1:13080/sse") as client:
        # result = await client.call_tool(
        #     "channels_list",
        #     arguments={
        #         "channel_types": "public_channel,private_channel"
        #     }
        # )
        # print(result)
        result = await client.call_tool(
            "channels_list",
            arguments={
                "channel_types": "im"
            }
        )
        print(result)
        result = await client.call_tool(
            "conversations_add_message",
            arguments={
                "channel_id": "@alrzbqr",
                "payload": "In omnipotent we trust!"
            }
        )

if __name__ == "__main__":
    asyncio.run(main())