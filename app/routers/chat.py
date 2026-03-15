from fastapi import APIRouter
from pydantic import BaseModel
from app.tools.tool_definitions import tools
from app.tools.tool_runner import run_tool
import json
from openai import OpenAI

client = OpenAI()
router = APIRouter()


class ChatRequest(BaseModel):
    question: str


@router.post("/chat")
async def chat(request: ChatRequest):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant for kids activities and family-friendly local events. "
                "Give practical, location-aware answers. When helpful, suggest specific activity "
                "types like art classes, pottery studios, museums, parks, storytime, camps, or weekend events. "
                "Use tools when needed to fetch activity data or answer questions about activities."
            ),
        },
        {"role": "user", "content": request.question},
    ]

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        tools=tools,
    )

    message = response.choices[0].message

    if message.tool_calls:
        messages.append(message)

        for tool_call in message.tool_calls:
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            result = run_tool(
                tool_call.function.name,
                tool_args,
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
        )

    answer = response.choices[0].message.content or "Sorry, I couldn't generate a response."
    return {"answer": answer}