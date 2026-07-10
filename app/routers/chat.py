from fastapi import APIRouter
from pydantic import BaseModel
from app.tools.tool_definitions import tools
from app.tools.tool_runner import run_tool
import json
import requests
from openai import OpenAI
from app.services.google_places import search_places

client = OpenAI()
router = APIRouter()


class ChatRequest(BaseModel):
    question: str


# URL validator
def is_valid_website(url: str) -> bool:
    try:
        if not url:
            return False

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        response = requests.get(url, timeout=5, allow_redirects=True)

        bad_signs = [
            "domain may be for sale",
            "buy this domain",
            "resultsfinder",
            "parked",
        ]

        final_url = str(response.url).lower()
        page_text = response.text.lower()[:1000]

        if response.status_code >= 400:
            return False

        if any(bad in final_url for bad in bad_signs):
            return False

        if any(bad in page_text for bad in bad_signs):
            return False

        return True

    except Exception:
        return False

def explain_places_with_ai(user_question: str, places: list):
    if not places:
        return {"results": []}

    prompt = f"""
You are a helpful AI assistant for kids activities.

User question:
{user_question}

Here are real place results:
{json.dumps(places, indent=2)}

Return ONLY valid JSON in this format:
{{
  "results": [
    {{
      "name": "string",
      "website": "string",
      "reason": "short kid-friendly explanation"
    }}
  ]
}}

Rules:
- Use only the places provided
- Do not invent new businesses
- Keep each reason to one short sentence
- Focus on why it may fit families or kids
- If address is available, you may mention it briefly
- Preserve website exactly as given
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=500,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or '{"results": []}'
    try:
        return json.loads(content)
    except Exception:
        return {"results": places}

    #return json.loads(content)

@router.post("/chat")

async def chat(request: ChatRequest):
    places = search_places(request.question, max_results=3)
    

    if places:
        ai_answer = explain_places_with_ai(request.question, places)
        return {"answer": ai_answer}
    

    messages = [
        {
            "role": "system",
            "content": """Return ONLY valid JSON.

Format:
{
  "results": [
    {
      "name": "string",
      "website": "string",
      "reason": "string"
    }
  ]
}

Rules:
- Include real businesses
- Only include official business websites
- Do NOT include blogs, news, or aggregator sites
- If unsure about website, return empty string ""
- No extra text outside JSON
"""
        },
        {"role": "user", "content": request.question},
    ]

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        tools=tools,
        temperature=0.3,
        max_tokens=700,
        response_format={"type": "json_object"},
    )

    message = response.choices[0].message

    # handle tool calls
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
                    "content": json.dumps(result),
                }
            )

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            temperature=0.3,
            max_tokens=700,
            response_format={"type": "json_object"},
        )

    # extract response
    answer = response.choices[0].message.content or '{"results": []}'

    # parse + clean
    try:
        parsed = json.loads(answer)
        if isinstance(parsed, str):
         parsed = json.loads(parsed)

        if not isinstance(parsed, dict):
         parsed = {"results": []}

        results = parsed.get("results", [])

        if not isinstance(results, list):
         results = []

        clean_results = []

        blocked_domains = [
            "mummytravels",
            "secret",
            "tripadvisor",
            "yelp",
            "resultsfinder",
            "blog",
            "news",
        ]

        for place in results:
            if not isinstance(place, dict):
                continue

        website = str(place.get("website", "")).strip()

        if website and not website.startswith(("http://", "https://")):
            website = "https://" + website

        if any(bad in website.lower() for bad in blocked_domains):
            website = ""

        elif website and is_valid_website(website):
            pass

        else:
            website = ""

        clean_results.append(
            {
                "name": str(place.get("name", "Activity")),
                "website": website,
                "reason": str(place.get("reason", "")),
            }
        )

        return {
        "answer": {
            "results": clean_results
        }
    }

    except (json.JSONDecodeError, TypeError, AttributeError) as error:
        print(f"Failed to parse OpenAI response: {error}")
        print(f"Raw response: {answer}")

        return {
            "answer": {
                "results": []
          }
       }