from google.colab import userdata
from openai import OpenAI

client = OpenAI(
    api_key=userdata.get("GROK_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

SYSTEM = """
ROLE: You are the assistant for Sweet Crumbs Home Bakery.

FACTS: We bake to order.
Orders need 24 hours' notice.
All cakes have eggless versions.
Plum cake Rs 500.
Chocolate cake Rs 650.
Cupcake Rs 80.

RULES: Never promise a delivery time.
If the answer isn't in FACTS, say "I'll check with the owner."

FORMAT: Under 50 words, friendly, English or Telugu matching the customer.
"""

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Can I get an eggless cake today evening?"}
    ]
)

print(response.choices[0].message.content)
print(response.usage)
