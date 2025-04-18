
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_to_dict

memory = ConversationBufferMemory(return_messages=True)

# Load environment variables
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

def search_articles(query):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY}
    data = {"q": query}

    response = requests.post(url, headers=headers, json=data)
    results = response.json()

    articles = []
    for result in results.get("organic", [])[:5]:
        articles.append({
            "url": result["link"],
            "title": result["title"]
        })
    return articles

def fetch_article_content(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all(['h1', 'h2', 'h3', 'p'])
        content = "\n".join([p.get_text() for p in paragraphs])
        return content.strip()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def concatenate_content(articles):
    full_text = ""
    for article in articles:
        text = fetch_article_content(article['url'])
        full_text += f"\n\nTitle: {article['title']}\n{text}"
    return full_text[:12000]

memory = ConversationBufferMemory(return_messages=True)
def generate_answer(content, query):
    memory.chat_memory.add_user_message(query)

    # Prepare conversation history
    conversation_history = memory.load_memory_variables({})["history"]
    conversation_history_text = "\n".join(
        [f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in messages_to_dict(conversation_history)]
    )

    # Combine history with the current query
    prompt = f"Conversation history:\n{conversation_history_text}\n\nBased on the following content:\n\n{content}\n\nAnswer the question:\n{query}"
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version=2024-12-01-preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY
    }
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        answer = response.json()["choices"][0]["message"]["content"].strip()
        # Add assistant response to memory
        memory.chat_memory.add_ai_message(answer)
        return answer
    else:
        print("Azure OpenAI Error:", response.text)
        return "Sorry, I couldn't generate an answer at the moment."