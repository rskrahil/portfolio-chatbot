import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq

load_dotenv()

# Again the same thing used in the ingest.py file as the configuration at the top!
CHROMA_DIR  = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL  = "llama-3.3-70b-versatile"

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

print("Loading ChromaDB...")
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("chat.py ready!")


def get_answer(question: str) -> str:

    docs = vectorstore.similarity_search(question, k=3)

    context = ""
    for i, doc in enumerate(docs):
        context += f"--- chunk {i+1} ---\n{doc.page_content}\n\n"

    prompt = f"""You are an AI assistant for Rahil Shaikh's portfolio website.
Answer questions about Rahil using ONLY the context provided below.
If the answer is not in the context, say "I don't have that information about Rahil."
Keep answers concise and professional.

Context:
{context}
Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print("\nChatbot ready! Type 'quit' to exit.\n")
    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break
        answer = get_answer(question)
        print(f"Bot: {answer}\n")
