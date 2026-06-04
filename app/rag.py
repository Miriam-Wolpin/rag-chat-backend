import os
from typing import List, Tuple

from dotenv import load_dotenv
from pymongo import MongoClient

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

from app.schemas import ChatMessage

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "rag_chat")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "knowledge_base")
INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME", "vector_index")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is missing")

print("\n===== RAG CONFIG DEBUG =====")
print("MONGODB_DB:", MONGODB_DB)
print("MONGODB_COLLECTION:", MONGODB_COLLECTION)
print("INDEX_NAME:", INDEX_NAME)
print("OPENAI_API_KEY exists:", bool(OPENAI_API_KEY))
print("MONGODB_URI exists:", bool(MONGODB_URI))
print("============================\n")

client = MongoClient(MONGODB_URI)
collection = client[MONGODB_DB][MONGODB_COLLECTION]

try:
    doc_count = collection.count_documents({})
    print(f"MongoDB document count in {MONGODB_DB}.{MONGODB_COLLECTION}: {doc_count}")

    first_doc = collection.find_one({})
    if first_doc:
        print("First document keys:", list(first_doc.keys()))
        print("First document text preview:", str(first_doc.get("text", ""))[:400])
        embedding = first_doc.get("embedding")
        print("First document has embedding:", isinstance(embedding, list))
        print("First embedding length:", len(embedding) if isinstance(embedding, list) else None)
    else:
        print("No documents found in collection.")

except Exception as e:
    print("Could not read MongoDB collection during startup:", repr(e))

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)

vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name=INDEX_NAME,
    text_key="text",
    embedding_key="embedding",
    relevance_score_fn="cosine",
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=OPENAI_API_KEY,
)


def _to_langchain_history(history: List[ChatMessage]):
    messages = []

    for msg in history[-8:]:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))

    return messages


def _format_docs(docs: List[Document]) -> str:
    if not docs:
        return "No relevant knowledge base context was found."

    parts = []

    for i, doc in enumerate(docs, start=1):
        topic = doc.metadata.get("topic", "")
        sample_question = doc.metadata.get("sample_question", "")
        ground_truth = doc.metadata.get("ground_truth", "")
        content = doc.page_content

        parts.append(
            f"""Source {i}:
Topic: {topic}
Example question: {sample_question}
Reference answer: {ground_truth}

Full knowledge base context:
{content}
"""
        )

    return "\n---\n".join(parts)


def _debug_print_docs(message: str, docs: List[Document]):
    print("\n\n===== DEBUG RAG REQUEST =====")
    print("User message:", message)
    print("Retrieved docs count:", len(docs))

    if not docs:
        print("No documents were retrieved.")
        print("Likely causes:")
        print("1. MongoDB Atlas Vector Search index is missing.")
        print("2. Index name does not match:", INDEX_NAME)
        print("3. Index is not active yet.")
        print("4. Index path is not 'embedding'.")
        print("5. embedding dimension in index is not 1536.")
    else:
        for i, doc in enumerate(docs, start=1):
            print(f"\n--- RETRIEVED DOC {i} ---")
            print("Metadata:", doc.metadata)
            print("Page content preview:")
            print(doc.page_content[:1000])

    print("===== END DEBUG RAG REQUEST =====\n\n")


def answer_chat(message: str, history: List[ChatMessage]) -> Tuple[str, List[dict]]:
    """
    Main RAG function with debug:
    1. Checks MongoDB collection.
    2. Retrieves docs from MongoDB Atlas Vector Search.
    3. Prints retrieved docs to server terminal.
    4. Sends retrieved context to OpenAI.
    """

    try:
        current_count = collection.count_documents({})
        print(f"\nCurrent MongoDB document count: {current_count}")
    except Exception as e:
        print("Could not count documents before retrieval:", repr(e))

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    try:
        docs = retriever.invoke(message)
    except Exception as e:
        print("\n===== VECTOR SEARCH ERROR =====")
        print("Error type:", type(e).__name__)
        print("Error:", repr(e))
        print("===============================\n")

        return (
            "There was an error while searching the knowledge base. Please check the backend logs.",
            [],
        )

    _debug_print_docs(message, docs)

    context = _format_docs(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful IT support assistant answering questions using a provided knowledge base.

The knowledge base context below contains the most relevant retrieved articles.
Use it to answer the user's question.

Important rules:
- If the retrieved context is related to the user's question, answer using it.
- The user's wording may differ from the example question in the context.
- Do not require an exact wording match.
- Do not invent details that are not supported by the context.
- If the retrieved context is clearly unrelated or empty, say: "I couldn't find that in the knowledge base."
- Keep the answer concise and practical.

Knowledge base context:
{context}
""",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    chain = prompt | llm

    try:
        response = chain.invoke(
            {
                "context": context,
                "history": _to_langchain_history(history),
                "question": message,
            }
        )
    except Exception as e:
        print("\n===== OPENAI CHAT ERROR =====")
        print("Error type:", type(e).__name__)
        print("Error:", repr(e))
        print("=============================\n")

        return (
            "There was an error while generating the answer. Please check the backend logs.",
            [],
        )

    sources = []

    for doc in docs:
        sources.append(
            {
                "question": doc.metadata.get("sample_question"),
                "answer": doc.metadata.get("ground_truth"),
                "score": None,
            }
        )

    print("\n===== FINAL ANSWER DEBUG =====")
    print("Answer:", response.content)
    print("Sources count:", len(sources))
    print("==============================\n")

    return response.content, sources