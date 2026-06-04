import os
import sys
import pandas as pd

from dotenv import load_dotenv
from pymongo import MongoClient

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "rag_chat")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "knowledge_base")
INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME", "vector_index")

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/rag_sample_qas_from_kis.csv"

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is missing")


def main():
    print(f"Reading CSV from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    required_columns = [
        "ki_topic",
        "ki_text",
        "sample_question",
        "sample_ground_truth",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. Found columns: {list(df.columns)}"
        )

    client = MongoClient(MONGODB_URI)
    collection = client[MONGODB_DB][MONGODB_COLLECTION]

    print("Clearing old documents...")
    collection.delete_many({})

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

    documents = []

    for idx, row in df.iterrows():
        topic = str(row["ki_topic"]).strip()
        ki_text = str(row["ki_text"]).strip()
        sample_question = str(row["sample_question"]).strip()
        ground_truth = str(row["sample_ground_truth"]).strip()

        if (
            not topic
            or not ki_text
            or not sample_question
            or not ground_truth
            or topic == "nan"
            or ki_text == "nan"
            or sample_question == "nan"
            or ground_truth == "nan"
        ):
            continue

        # This is the actual searchable knowledge document.
        text = f"""
Topic: {topic}

Knowledge base article:
{ki_text}

Example user question:
{sample_question}

Reference answer:
{ground_truth}
""".strip()

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "row": int(idx),
                    "topic": topic,
                    "sample_question": sample_question,
                    "ground_truth": ground_truth,
                    "source": "rag_sample_qas_from_kis.csv",
                },
            )
        )

    print(f"Ingesting {len(documents)} documents...")

    if not documents:
        raise ValueError("No valid documents found in CSV")

    vector_store.add_documents(documents)

    print("Done.")


if __name__ == "__main__":
    main()