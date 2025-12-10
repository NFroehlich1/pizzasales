import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import streamlit as st

# Configuration
QDRANT_URL = "https://78fc5c76-da3d-4702-ab3a-b74a61f84aba.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.kI5r8TWne69k0N8IriWA07HhCVUt3AH5Pq5_Z0NsFXE"
COLLECTION_NAME = "pizza_sales"

def format_row(row):
    """
    Creates a text representation of a pizza order for embedding.
    """
    return (
        f"Order ID: {row['order_id']}, Date: {row['order_date']}, Time: {row['order_time']}. "
        f"Item: {row['quantity']}x {row['pizza_size']} {row['pizza_name']} ({row['pizza_category']}). "
        f"Ingredients: {row['pizza_ingredients']}. "
        f"Unit Price: ${row['unit_price']}, Total Price: ${row['total_price']}."
    )

def main():
    print("Initializing Qdrant client...")
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    print("Loading data...")
    try:
        df = pd.read_csv('pizza_sales.csv')
    except FileNotFoundError:
        print("Error: pizza_sales.csv not found.")
        return

    # Create text descriptions for embedding
    print("Preparing data for embedding...")
    documents = df.apply(format_row, axis=1).tolist()
    
    # Generate metadata
    payloads = df.to_dict(orient='records')

    print("Loading embedding model (this may take a moment)...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Generating embeddings for {len(documents)} records...")
    embeddings = encoder.encode(documents, show_progress_bar=True)

    print(f"Recreating collection '{COLLECTION_NAME}'...")
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE
        )
    )

    print("Uploading data to Qdrant...")
    # Batch upload for efficiency
    points = [
        models.PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload=payload
        ) for idx, (embedding, payload) in enumerate(zip(embeddings, payloads))
    ]
    
    qdrant_client.upload_points(
        collection_name=COLLECTION_NAME,
        points=points,
        batch_size=256
    )

    print("Upload complete!")
    print(f"Collection '{COLLECTION_NAME}' is ready with {len(documents)} vectors.")
    
    # Verify
    print("Verifying collection info:")
    print(qdrant_client.get_collection(COLLECTION_NAME))

if __name__ == "__main__":
    main()

