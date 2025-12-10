from qdrant_client import QdrantClient
print(dir(QdrantClient))
try:
    client = QdrantClient(":memory:")
    print("search" in dir(client))
except Exception as e:
    print(e)

