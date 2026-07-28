from embeddings import HFTextEmbedding
from chat_vdb import ChatVectorStore

embedder = HFTextEmbedding("en")
store = ChatVectorStore(persist_dir="./chroma_db_chat", embedding_fn=embedder)
        
store.list_all(limit=100)
# store.col.delete(ids=["31:1756273603:chunk:0"])


