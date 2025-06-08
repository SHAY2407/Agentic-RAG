import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
from llama_index.core.settings import Settings

# Set the embedding model
Settings.chunk_size = 2046
Settings.chunk_overlap = 80
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.readers.file import PDFReader

def get_index(data, index_name):
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )
    return index

# Load multiple PDFs
pdf_dir = "data"
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
all_docs = []
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_dir, pdf_file)
    docs = PDFReader().load_data(file=pdf_path)
    all_docs.extend(docs)

multi_pdf_index = get_index(all_docs, "multi_pdf_index")
multi_pdf_engine = multi_pdf_index.as_query_engine()