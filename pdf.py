import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
from llama_index.core.settings import Settings

# Set the embedding model (correct)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None

# If you want an actual LLM, import and set it (optional)
# from llama_index.llms.openai import OpenAI
# Settings.llm = OpenAI(model="gpt-3.5-turbo")

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

pdf_path = os.path.join("data", "CircularMandatory.pdf")
circular_pdf = PDFReader().load_data(file=pdf_path)
circular_index = get_index(circular_pdf, "circular")
circular_engine = circular_index.as_query_engine()
