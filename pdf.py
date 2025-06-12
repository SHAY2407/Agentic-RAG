import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SentenceSplitter
from pathlib import Path

# Set the embedding model with improved chunking
Settings.chunk_size = 512  # Reduced chunk size for better granularity
Settings.chunk_overlap = 50  # Reduced overlap
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None

# Create a sentence splitter for better text chunking
text_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separator=" "
)

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.readers.file import PDFReader

def get_index(data, index_name):
    if not os.path.exists(index_name):
        print("building index", index_name)
        # Use the text splitter for better chunking
        index = VectorStoreIndex.from_documents(
            data, 
            transformations=[text_splitter],
            show_progress=True
        )
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )
    return index

# Load PDFs from both original data folder and processed folder
pdf_dirs = ["data", "data/processed"]
all_docs = []
pdf_files = []

for pdf_dir in pdf_dirs:
    if os.path.exists(pdf_dir):
        current_pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
        pdf_files.extend(current_pdf_files)
        
        for pdf_file in current_pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            try:
                docs = PDFReader().load_data(file=pdf_path)
                
                # Add enhanced metadata to each document
                for i, doc in enumerate(docs):
                    doc.metadata.update({
                        "file_name": pdf_file,
                        "file_path": pdf_path,
                        "source": pdf_dir,
                        "page_number": i + 1,
                        "total_pages": len(docs),
                        "document_type": "circular" if "circular" in pdf_file.lower() else "document"
                    })
                
                all_docs.extend(docs)
                print(f"Successfully loaded: {pdf_file} ({len(docs)} pages)")
            except Exception as e:
                print(f"Error loading {pdf_file}: {e}")

print(f"Total PDFs loaded: {len(set(pdf_files))}")
print(f"Total documents: {len(all_docs)}")

# Print file distribution for debugging
file_counts = {}
for doc in all_docs:
    file_name = doc.metadata.get('file_name', 'Unknown')
    file_counts[file_name] = file_counts.get(file_name, 0) + 1

print("\nDocument distribution by file:")
for file_name, count in file_counts.items():
    print(f"  {file_name}: {count} pages")

multi_pdf_index = get_index(all_docs, "multi_pdf_index")

# Create query engine with basic settings for speed
multi_pdf_engine = multi_pdf_index.as_query_engine(
    similarity_top_k=3,  # Reduced for speed
    verbose=False  # Disable verbose for speed
)
