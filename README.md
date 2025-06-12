# Agentic RAG System with OCR Support

A powerful Retrieval-Augmented Generation (RAG) system that can query population data, PDF documents, and save notes using Google's Gemini AI model. Now includes OCR support for scanned PDFs!

## Features

- **Population Data Queries**: Query world population statistics from CSV data
- **PDF Document Search**: Search through multiple PDF documents
- **Note Saving**: Save important information to text files
- **OCR Support**: Convert scanned PDFs to searchable text
- **Web UI**: Simple Streamlit interface for easy interaction
- **AI Agent**: Uses Google Gemini 2.0 Flash model for intelligent responses

## Setup

### 1. Activate Virtual Environment
```bash
source ai/bin/activate
```

### 2. Environment Configuration
Create a `.env` file with your Google Gemini API key:
```
API_KEY=your_gemini_api_key_here
```

### 3. Install System Dependencies (if not already installed)
```bash
# On Arch Linux
sudo pacman -S tesseract tesseract-data-eng poppler

# On Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils

# On macOS
brew install tesseract poppler
```

## Usage

### Web Interface
Start the Streamlit web application:
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

### Command Line Interface
Run the main script:
```bash
python main.py
```

### OCR Processing for Scanned PDFs

If you have scanned PDFs that need OCR processing:

```bash
python ocr_processor.py
```

This script will:
1. Automatically detect which PDFs are scanned (non-searchable)
2. Process them using OCR to extract text
3. Create searchable versions in the `data/processed/` folder
4. Update the system to include both original and OCR-processed PDFs

## File Structure

```
Agentic-RAG/
├── ai/                     # Virtual environment
├── data/                   # Original data files
│   ├── *.pdf              # PDF documents
│   ├── population.csv     # Population data
│   ├── notes.txt          # Saved notes
│   └── processed/         # OCR-processed PDFs
│       └── ocr_*.pdf      # Searchable versions of scanned PDFs
├── multi_pdf_index/        # Vector database index
├── app.py                  # Streamlit web interface
├── main.py                 # Command-line interface
├── ocr_processor.py        # OCR processing script
├── pdf.py                  # PDF handling and indexing
├── note_engine.py          # Note saving functionality
├── prompts.py              # AI prompts and context
└── README.md               # This file
```

## OCR Features

### Automatic Detection
The system automatically detects whether PDFs contain searchable text or are scanned images by analyzing the text content.

### High-Quality OCR
- Uses Tesseract OCR engine for text recognition
- Processes at 300 DPI for high accuracy
- Preserves original image quality
- Adds invisible text layer for searchability
- Filters low-confidence text (>30% confidence threshold)

### Batch Processing
Process multiple scanned PDFs in one run. The script:
- Skips already searchable PDFs
- Processes only scanned documents
- Creates OCR versions with "ocr_" prefix
- Automatically updates the RAG system

## Example Queries

- "What is the population of India?"
- "Show me countries with population over 100 million"
- "What information is available about toll fee revisions?"
- "Summarize the safety requirements from the compliance document"
- "Save a note: Meeting scheduled for tomorrow"
- "What are the key points about ATMS implementation?"

## Available Data

### PDF Documents
- Guidelines for provision of signages
- Policy circulars and notifications
- Compliance and safety requirements
- Toll fee revision documents
- NHAI administrative documents
- OCR-processed versions of scanned documents

### Population Data
- World population statistics
- Demographics information
- Country-wise data

## Technical Details

### AI Model
- Google Gemini 2.0 Flash for language understanding
- HuggingFace BGE embeddings for document search
- ReAct agent architecture for tool usage

### OCR Technology
- Tesseract 5.x OCR engine
- PDF2Image for page conversion
- PyMuPDF for PDF manipulation
- Invisible text layer preservation

### Vector Database
- LlamaIndex for document indexing
- Persistent storage for fast retrieval
- Chunk size: 2046 tokens with 80 token overlap

## Troubleshooting

### OCR Issues
- Ensure Tesseract is properly installed
- Check that poppler is available for PDF conversion
- Large PDFs may take significant time to process
- OCR-processed files will be larger than originals

### API Issues
- Verify your Gemini API key is correct
- Check for rate limiting if requests fail
- Ensure you have sufficient API quota

### Performance
- Initial indexing may take time for large document collections
- OCR processing is CPU-intensive
- Consider processing PDFs in smaller batches if memory is limited

## Contributing

To add new PDFs:
1. Place them in the `data/` folder
2. Run `python ocr_processor.py` to process any scanned documents
3. Restart the application to rebuild the index

The system will automatically detect and process new scanned PDFs while preserving existing searchable documents.

