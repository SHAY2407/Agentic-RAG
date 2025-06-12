import os
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import tempfile
import shutil
from pathlib import Path

def is_pdf_searchable(pdf_path, sample_pages=3):
    """
    Check if a PDF contains searchable text by examining the first few pages.
    Returns True if text is found, False if it's likely a scanned image.
    """
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        pages_to_check = min(sample_pages, total_pages)
        
        total_text_length = 0
        for page_num in range(pages_to_check):
            page = doc[page_num]
            text = page.get_text().strip()
            total_text_length += len(text)
        
        doc.close()
        
        # If we found very little text across multiple pages, it's likely scanned
        avg_text_per_page = total_text_length / pages_to_check
        return avg_text_per_page > 50  # Threshold for "searchable"
        
    except Exception as e:
        print(f"Error checking PDF {pdf_path}: {e}")
        return True  # Assume searchable if we can't check

def ocr_pdf_to_searchable(input_path, output_path, dpi=300):
    """
    Convert a scanned PDF to a searchable PDF using OCR.
    """
    try:
        print(f"Processing {input_path} with OCR...")
        
        # Convert PDF pages to images
        print("Converting PDF to images...")
        images = convert_from_path(input_path, dpi=dpi)
        
        # Create a new PDF document
        doc = fitz.open()
        
        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}...")
            
            # Save image to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
                image.save(tmp_img.name, 'PNG')
                
                # Perform OCR to get text and bounding boxes
                ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                
                # Create a new page with the image
                img_doc = fitz.open(tmp_img.name)
                page = doc.new_page(width=img_doc[0].rect.width, height=img_doc[0].rect.height)
                
                # Insert the image
                page.insert_image(page.rect, filename=tmp_img.name)
                
                # Add invisible text layer for searchability
                for j in range(len(ocr_data['text'])):
                    text = ocr_data['text'][j].strip()
                    if text and int(ocr_data['conf'][j]) > 30:  # Only add text with decent confidence
                        x = ocr_data['left'][j]
                        y = ocr_data['top'][j]
                        w = ocr_data['width'][j]
                        h = ocr_data['height'][j]
                        
                        # Create a rectangle for the text
                        rect = fitz.Rect(x, y, x + w, y + h)
                        
                        # Add invisible text (white text that matches the background)
                        page.insert_text(
                            (x, y + h),  # Bottom-left corner of text
                            text,
                            fontsize=max(8, h * 0.8),  # Approximate font size
                            color=(1, 1, 1),  # White text (invisible)
                            overlay=False
                        )
                
                img_doc.close()
                os.unlink(tmp_img.name)  # Clean up temporary image
        
        # Save the searchable PDF
        doc.save(output_path)
        doc.close()
        
        print(f"Successfully created searchable PDF: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def process_scanned_pdfs(data_folder="data", processed_folder="data/processed"):
    """
    Process all scanned PDFs in the data folder and create searchable versions.
    """
    data_path = Path(data_folder)
    processed_path = Path(processed_folder)
    
    # Create processed folder if it doesn't exist
    processed_path.mkdir(exist_ok=True)
    
    # Get all PDF files
    pdf_files = list(data_path.glob("*.pdf"))
    
    scanned_pdfs = []
    searchable_pdfs = []
    
    print(f"Found {len(pdf_files)} PDF files. Checking which ones need OCR...")
    
    for pdf_file in pdf_files:
        print(f"\nChecking: {pdf_file.name}")
        
        if is_pdf_searchable(str(pdf_file)):
            print(f"✓ {pdf_file.name} is already searchable")
            searchable_pdfs.append(pdf_file)
        else:
            print(f"⚠ {pdf_file.name} appears to be scanned - needs OCR")
            scanned_pdfs.append(pdf_file)
    
    print(f"\nSummary:")
    print(f"Already searchable: {len(searchable_pdfs)}")
    print(f"Need OCR processing: {len(scanned_pdfs)}")
    
    if scanned_pdfs:
        print(f"\nProcessing {len(scanned_pdfs)} scanned PDFs...")
        
        for pdf_file in scanned_pdfs:
            output_file = processed_path / f"ocr_{pdf_file.name}"
            
            if output_file.exists():
                print(f"Skipping {pdf_file.name} - OCR version already exists")
                continue
                
            success = ocr_pdf_to_searchable(str(pdf_file), str(output_file))
            
            if success:
                print(f"✓ Successfully processed {pdf_file.name}")
            else:
                print(f"✗ Failed to process {pdf_file.name}")
    
    return scanned_pdfs, searchable_pdfs

def main():
    print("PDF OCR Processor")
    print("=================")
    
    # Process PDFs
    scanned, searchable = process_scanned_pdfs()
    
    print(f"\nProcessing complete!")
    print(f"Original searchable PDFs: {len(searchable)}")
    print(f"Scanned PDFs processed: {len(scanned)}")
    
    if scanned:
        print(f"\nOCR-processed PDFs are saved in 'data/processed/' folder")
        print("You can now update your pdf.py to include these processed PDFs.")

if __name__ == "__main__":
    main()

