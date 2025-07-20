import os
import csv
import io
from typing import Generator, Optional
import mmap
import platform

def is_windows():
    return platform.system() == 'Windows'

def load_text_from_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    
    try:
        ext = os.path.splitext(path)[1].lower()
        

        if ext in ['.txt', '.md', '.csv']:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                if os.path.getsize(path) > 10 * 1024 * 1024:

                    if is_windows():

                        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                            return mm.read().decode('utf-8', errors='ignore')
                    else:

                        with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as mm:
                            return mm.read().decode('utf-8', errors='ignore')
                else:

                    return f.read()

        elif ext == '.csv':
            try:
                import pandas as pd

                chunks = []
                for chunk in pd.read_csv(path, chunksize=10000, encoding='utf-8', on_bad_lines='skip'):
                    chunks.append(chunk.to_string())
                return '\n'.join(chunks)
            except ImportError:
                print("[ERROR] Module 'pandas' not installed. Install via 'pip install pandas'")

                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f)
                    return '\n'.join(','.join(row) for row in reader)

        elif ext == '.xlsx':
            try:
                import pandas as pd

                chunks = []
                for chunk in pd.read_excel(path, chunksize=10000):
                    chunks.append(chunk.to_string())
                return '\n'.join(chunks)
            except ImportError:
                print("[ERROR] Module 'pandas' not installed. Install via 'pip install pandas'")

        elif ext == '.docx':
            try:
                from docx import Document
                doc = Document(path)
                return "\n".join(p.text for p in doc.paragraphs)
            except ImportError:
                print("[ERROR] Module 'python-docx' not installed. Install via 'pip install python-docx'")

        elif ext == '.pdf':
            try:
                from pdfminer.high_level import extract_text
                from pdfminer.pdfparser import PDFSyntaxError
                from pdfminer.pdfdocument import PDFDocument
                from pdfminer.pdfpage import PDFPage
                from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
                from pdfminer.converter import TextConverter
                from pdfminer.layout import LAParams
                from io import StringIO
                
                def extract_text_with_layout(pdf_path):
                    resource_manager = PDFResourceManager()
                    string_io = StringIO()
                    codec = 'utf-8'
                    laparams = LAParams()
                    device = TextConverter(resource_manager, string_io, codec=codec, laparams=laparams)
                    
                    with open(pdf_path, 'rb') as file:
                        interpreter = PDFPageInterpreter(resource_manager, device)
                        for page in PDFPage.get_pages(file):
                            interpreter.process_page(page)
                    
                    text = string_io.getvalue()
                    device.close()
                    string_io.close()
                    return text

                try:

                    text = extract_text_with_layout(path)
                    if not text.strip():

                        text = extract_text(path)
                    
                    if not text.strip():
                        print(f"[WARNING] No text content found in PDF: {path}")
                        return ""
                    

                    text = ' '.join(text.split())
                    return text
                    
                except PDFSyntaxError as e:
                    print(f"[ERROR] Invalid or corrupted PDF file: {path}")
                    print(f"[ERROR] PDF Error: {str(e)}")
                    return ""
                except Exception as e:
                    print(f"[ERROR] Failed to extract text from PDF: {path}")
                    print(f"[ERROR] PDF Error: {str(e)}")
                    return ""
            except ImportError:
                print("[ERROR] Module 'pdfminer.six' not installed. Install via 'pip install pdfminer.six'")
                return ""

    except Exception as e:
        print(f"[ERROR] Failed to load file {path}: {e}")
    return ""

def stream_file_content(path: str, chunk_size: int = 1024 * 1024) -> Generator[str, None, None]:
    
    try:
        ext = os.path.splitext(path)[1].lower()
        
        if ext in ['.txt', '.md', '.csv']:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

        elif ext == '.csv':
            try:
                import pandas as pd
                for chunk in pd.read_csv(path, chunksize=10000, encoding='utf-8', on_bad_lines='skip'):
                    yield chunk.to_string()
            except ImportError:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        yield ','.join(row)

        elif ext == '.xlsx':
            try:
                import pandas as pd
                for chunk in pd.read_excel(path, chunksize=10000):
                    yield chunk.to_string()
            except ImportError:
                print("[ERROR] Module 'pandas' not installed. Install via 'pip install pandas'")

    except Exception as e:
        print(f"[ERROR] Failed to stream file {path}: {e}")
        yield ""
