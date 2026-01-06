"""
PDF Processing Module - Optimized for Performance
Handles text extraction with parallel processing and smart caching
"""

import io
import hashlib
import concurrent.futures
from typing import List, Dict, Tuple, Optional
import PyPDF2
import streamlit as st


class PDFProcessor:
    """High-performance PDF text extraction with parallel processing"""
    
    # Class-level cache for extracted text (persists across calls)
    _text_cache: Dict[str, str] = {}
    
    @staticmethod
    def get_file_hash(file_content: bytes) -> str:
        """Generate MD5 hash for file content to use as cache key"""
        return hashlib.md5(file_content).hexdigest()
    
    @staticmethod
    def extract_text_from_pdf(file_content: bytes) -> Optional[str]:
        """
        Extract text from PDF with optimized processing.
        Uses page-level parallelization for large PDFs.
        """
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            num_pages = len(pdf_reader.pages)
            
            # For small PDFs (< 5 pages), process sequentially
            if num_pages < 5:
                text_parts = []
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"\n--- Page {page_num+1} ---\n{page_text}")
                text = "".join(text_parts)
            else:
                # For larger PDFs, extract pages in parallel
                def extract_page(args):
                    page_num, page = args
                    page_text = page.extract_text()
                    return (page_num, f"\n--- Page {page_num+1} ---\n{page_text}" if page_text else "")
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(extract_page, enumerate(pdf_reader.pages)))
                
                # Sort by page number and join
                results.sort(key=lambda x: x[0])
                text = "".join(r[1] for r in results)
            
            if not text.strip():
                return None
            
            return text
            
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return None
    
    @classmethod
    def extract_with_cache(cls, file_content: bytes) -> Tuple[str, str]:
        """
        Extract text with caching - returns (hash, text)
        Uses class-level cache for session persistence
        """
        file_hash = cls.get_file_hash(file_content)
        
        if file_hash in cls._text_cache:
            return file_hash, cls._text_cache[file_hash]
        
        text = cls.extract_text_from_pdf(file_content)
        if text:
            cls._text_cache[file_hash] = text
        
        return file_hash, text
    
    @classmethod
    def process_multiple_pdfs_parallel(cls, pdf_files: List[Dict]) -> List[Dict]:
        """
        Process multiple PDFs in parallel for maximum speed.
        
        Args:
            pdf_files: List of dicts with 'name' and 'content' keys
            
        Returns:
            List of dicts with 'name', 'hash', 'text', 'success' keys
        """
        results = []
        
        def process_single(pdf_file):
            file_content = pdf_file['content']
            file_hash, text = cls.extract_with_cache(file_content)
            return {
                'name': pdf_file['name'],
                'hash': file_hash,
                'text': text,
                'success': text is not None
            }
        
        # Use ThreadPoolExecutor for I/O-bound PDF extraction
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_pdf = {executor.submit(process_single, pdf): pdf for pdf in pdf_files}
            
            for future in concurrent.futures.as_completed(future_to_pdf):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    pdf = future_to_pdf[future]
                    results.append({
                        'name': pdf['name'],
                        'hash': None,
                        'text': None,
                        'success': False,
                        'error': str(e)
                    })
        
        return results
    
    @classmethod
    def clear_cache(cls):
        """Clear the text extraction cache"""
        cls._text_cache.clear()
