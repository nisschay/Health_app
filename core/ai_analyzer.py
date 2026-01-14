"""
AI Analysis Module - Optimized Gemini Integration
Handles medical report analysis with batching and streaming
"""

import re
import json
import hashlib
import concurrent.futures
import threading
import logging
from typing import Dict, List, Optional, Any
import streamlit as st
import google.generativeai as genai

# Get logger
logger = logging.getLogger('MedicalAnalyzer')


class AIAnalyzer:
    """
    High-performance AI analysis with optimized prompts and caching.
    
    Key optimizations:
    - Reduced prompt size (removed redundant instructions)
    - Response caching with content hash
    - PARALLEL processing for multiple PDFs
    - Streaming responses for better UX
    """
    
    _extraction_model = None
    _chat_model = None
    _response_cache: Dict[str, Dict] = {}
    _lock = threading.Lock()  # Thread-safe cache access
    
    # Optimized, shorter prompt template (reduced from ~2500 to ~1200 tokens)
    EXTRACTION_PROMPT = """Extract medical test data from this report as JSON.

REQUIRED OUTPUT FORMAT:
{{
    "patient_info": {{
        "name": "Full Name (prefer complete formal names)",
        "age": "Age (e.g., '35 years')",
        "gender": "Male/Female",
        "patient_id": "ID if present",
        "date": "DD-MM-YYYY format",
        "lab_name": "Main laboratory BRAND name from header/logo (e.g., 'Neuberg Abha', 'Thyrocare', 'Apollo Diagnostics', 'Dr Lal PathLabs')"
    }},
    "test_results": [
        {{
            "test_name": "Test Name",
            "result": "Value or finding",
            "unit": "Unit",
            "reference_range": "Normal range",
            "status": "Low/Normal/High/Critical/Positive/Negative/N/A",
            "category": "Haematology/Liver Function Test/Kidney Function Test/Lipid Profile/Thyroid Profile/etc."
        }}
    ]
}}

RULES:
- Date must be DD-MM-YYYY (day first)
- Extract ALL test parameters
- Infer status from result vs reference range if not stated
- Lab name: Extract the MAIN BRAND NAME from the header/logo at the TOP of the report (e.g., "Neuberg Abha", "Thyrocare", "Apollo"). NEVER use billing location addresses or branch names. Look for the prominent company name/logo.

REPORT TEXT:
---
{text}
---"""
    
    # Shorter chat prompt
    CHAT_PROMPT = """Medical assistant analyzing test results. Be precise and use bullet points.

Guidelines:
- Explain results simply without medical advice
- For abnormal values, include reference range
- Categorize findings as Normal/Borderline/Concerning
- Use sub-bullets for details

Recent Chat:
{history}

Test Data:
{data}

User: {question}

Assistant (use bullet points):"""
    
    @classmethod
    def initialize(cls, api_key: str) -> bool:
        """Initialize Gemini models with the provided API key"""
        try:
            if not api_key:
                st.error("Gemini API Key is missing.")
                return False
            
            genai.configure(api_key=api_key)
            
            # Using flash model for speed
            cls._extraction_model = genai.GenerativeModel('gemini-2.5-flash')
            cls._chat_model = genai.GenerativeModel('gemini-2.5-flash')
            
            return True
        except Exception as e:
            st.error(f"Error configuring Gemini: {e}")
            cls._extraction_model = None
            cls._chat_model = None
            return False
    
    @classmethod
    def _get_content_hash(cls, text: str) -> str:
        """Generate hash for text content"""
        return hashlib.md5(text.encode()).hexdigest()
    
    @classmethod
    def analyze_report(cls, report_text: str, api_key: str, filename: str = "unknown") -> tuple:
        """
        Analyze a medical report and extract structured data.
        Uses caching to avoid re-analyzing identical content.
        Thread-safe for parallel processing.
        
        Returns:
            tuple: (result_dict or None, error_message or None)
        """
        if not cls._extraction_model and not cls.initialize(api_key):
            return None, "AI model not initialized"
        
        if not report_text or not report_text.strip():
            return None, "Empty report text"
        
        # Check cache (thread-safe)
        text_hash = cls._get_content_hash(report_text)
        with cls._lock:
            if text_hash in cls._response_cache:
                logger.info(f"   ⚡ Cache hit for {filename}")
                return cls._response_cache[text_hash], None
        
        # Truncate very long reports to save tokens (keep first 18000 chars)
        # Note: Larger limit needed for comprehensive medical reports
        truncated_text = report_text[:18000] if len(report_text) > 18000 else report_text
        text_chars = len(truncated_text)
        logger.debug(f"   📝 {filename}: Processing {text_chars} chars")
        
        prompt = cls.EXTRACTION_PROMPT.format(text=truncated_text)
        
        # Retry logic for transient failures
        max_retries = 2
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                response = cls._extraction_model.generate_content(prompt)
                response_text = response.text
                
                # Extract JSON from response
                json_str = cls._extract_json(response_text)
                if not json_str:
                    error_msg = f"No valid JSON in response (attempt {attempt + 1})"
                    logger.warning(f"   ⚠ {filename}: {error_msg}")
                    last_error = error_msg
                    if attempt < max_retries:
                        import time
                        time.sleep(1)  # Brief pause before retry
                        continue
                    return None, f"No valid JSON found after {max_retries + 1} attempts. Last response: {response_text[:150]}..."
                
                result = json.loads(json_str)
                
                # Cache the result (thread-safe)
                with cls._lock:
                    cls._response_cache[text_hash] = result
                
                return result, None
                
            except json.JSONDecodeError as e:
                error_msg = f"JSON parsing error: {str(e)}"
                logger.error(f"   ✗ {filename}: {error_msg}")
                return None, error_msg
            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"
                logger.warning(f"   ⚠ {filename}: Attempt {attempt + 1} failed - {last_error}")
                if attempt < max_retries:
                    import time
                    time.sleep(2)  # Wait before retry
                    continue
                logger.error(f"   ✗ {filename}: All {max_retries + 1} attempts failed")
                return None, last_error
        
        return None, last_error or "Unknown error after retries"
    
    @classmethod
    def analyze_reports_parallel(cls, reports: List[Dict], api_key: str, 
                                  max_workers: int = 3) -> List[Dict]:
        """
        Analyze multiple reports in PARALLEL for significant speedup.
        Uses ThreadPoolExecutor with controlled concurrency.
        
        Args:
            reports: List of {'name': str, 'text': str} dicts
            api_key: Gemini API key
            max_workers: Number of parallel workers (default 3 to avoid rate limits)
            
        Returns:
            List of {'name': str, 'result': Dict, 'success': bool, 'duration': float, 'error': str} dicts
        """
        import time
        
        if not cls._extraction_model and not cls.initialize(api_key):
            error_msg = "AI model initialization failed"
            logger.error(f"   ✗ {error_msg}")
            return [{'name': r['name'], 'result': None, 'success': False, 'duration': 0, 'error': error_msg} 
                    for r in reports]
        
        def analyze_single(report: Dict) -> Dict:
            """Analyze a single report and return result with timing"""
            start = time.time()
            filename = report['name']
            try:
                result, error = cls.analyze_report(report['text'], api_key, filename)
                duration = time.time() - start
                
                if result:
                    test_count = len(result.get('test_results', []))
                    logger.info(f"   ✓ AI found {test_count} tests in {duration:.2f}s for {filename}")
                
                return {
                    'name': filename,
                    'result': result,
                    'success': result is not None,
                    'error': error,
                    'duration': duration
                }
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error(f"   ✗ {filename}: Unexpected error - {error_msg}")
                return {
                    'name': filename,
                    'result': None,
                    'success': False,
                    'error': error_msg,
                    'duration': time.time() - start
                }
        
        # Process in parallel
        logger.info(f"   🚀 Starting parallel AI analysis with {max_workers} workers...")
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_report = {executor.submit(analyze_single, r): r for r in reports}
            
            for future in concurrent.futures.as_completed(future_to_report):
                result = future.result()
                results.append(result)
        
        # Sort by original order
        name_order = {r['name']: i for i, r in enumerate(reports)}
        results.sort(key=lambda x: name_order.get(x['name'], 999))
        
        # Log summary
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        if failed > 0:
            logger.warning(f"   ⚠ AI analysis: {successful}/{len(results)} successful, {failed} failed")
            for r in results:
                if not r['success']:
                    logger.error(f"      - {r['name']}: {r.get('error', 'Unknown error')}")
        
        return results
    
    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Extract JSON object from response text"""
        # Try to find JSON in code block first
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            return match.group(1)
        
        # Try to find raw JSON
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end > start:
            return text[start:end + 1]
        
        return None
    
    @classmethod
    def analyze_multiple_reports(cls, reports: List[Dict], api_key: str, 
                                  progress_callback=None) -> List[Dict]:
        """
        Analyze multiple reports with progress tracking (sequential).
        Use analyze_reports_parallel for faster processing.
        
        Args:
            reports: List of {'name': str, 'text': str} dicts
            api_key: Gemini API key
            progress_callback: Optional function(current, total, name) for progress
            
        Returns:
            List of {'name': str, 'result': Dict, 'success': bool, 'error': str} dicts
        """
        results = []
        total = len(reports)
        
        for i, report in enumerate(reports):
            if progress_callback:
                progress_callback(i + 1, total, report['name'])
            
            result, error = cls.analyze_report(report['text'], api_key, report['name'])
            results.append({
                'name': report['name'],
                'result': result,
                'success': result is not None,
                'error': error
            })
        
        return results
    
    @classmethod
    def get_chat_response(cls, df, question: str, chat_history: List[Dict], 
                          api_key: str) -> str:
        """Get AI response for chat query about the medical data"""
        if not cls._chat_model and not cls.initialize(api_key):
            return "Chat model not initialized."
        
        if df.empty:
            return "No report data available."
        
        # Prepare concise data summary (limit to 50 rows)
        cols = ['Test_Date', 'Test_Category', 'Test_Name', 'Result', 'Unit', 
                'Reference_Range', 'Status']
        existing_cols = [c for c in cols if c in df.columns]
        data_str = df[existing_cols].head(50).to_string(index=False)
        
        # Format chat history (last 5 messages)
        history_str = ""
        for entry in chat_history[-5:]:
            history_str += f"{entry['role'].capitalize()}: {entry['content']}\n"
        
        prompt = cls.CHAT_PROMPT.format(
            history=history_str,
            data=data_str,
            question=question
        )
        
        try:
            response = cls._chat_model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Chat error: {e}")
            return "Sorry, I encountered an error."
    
    @classmethod
    def clear_cache(cls):
        """Clear the response cache"""
        cls._response_cache.clear()
