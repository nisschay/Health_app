"""
Performance Logging Module
Tracks timing, errors, and provides analytics for the analysis pipeline
Logs to file for developer debugging (not shown to users)
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# Create logs directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Configure file logging
log_filename = os.path.join(LOG_DIR, f"analysis_{datetime.now().strftime('%Y%m%d')}.log")
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

# Console handler for development
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S'))

# Setup logger
logger = logging.getLogger('MedicalAnalyzer')
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(console_handler)


@dataclass
class PDFMetrics:
    """Metrics for a single PDF processing"""
    filename: str
    file_size_kb: float = 0
    extraction_time: float = 0
    analysis_time: float = 0
    total_time: float = 0
    pages_extracted: int = 0
    tests_found: int = 0
    is_corrupted: bool = False
    corruption_details: str = ""
    success: bool = False
    error: str = ""


@dataclass  
class SessionMetrics:
    """Metrics for an entire analysis session"""
    start_time: float = field(default_factory=time.time)
    end_time: float = 0
    pdf_metrics: List[PDFMetrics] = field(default_factory=list)
    total_pdfs: int = 0
    successful_pdfs: int = 0
    corrupted_pdfs: int = 0
    failed_pdfs: int = 0
    total_tests_extracted: int = 0
    
    @property
    def total_time(self) -> float:
        return self.end_time - self.start_time if self.end_time else time.time() - self.start_time
    
    @property
    def avg_time_per_pdf(self) -> float:
        return self.total_time / self.total_pdfs if self.total_pdfs > 0 else 0


class PerformanceTracker:
    """
    Tracks performance metrics for the analysis pipeline.
    Provides logging, timing, and error reporting.
    """
    
    def __init__(self):
        self.session = SessionMetrics()
        self.current_pdf: Optional[PDFMetrics] = None
        self._step_start: float = 0
    
    def start_session(self, total_pdfs: int):
        """Start a new analysis session"""
        self.session = SessionMetrics(total_pdfs=total_pdfs)
        logger.info(f"═══ Starting analysis session with {total_pdfs} PDF(s) ═══")
    
    def start_pdf(self, filename: str, file_size_bytes: int = 0):
        """Start processing a PDF"""
        self.current_pdf = PDFMetrics(
            filename=filename,
            file_size_kb=file_size_bytes / 1024 if file_size_bytes else 0
        )
        self._step_start = time.time()
        logger.info(f"📄 Processing: {filename}")
    
    def log_extraction_complete(self, filename: str, text_length: int, error: str = None):
        """Log PDF text extraction completion"""
        if not self.current_pdf or self.current_pdf.filename != filename:
            # Find or create the PDF metrics
            self.current_pdf = PDFMetrics(filename=filename)
        
        self.current_pdf.extraction_time = time.time() - self._step_start
        
        if error:
            self.current_pdf.is_corrupted = True
            self.current_pdf.corruption_details = error
            self.current_pdf.success = False
            self.session.corrupted_pdfs += 1
            logger.warning(f"   ⚠ {filename}: {error[:80]}")
        else:
            logger.info(f"   ✓ Extracted {text_length:,} chars in {self.current_pdf.extraction_time:.2f}s")
        
        self._step_start = time.time()
    
    def log_ai_complete(self, filename: str, tests_found: int, duration: float, error: str = None):
        """Log AI analysis completion"""
        if not self.current_pdf or self.current_pdf.filename != filename:
            self.current_pdf = PDFMetrics(filename=filename)
        
        self.current_pdf.analysis_time = duration
        self.current_pdf.tests_found = tests_found
        
        if error:
            self.current_pdf.error = error
            self.current_pdf.success = False
            self.session.failed_pdfs += 1
            logger.error(f"   ✗ AI analysis failed for {filename}: {error[:80]}")
        else:
            self.current_pdf.success = True
            self.session.successful_pdfs += 1
            self.session.total_tests_extracted += tests_found
            logger.info(f"   ✓ AI found {tests_found} tests in {duration:.2f}s")
        
        self.current_pdf.total_time = self.current_pdf.extraction_time + self.current_pdf.analysis_time
        self.session.pdf_metrics.append(self.current_pdf)
        self.current_pdf = None
    
    def end_session(self):
        """End the analysis session"""
        self.session.end_time = time.time()
        logger.info(f"═══ Session complete ═══")
        logger.info(f"   Total time: {self.session.total_time:.2f}s")
        logger.info(f"   PDFs: {self.session.successful_pdfs}/{self.session.total_pdfs} successful")
        logger.info(f"   Tests extracted: {self.session.total_tests_extracted}")
        if self.session.corrupted_pdfs > 0:
            logger.warning(f"   Corrupted PDFs: {self.session.corrupted_pdfs}")

    def log_extraction_failed(self, is_corrupted: bool = False, details: str = ""):
        """Log PDF extraction failure"""
        if self.current_pdf:
            self.current_pdf.extraction_time = time.time() - self._step_start
            self.current_pdf.is_corrupted = is_corrupted
            self.current_pdf.corruption_details = details
            self.current_pdf.success = False
            
            if is_corrupted:
                self.session.corrupted_pdfs += 1
                logger.warning(f"   ⚠ Corrupted PDF: {details[:100]}")
            else:
                self.session.failed_pdfs += 1
                logger.error(f"   ✗ Extraction failed: {details[:100]}")
    
    def log_analysis_complete(self, tests_found: int):
        """Log AI analysis completion"""
        if self.current_pdf:
            self.current_pdf.analysis_time = time.time() - self._step_start
            self.current_pdf.tests_found = tests_found
            self.current_pdf.total_time = self.current_pdf.extraction_time + self.current_pdf.analysis_time
            self.current_pdf.success = True
            self.session.successful_pdfs += 1
            self.session.total_tests_extracted += tests_found
            logger.info(f"   ✓ AI found {tests_found} tests in {self.current_pdf.analysis_time:.2f}s")
            logger.info(f"   ✓ Total: {self.current_pdf.total_time:.2f}s")
    
    def log_analysis_failed(self, error: str):
        """Log AI analysis failure"""
        if self.current_pdf:
            self.current_pdf.analysis_time = time.time() - self._step_start
            self.current_pdf.error = error
            self.current_pdf.success = False
            self.session.failed_pdfs += 1
            logger.error(f"   ✗ AI analysis failed: {error[:100]}")
    
    def finish_pdf(self):
        """Finish processing current PDF"""
        if self.current_pdf:
            self.session.pdf_metrics.append(self.current_pdf)
            self.current_pdf = None
    
    def finish_session(self):
        """Finish the analysis session"""
        self.session.end_time = time.time()
        logger.info(f"═══ Session complete ═══")
        logger.info(f"   Total time: {self.session.total_time:.2f}s")
        logger.info(f"   PDFs: {self.session.successful_pdfs}/{self.session.total_pdfs} successful")
        logger.info(f"   Tests extracted: {self.session.total_tests_extracted}")
        if self.session.corrupted_pdfs > 0:
            logger.warning(f"   Corrupted PDFs: {self.session.corrupted_pdfs}")
    
    def get_corrupted_pdfs(self) -> List[PDFMetrics]:
        """Get list of corrupted PDFs"""
        return [m for m in self.session.pdf_metrics if m.is_corrupted]
    
    def get_failed_pdfs(self) -> List[PDFMetrics]:
        """Get list of failed PDFs (excluding corrupted)"""
        return [m for m in self.session.pdf_metrics if not m.success and not m.is_corrupted]
    
    def display_metrics_ui(self):
        """Display performance metrics in Streamlit UI"""
        if not self.session.pdf_metrics:
            return
        
        st.markdown("### ⏱️ Performance Metrics")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Time", f"{self.session.total_time:.1f}s")
        with col2:
            st.metric("Avg per PDF", f"{self.session.avg_time_per_pdf:.1f}s")
        with col3:
            st.metric("Success Rate", f"{self.session.successful_pdfs}/{self.session.total_pdfs}")
        with col4:
            st.metric("Tests Found", self.session.total_tests_extracted)
        
        # Speed chart
        if len(self.session.pdf_metrics) > 0:
            import plotly.graph_objs as go
            
            fig = go.Figure()
            
            filenames = [m.filename[:20] + "..." if len(m.filename) > 20 else m.filename 
                        for m in self.session.pdf_metrics]
            extraction_times = [m.extraction_time for m in self.session.pdf_metrics]
            analysis_times = [m.analysis_time for m in self.session.pdf_metrics]
            
            fig.add_trace(go.Bar(
                name='PDF Extraction',
                x=filenames,
                y=extraction_times,
                marker_color='#3b82f6'
            ))
            
            fig.add_trace(go.Bar(
                name='AI Analysis',
                x=filenames,
                y=analysis_times,
                marker_color='#8b5cf6'
            ))
            
            fig.update_layout(
                title='Processing Time by PDF',
                xaxis_title='PDF File',
                yaxis_title='Time (seconds)',
                barmode='stack',
                height=300,
                template='plotly_dark',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Corrupted PDFs warning
        corrupted = self.get_corrupted_pdfs()
        if corrupted:
            st.warning(f"⚠️ **{len(corrupted)} Corrupted PDF(s) Detected**")
            with st.expander("View corrupted file details"):
                for pdf in corrupted:
                    st.markdown(f"""
                    - **{pdf.filename}**
                      - Issue: {pdf.corruption_details[:200] if pdf.corruption_details else 'Unknown corruption'}
                      - Some data may have been recovered
                    """)
        
        # Failed PDFs
        failed = self.get_failed_pdfs()
        if failed:
            st.error(f"❌ **{len(failed)} PDF(s) Failed to Process**")
            with st.expander("View failed file details"):
                for pdf in failed:
                    st.markdown(f"- **{pdf.filename}**: {pdf.error[:200] if pdf.error else 'Unknown error'}")


# Global tracker instance
tracker = PerformanceTracker()
