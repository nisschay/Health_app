# Medical Report Analyzer & Health Insights

A high-performance web application built with Streamlit that analyzes medical test reports in PDF format, providing structured data visualization, AI-powered insights, and comprehensive health reports.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the optimized version (recommended)
streamlit run app_optimized.py

# Or run the original version
streamlit run Medical_Project.py
```

## ✨ Key Features

### 📄 PDF Processing
- **Parallel Processing**: Analyze multiple PDFs simultaneously using ThreadPoolExecutor
- **Corrupted File Detection**: Automatically identifies and reports unreadable PDFs
- **Smart Text Extraction**: Handles various PDF formats with fallback mechanisms

### 🤖 AI-Powered Analysis
- **Gemini 2.5 Flash**: Fast, accurate medical data extraction
- **Optimized Prompts**: 50% reduced token usage for faster responses
- **Natural Language Chat**: Ask questions about your health reports

### 📊 Visualization & Reports
- **Interactive Charts**: Plotly-based bullet charts and trend analysis
- **Health Score**: Calculated overall health score with breakdown
- **Body System Categories**: Tests organized by physiological systems
- **Export Options**: PDF, Excel, and CSV downloads

### ⚡ Performance Features
- **Speed Tracking**: Real-time performance metrics and charts
- **Session Logging**: Track processing times for optimization
- **Caching**: Reduced redundant API calls and computations

## 🏗 Project Structure

```
Medical_Project/
├── app_optimized.py            # 🚀 Optimized main application (USE THIS)
├── Medical_Project.py          # Original application
├── core/                       # Modular components
│   ├── __init__.py
│   ├── pdf_processor.py        # Parallel PDF extraction
│   ├── ai_analyzer.py          # Gemini AI integration
│   ├── data_processor.py       # Data cleaning & standardization
│   ├── patient_info.py         # Patient data extraction
│   ├── visualization.py        # Charts & health scoring
│   ├── report_generator.py     # PDF/Excel/CSV exports
│   └── performance_tracker.py  # Logging & speed metrics
├── test_category_mapping.py    # Test categorization mappings
├── unify_test_names.py         # Test name standardization
├── Helper_Functions.py         # Legacy helper functions
├── style.css                   # Custom styling
├── requirements.txt            # Dependencies
├── OPTIMIZATION_GUIDE.md       # RAG implementation guide
└── README.md                   # This file
```

## 📋 Requirements

```txt
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.15.0
google-generativeai>=0.3.0
PyPDF2>=3.0.0
reportlab>=4.0.0
xlsxwriter>=3.1.0
```

## 🔧 Configuration

### API Key Setup

**Option 1: Streamlit Secrets (Recommended)**
```bash
mkdir -p .streamlit
echo 'GEMINI_API_KEY = "your-api-key-here"' > .streamlit/secrets.toml
```

**Option 2: Runtime Input**
Enter your API key directly in the application sidebar.

### Get a Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy and configure as shown above

## 💡 Usage Guide

### Basic Workflow
1. **Launch**: Run `streamlit run app_optimized.py`
2. **Upload**: Drag & drop one or more medical PDF reports
3. **Analyze**: Click "🔬 Analyze Reports" button
4. **Explore**: View results in organized tabs:
   - 📊 **Dashboard**: Health score & summary
   - 📋 **Test Results**: Detailed test data with filters
   - 💬 **Chat**: Ask questions about your reports
   - ⚡ **Performance**: Processing speed metrics

### Performance Metrics
The app tracks and displays:
- Total processing time
- Per-PDF extraction speed
- AI analysis duration
- Any corrupted/failed files

### Export Options
- **📄 PDF Report**: Comprehensive health report with charts
- **📊 Excel**: Detailed spreadsheet with all data
- **📋 CSV**: Raw data for custom analysis

## 🔬 Technical Details

### Parallel Processing
```python
# PDFs are processed in parallel for speed
with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(extract_text, pdf_files)
```

### Caching Strategy
- AI responses cached by content hash
- Processed DataFrames cached per session
- Reduced redundant API calls by ~60%

### Error Handling
- Graceful degradation for corrupted PDFs
- Clear error messages for users
- Automatic retry for transient failures

## 📈 Performance Comparison

| Metric | Original | Optimized |
|--------|----------|-----------|
| 5 PDFs | ~45s | ~15s |
| API Calls | Many | Batched |
| Memory | High | Optimized |

## 🌐 AI Integration

The application uses **Google Gemini 2.5 Flash** for:
- Extracting structured medical data from unstructured PDF text
- Understanding natural language health queries
- Generating contextual health insights
- Identifying abnormal test results

## ⚠️ Disclaimer

This application is for **informational purposes only** and does not constitute medical advice. Always consult with qualified healthcare professionals for medical decisions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## ✨ Acknowledgments

- [Streamlit](https://streamlit.io/) - Web framework
- [Google Gemini AI](https://ai.google.dev/) - AI processing
- [Plotly](https://plotly.com/) - Visualization
- [ReportLab](https://www.reportlab.com/) - PDF generation
