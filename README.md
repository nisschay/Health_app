# Medical Report Analyzer & Health Insights

A powerful web application built with Streamlit that analyzes medical test reports in PDF format, providing structured data visualization and AI-powered insights.

## 🌟 Features

- **PDF Report Analysis**: Extract medical test data from PDF reports
- **Smart Data Structuring**: Automatically standardizes test names, units, and status values
- **Interactive Visualizations**:
  - Bullet charts for single test results
  - Trend analysis for multiple test dates
  - Color-coded reference ranges
- **AI-Powered Insights**:
  - Natural language chat interface for report queries
  - Comprehensive health summaries
  - Abnormal findings detection
- **Multi-Report Support**: Analyze and consolidate data from multiple PDF reports
- **Body System Categories**: Organize tests by physiological systems
- **Data Export**: Download analyzed data in CSV format

## 🔧 Technical Stack

- **Backend**: Python
- **Frontend**: Streamlit
- **AI/ML**: Google Gemini AI
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **PDF Processing**: PyPDF2

## 📋 Requirements

```txt
streamlit
pandas
plotly
google-generativeai
PyPDF2
```

## 🚀 Setup & Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Medical_Project
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Configure Gemini API:
   - Obtain an API key from Google Cloud Console
   - Set it up in Streamlit secrets or input during runtime

4. Run the application:
```bash
streamlit run Medical_Project.py
```

## 📊 Features in Detail

### Data Processing
- Standardizes test names, units, and status values using predefined mappings
- Consolidates patient information across multiple reports
- Converts dates to standard format
- Handles various reference range formats

### Visualization Types
1. **Bullet Charts**:
   - Shows current value against reference range
   - Color-coded status indicators
   - Clear display of test name, value, and units

2. **Trend Analysis**:
   - Time series plots for multiple test dates
   - Reference range bands
   - Interactive legends

### Body System Categories
- Blood Tests
- Liver Function
- Kidney Function
- Lipid Profile
- Thyroid Profile
- And more...

## 🏗 Project Structure

```
Medical_Project/
├── Medical_Project.py          # Main application file
├── test_category_mapping.py    # Test categorization mappings
├── Backend/
│   └── Medical_Report_Analysis_(V2).ipynb  # Development notebook
└── README.md                   # This file
```

## 💡 Usage

1. Launch the application
2. Upload one or more medical PDF reports
3. Click "Analyze Reports" to process
4. Explore:
   - View patient information
   - Interact with the AI chatbot
   - Visualize test results
   - Filter by body systems
   - Download processed data

## 🌐 AI Integration

The application uses Google's Gemini AI for:
- Extracting structured data from PDF reports
- Natural language understanding
- Providing contextual responses to user queries
- Summarizing health insights


## ✨ Acknowledgments

- Streamlit for the wonderful web framework
- Google Gemini AI for powerful natural language processing
- The medical community for standardization guidelines
