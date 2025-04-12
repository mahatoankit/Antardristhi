# DataChat: Interactive Data Analytics Assistant 🤖

An intelligent data analysis tool that helps non-technical users explore and understand their data through natural language conversations.

## Features 🌟

- **Natural Language Queries**: Ask questions about your data in plain English
- **Intelligent Visualization**: Automatically generates relevant visualizations
- **Smart Data Processing**: Handles various file formats and data types
- **Interactive Interface**: User-friendly Streamlit-based UI
- **Automatic Insights**: Generates helpful insights and suggested questions
- **Data Cleaning**: Automatic handling of missing values and data type conversions

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DataChat.git
cd DataChat
```

2. Create and activate a virtual environment:
```bash
python -m venv env
# Windows
env\Scripts\activate
# Linux/Mac
source env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory and add:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

## Usage 💡

1. Start the application:
```bash
streamlit run src/main.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload your data file (supported formats: CSV, XLSX, JSON)

4. Start analyzing your data:
   - View automatic data insights
   - Ask questions in natural language
   - Explore suggested visualizations
   - Get AI-powered explanations

## Supported Features 📊

### Data Loading
- CSV files
- Excel files (XLSX)
- JSON files
- Automatic data type detection
- Missing value handling
- Duplicate removal

### Visualization Types
- Histograms
- Line charts
- Bar charts
- Scatter plots
- Pie charts
- Time series analysis

### Query Types
- Filtering data
- Aggregating values
- Trend analysis
- Comparisons
- Correlations
- Custom queries

## Technical Details 🔧

### Dependencies
- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Plotly
- Google Generative AI (Gemini)
- Other requirements listed in `requirements.txt`

### Project Structure
```
DataChat/
├── src/
│   ├── main.py          # Main application code
│   └── package_manager/ # Package management utilities
├── .env                # Environment variables
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Contributing 🤝

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact 📧

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - email@example.com

Project Link: [https://github.com/mahatoankit/antardristhi](https://github.com/yourusername/DataChat)

## Acknowledgments 🙏

- Google Generative AI for providing the Gemini API
- Streamlit for the awesome web framework
- The open-source community for various dependencies