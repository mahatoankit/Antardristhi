from xml.etree.ElementInclude import include
import streamlit as st
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple  # Added Tuple
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os
import requests
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import tempfile
from datetime import datetime, timedelta
from package_manager.installer import PackageInstaller

installer = PackageInstaller()

load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

async def ensure_dependencies():
    required_packages = ['pandas', 'numpy', 'plotly']
    for package in required_packages:
        await installer.install_package(package)

def load_data(uploaded_file) -> pd.DataFrame:
    """Load data from various file formats and convert to pandas DataFrame with improved cleaning."""
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    try:
        if file_extension == '.csv':
            df = pd.read_csv(uploaded_file, low_memory=False)
        elif file_extension == '.xlsx':
            df = pd.read_excel(uploaded_file)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                df.to_csv(tmp_file.name, index=False)
                st.session_state['converted_csv_path'] = tmp_file.name
        elif file_extension == '.json':
            df = pd.read_json(uploaded_file)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                df.to_csv(tmp_file.name, index=False)
                st.session_state['converted_csv_path'] = tmp_file.name
        else:
            raise ValueError("Unsupported file format. Please upload CSV, XLSX, or JSON file.")
        
        # Data Cleaning Steps
        # 1. Handle missing values
        for col in df.columns:
            missing_pct = df[col].isnull().mean()
            if missing_pct > 0.5:  # If more than 50% missing, drop the column
                df = df.drop(columns=[col])
                st.warning(f"Column '{col}' was dropped due to having {missing_pct*100:.1f}% missing values.")
            
        # 2. Convert data types intelligently
        for col in df.columns:
            # Try to convert to numeric
            try:
                if df[col].dtype == 'object':
                    # Check if it's a numeric column with some text values
                    numeric_mask = pd.to_numeric(df[col], errors='coerce').notnull()
                    if numeric_mask.mean() > 0.8:  # If more than 80% are numeric
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Try to parse as datetime using common formats with explicit format strings
                    if not numeric_mask.any():
                        date_formats = {
                            '%Y-%m-%d': r'\d{4}-\d{2}-\d{2}',
                            '%d/%m/%Y': r'\d{2}/\d{2}/\d{4}',
                            '%m/%d/%Y': r'\d{2}/\d{2}/\d{4}',
                            '%Y/%m/%d': r'\d{4}/\d{2}/\d{2}',
                            '%Y-%m-%d %H:%M:%S': r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
                            '%d-%m-%Y': r'\d{2}-\d{2}-\d{4}',
                            '%m-%d-%Y': r'\d{2}-\d{2}-\d{4}'
                        }
                        
                        for date_format, pattern in date_formats.items():
                            # Check if at least 80% of non-null values match the pattern
                            sample = df[col].dropna().astype(str)
                            if sample.str.match(pattern).mean() > 0.8:
                                try:
                                    df[col] = pd.to_datetime(df[col], format=date_format)
                                    break
                                except (ValueError, TypeError):
                                    continue
            except (ValueError, TypeError):
                continue
        
        # 3. Handle duplicates
        if df.duplicated().any():
            dup_count = df.duplicated().sum()
            df = df.drop_duplicates()
            st.warning(f"Removed {dup_count} duplicate rows from the dataset.")
        
        # 4. Clean column names
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error loading file: {str(e)}")

def get_basic_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform basic exploratory data analysis on the dataset."""
    analysis = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'columns': df.columns.tolist(),
        'data_types': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'sample_records': df.head().to_dict('records'),
        'numeric_summary': {}
    }
    
    # Calculate statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        analysis['numeric_summary'][col] = {
            'mean': float(df[col].mean()),
            'median': float(df[col].median()),
            'min': float(df[col].min()),
            'max': float(df[col].max())
        }
    
    return analysis

def generate_content_with_gemini(prompt: str) -> str:
    """Generate content using Gemini 2.0 Flash API directly."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}"
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    data = {
        "contents": [{
            "parts":[{"text": prompt}]
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        if 'candidates' in result and len(result['candidates']) > 0:
            return result['candidates'][0]['content']['parts'][0]['text']
        return "No response generated"
    except Exception as e:
        st.error(f"Error calling Gemini API: {str(e)}")
        return str(e)

def generate_suggested_questions(df: pd.DataFrame) -> List[str]:
    """Generate relevant questions based on the dataset structure."""
    columns = df.columns.tolist()
    
    prompt = f"Given a dataset with columns: {', '.join(columns)}, "
    prompt += "suggest 3-5 natural language questions that a non-technical person might want to ask about this data. "
    prompt += "Make questions simple and conversational. Return each question on a new line."
    
    try:
        response = generate_content_with_gemini(prompt)
        questions = [q.strip() for q in response.split('\n') if q.strip()]
        return questions[:5]
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return ["What is the total number of records?",
                "What are the main trends in this data?",
                "Can you show me a summary of the data?"]

def extract_query_intent(question: str) -> Dict[str, Any]:
    """Extract the intent and parameters from a natural language question."""
    prompt = f"""Analyze this question: "{question}"
    Extract the query intent and parameters in this JSON format:
    {{
        "intent": "one of: filter, aggregate, trend, comparison, correlation",
        "parameters": {{
            "columns": ["column names involved"],
            "filters": {{"column": "value"}},
            "aggregation": "sum/mean/count/etc if applicable",
            "groupby": ["columns to group by if applicable"]
        }}
    }}
    Return ONLY the JSON, no other text."""
    
    try:
        response = generate_content_with_gemini(prompt)
        # Clean and parse the JSON response
        response = response.strip()
        if response.startswith('```') and response.endswith('```'):
            response = response[response.find('{'):response.rfind('}')+1]
        return json.loads(response)
    except Exception as e:
        st.error(f"Error extracting query intent: {str(e)}")
        return {
            "intent": "filter",
            "parameters": {
                "columns": [],
                "filters": {},
                "aggregation": "count",
                "groupby": []
            }
        }

def execute_data_query(df: pd.DataFrame, query_intent: Dict[str, Any]) -> Tuple[pd.DataFrame, str, Dict]:
    """Execute a data query based on the extracted intent and parameters."""
    try:
        result_df = df.copy()
        viz_config = None
        explanation = ""
        
        # Apply filters
        if query_intent["parameters"]["filters"]:
            for col, value in query_intent["parameters"]["filters"].items():
                if col in df.columns:
                    if isinstance(value, (int, float)):
                        result_df = result_df[result_df[col] == value]
                    else:
                        result_df = result_df[result_df[col].astype(str).str.contains(str(value), case=False)]
        
        # Apply grouping and aggregation
        if query_intent["parameters"]["groupby"] and query_intent["parameters"]["aggregation"]:
            agg_cols = [col for col in query_intent["parameters"]["columns"] 
                       if col not in query_intent["parameters"]["groupby"]]
            
            if agg_cols:
                agg_dict = {col: query_intent["parameters"]["aggregation"] for col in agg_cols}
                result_df = result_df.groupby(query_intent["parameters"]["groupby"]).agg(agg_dict).reset_index()
        
        # Generate visualization config based on intent
        if query_intent["intent"] == "trend":
            viz_config = {
                "title": "Trend Analysis",
                "plot_type": "line",
                "columns": query_intent["parameters"]["columns"][:2],
                "explanation": "Shows the trend over time"
            }
        elif query_intent["intent"] == "comparison":
            viz_config = {
                "title": "Comparison Analysis",
                "plot_type": "bar",
                "columns": query_intent["parameters"]["columns"][:2],
                "explanation": "Compares values across categories"
            }
        elif query_intent["intent"] == "correlation":
            viz_config = {
                "title": "Correlation Analysis",
                "plot_type": "scatter",
                "columns": query_intent["parameters"]["columns"][:2],
                "explanation": "Shows relationship between variables"
            }
        
        # Generate natural language explanation
        if query_intent["parameters"]["filters"]:
            explanation += f"Filtered data where {', '.join([f'{k}={v}' for k,v in query_intent['parameters']['filters'].items()])}. "
        if query_intent["parameters"]["groupby"]:
            explanation += f"Grouped by {', '.join(query_intent['parameters']['groupby'])}. "
        if query_intent["parameters"]["aggregation"]:
            explanation += f"Calculated {query_intent['parameters']['aggregation']} of {', '.join(query_intent['parameters']['columns'])}."
        
        return result_df, explanation, viz_config
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return df.head(), "Could not process query completely.", None

def handle_sales_query(df: pd.DataFrame, query: str) -> Tuple[pd.DataFrame, str, Dict]:
    """Handle specific sales-related queries with direct data manipulation."""
    # Clean the query
    query = query.lower().strip()
    
    # Define query patterns and their handlers
    if "sales order" in query and ("belong" in query or "related" in query or "associated" in query):
        # Extract any numbers from the query as potential order IDs
        import re
        numbers = re.findall(r'\d+', query)
        
        if numbers:
            order_id = numbers[0]
            # Look for columns that might contain order information
            order_cols = [col for col in df.columns if any(term in col.lower() 
                         for term in ['order', 'sales', 'transaction'])]
            
            if order_cols:
                result_df = df[df[order_cols[0]] == order_id]
                explanation = f"Found {len(result_df)} order lines for sales order {order_id}"
                
                viz_config = {
                    "title": f"Order Details for Sales Order {order_id}",
                    "plot_type": "bar",
                    "columns": [col for col in df.columns if col.lower() != order_cols[0]][:2],
                    "explanation": "Shows the breakdown of the sales order"
                }
                
                return result_df, explanation, viz_config
    
    # If no specific pattern matches, return None to fall back to general query processing
    return None, None, None

def generate_natural_language_answer(df: pd.DataFrame, result_df: pd.DataFrame, query: str) -> str:
    """Generate a natural language answer for the query based on the results."""
    prompt = f"""Given this query: "{query}"
    And these results from the data:
    - Total rows in result: {len(result_df)}
    - Columns available: {', '.join(result_df.columns)}
    - Sample data: {result_df.head(3).to_dict('records')}
    
    Generate a 2-3 sentence natural language answer that a non-technical person would understand.
    Focus on insights and business value rather than technical details.
    Keep it simple and conversational."""
    
    try:
        answer = generate_content_with_gemini(prompt)
        return answer.strip()
    except Exception as e:
        return f"Found {len(result_df)} matching records. {str(e)}"

def process_question(df: pd.DataFrame, question: str) -> None:
    """Process a question about the data and display results."""
    try:
        # First try specialized query handlers
        result_df, explanation, viz_config = handle_sales_query(df, question)
        
        # If no specialized handler matched, use general query processing
        if result_df is None:
            query_intent = extract_query_intent(question)
            result_df, explanation, viz_config = execute_data_query(df, query_intent)
        
        # Generate natural language answer
        answer = generate_natural_language_answer(df, result_df, question)
        
        # Display results
        st.write("💡 **Answer:**", answer)
        
        if len(result_df) > 0:
            with st.expander("View Detailed Results"):
                st.write("📊 **Data:**")
                st.dataframe(result_df)
                st.write("🔍 **Technical Details:**", explanation)
            
            # Generate and display visualization if applicable
            if viz_config:
                st.write("📈 **Visualization:**")
                code = generate_visualization_code(viz_config, result_df)
                fig = execute_viz_code(code, result_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Generate additional insights if there are enough rows
            if len(result_df) > 1:
                insight_prompt = f"""Given this result of {len(result_df)} rows with columns {', '.join(result_df.columns)},
                provide 2-3 key insights about the data in bullet points. Keep it simple and non-technical."""
                insights = generate_content_with_gemini(insight_prompt)
                with st.expander("Additional Insights"):
                    st.write(insights)
        else:
            st.warning("No results found matching your query.")
            
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        st.write("I apologize, but I couldn't process that question. Could you rephrase it?")

import pandas as pd
import numpy as np
from typing import List, Dict

def generate_fallback_visualizations(df: pd.DataFrame) -> List[Dict]:
    """Generate basic visualization suggestions when the API fails."""
    suggestions = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include(['datetime64[ns]'])).columns.tolist()  # ✅ Fixed parenthesis and syntax

    # Try to create meaningful visualizations based on available column types
    if numeric_cols:
        # Add distribution plot for first numeric column
        suggestions.append({
            'title': f'Distribution of {numeric_cols[0]}',
            'plot_type': 'histogram',
            'columns': [numeric_cols[0]],
            'explanation': f'Shows how {numeric_cols[0]} values are distributed'
        })

    return suggestions

def get_sample_data(df: pd.DataFrame, max_rows: int = 1000) -> pd.DataFrame:
    """Get a representative sample of the dataframe for analysis."""
    if len(df) > max_rows:
        return df.sample(n=max_rows, random_state=42)
    return df.copy()

def analyze_data_for_viz(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze dataset to determine suitable visualizations with error handling."""
    try:
        # Get a sample of data for analysis
        sample_df = get_sample_data(df)
        
        analysis = {
            'row_count': len(df),
            'col_count': len(df.columns),
            'column_info': {}
        }
        
        # Analyze each column individually with error handling
        for col in df.columns:
            try:
                col_info = {
                    'name': col,
                    'dtype': str(df[col].dtype),
                    'unique_count': df[col].nunique(),
                    'missing_count': df[col].isnull().sum()
                }
                
                # Identify column type
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_info['type'] = 'numeric'
                    if df[col].nunique() < 20:  # Discrete numeric
                        col_info['subtype'] = 'discrete'
                    else:
                        col_info['subtype'] = 'continuous'
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    col_info['type'] = 'datetime'
                elif isinstance(df[col].dtype, pd.CategoricalDtype) or (df[col].dtype == 'object' and df[col].nunique() < df.shape[0] * 0.5):
                    col_info['type'] = 'categorical'
                else:
                    col_info['type'] = 'other'
                
                analysis['column_info'][col] = col_info
            except Exception as e:
                st.warning(f"Could not analyze column {col}: {str(e)}")
                continue
        
        return analysis
    except Exception as e:
        st.error(f"Error in data analysis: {str(e)}")
        return {'row_count': len(df), 'col_count': len(df.columns), 'column_info': {}}

def suggest_visualizations(df: pd.DataFrame) -> List[Dict]:
    """Generate visualization suggestions with improved error handling."""
    try:
        # Get data analysis
        analysis = analyze_data_for_viz(df)
        if not analysis['column_info']:
            return generate_fallback_visualizations(df)
        
        suggestions = []
        
        # 1. Find numeric columns for distribution plots
        numeric_cols = [col for col, info in analysis['column_info'].items() 
                       if info['type'] == 'numeric']
        
        # 2. Find datetime columns for trend analysis
        datetime_cols = [col for col, info in analysis['column_info'].items() 
                        if info['type'] == 'datetime']
        
        # 3. Find categorical columns
        categorical_cols = [col for col, info in analysis['column_info'].items() 
                          if info['type'] == 'categorical']
        
        # Generate visualization suggestions based on available column types
        if numeric_cols:
            # Add distribution plot for first numeric column
            suggestions.append({
                'title': f'Distribution of {numeric_cols[0]}',
                'plot_type': 'histogram',
                'columns': [numeric_cols[0]],
                'explanation': f'Shows how {numeric_cols[0]} values are distributed'
            })
            
            # If we have datetime and numeric, add trend plot
            if datetime_cols:
                suggestions.append({
                    'title': f'{numeric_cols[0]} Over Time',
                    'plot_type': 'line',
                    'columns': [datetime_cols[0], numeric_cols[0]],
                    'explanation': f'Shows how {numeric_cols[0]} changes over time'
                })
            
            # If we have categorical and numeric, add comparison plot
            elif categorical_cols:
                suggestions.append({
                    'title': f'{numeric_cols[0]} by {categorical_cols[0]}',
                    'plot_type': 'bar',
                    'columns': [categorical_cols[0], numeric_cols[0]],
                    'explanation': f'Compares {numeric_cols[0]} across different {categorical_cols[0]} categories'
                })
            
            # If we have multiple numeric columns, add correlation plot
            elif len(numeric_cols) > 1:
                suggestions.append({
                    'title': f'Relationship: {numeric_cols[0]} vs {numeric_cols[1]}',
                    'plot_type': 'scatter',
                    'columns': [numeric_cols[0], numeric_cols[1]],
                    'explanation': f'Shows relationship between {numeric_cols[0]} and {numeric_cols[1]}'
                })
        
        # If we couldn't generate enough suggestions, add some safe fallbacks
        while len(suggestions) < 3 and len(numeric_cols) > len(suggestions):
            next_col = numeric_cols[len(suggestions)]
            suggestions.append({
                'title': f'Distribution of {next_col}',
                'plot_type': 'histogram',
                'columns': [next_col],
                'explanation': f'Shows how {next_col} values are distributed'
            })
        
        return suggestions if suggestions else generate_fallback_visualizations(df)
        
    except Exception as e:
        st.error(f"Error generating visualization suggestions: {str(e)}")
        return generate_fallback_visualizations(df)

def generate_visualization_code(viz_config: Dict, df: pd.DataFrame) -> str:
    """Generate Python code for the visualization based on configuration."""
    try:
        plot_type = viz_config['plot_type'].lower()
        columns = viz_config['columns']
        title = viz_config['title']
        
        code = f"""fig = None  # Initialize figure
# Clean data for visualization
df_viz = df.copy()
"""
        
        # Add data cleaning for numeric columns if needed
        if plot_type in ['scatter', 'line', 'histogram']:
            code += r"""
# Convert numeric columns and handle any currency strings
for col in df_viz.columns:
    if df_viz[col].dtype == 'object':
        try:
            # Remove currency symbols and commas using raw string
            df_viz[col] = df_viz[col].replace(r'[$,]', '', regex=True)
            df_viz[col] = pd.to_numeric(df_viz[col], errors='coerce')
        except:
            pass
"""

        # Generate plot based on type
        if plot_type == 'histogram':
            code += f"""
fig = px.histogram(df_viz, x="{columns[0]}", title="{title}")
fig.update_layout(
    title_x=0.5,
    xaxis_title="{columns[0]}",
    yaxis_title="Count",
    showlegend=True,
    template='plotly_white'
)"""

        elif plot_type == 'bar':
            if len(columns) >= 2:
                code += f"""
# Group and aggregate data
grouped_df = df_viz.groupby("{columns[0]}")["{columns[1]}"].agg(['mean', 'count']).reset_index()
fig = px.bar(grouped_df, x="{columns[0]}", y=('mean'), 
             title="{title}",
             labels={{"{columns[0]}": "{columns[0]}", "mean": "Average {columns[1]}"}}
)
fig.update_layout(
    title_x=0.5,
    template='plotly_white',
    showlegend=True
)"""
            else:
                code += f"""
value_counts = df_viz["{columns[0]}"].value_counts().reset_index()
fig = px.bar(value_counts, x="index", y="{columns[0]}", title="{title}")
fig.update_layout(
    title_x=0.5,
    xaxis_title="{columns[0]}",
    yaxis_title="Count",
    template='plotly_white',
    showlegend=True
)"""

        elif plot_type == 'line':
            code += f"""
# Sort by date/time if it's a datetime column
if pd.api.types.is_datetime64_any_dtype(df_viz["{columns[0]}"]):
    df_viz = df_viz.sort_values("{columns[0]}")

fig = px.line(df_viz, x="{columns[0]}", y="{columns[1]}", title="{title}")
fig.update_layout(
    title_x=0.5,
    xaxis_title="{columns[0]}",
    yaxis_title="{columns[1]}",
    template='plotly_white',
    showlegend=True
)"""

        elif plot_type == 'scatter':
            code += f"""
fig = px.scatter(df_viz, x="{columns[0]}", y="{columns[1]}", 
                title="{title}",
                trendline="ols")  # Add trend line
fig.update_layout(
    title_x=0.5,
    template='plotly_white',
    showlegend=True
)"""

        elif plot_type == 'pie':
            code += f"""
pie_data = df_viz.groupby("{columns[0]}")["{columns[1]}"].sum().reset_index()
fig = px.pie(pie_data, names="{columns[0]}", values="{columns[1]}", title="{title}")
fig.update_layout(
    title_x=0.5,
    template='plotly_white',
    showlegend=True
)"""

        return code

    except Exception as e:
        st.error(f"Error generating visualization code: {str(e)}")
        return ""

def prepare_data_for_viz(df: pd.DataFrame, columns: List[str], max_rows: int = 1000) -> pd.DataFrame:
    """Prepare data for visualization with sampling and type validation."""
    try:
        # Get a sample if the dataset is large
        df_viz = get_sample_data(df, max_rows)
        
        # Validate and clean requested columns
        for col in columns:
            if col not in df_viz.columns:
                raise ValueError(f"Column {col} not found in dataset")
            
            # Handle different data types
            if pd.api.types.is_numeric_dtype(df_viz[col]):
                # Clean numeric data
                df_viz[col] = pd.to_numeric(df_viz[col], errors='coerce')
            elif pd.api.types.is_datetime64_any_dtype(df_viz[col]):
                # Ensure datetime format
                df_viz[col] = pd.to_datetime(df_viz[col], errors='coerce')
            else:
                # Convert to string for categorical data
                df_viz[col] = df_viz[col].astype(str)
        
        return df_viz
    except Exception as e:
        st.error(f"Error preparing data for visualization: {str(e)}")
        return df.head()  # Return a small subset as fallback

def execute_viz_code(code: str, df: pd.DataFrame) -> go.Figure:
    """Execute visualization code with improved error handling and data preparation."""
    try:
        # Get the columns mentioned in the code
        import re
        columns = list(set(re.findall(r'df\[[\'"](.*?)[\'"]\]', code)))
        
        # Prepare data for visualization
        df_viz = prepare_data_for_viz(df, columns)
        
        # Create namespace with required libraries
        namespace = {
            'pd': pd,
            'px': px,
            'go': go,
            'np': np,
            'df': df_viz,
            'df_viz': df_viz
        }
        
        # Execute the code
        exec(code, namespace)
        
        # Get the figure from namespace
        fig = namespace.get('fig')
        
        if fig is None:
            raise ValueError("Visualization code did not generate a figure")
        
        # Update layout with consistent styling
        fig.update_layout(
            font_family="Arial",
            title_font_size=20,
            template="plotly_white",
            height=600  # Set a reasonable default height
        )
        
        return fig
    except Exception as e:
        st.error(f"Error executing visualization code: {str(e)}")
        return None

def main():
    st.title("🤖 Interactive Data Analytics Assistant")
    st.write("Upload your data file and start analyzing it instantly!")

    uploaded_file = st.file_uploader("Choose a file (CSV, XLSX, or JSON)", 
                                   type=['csv', 'xlsx', 'json'])

    if uploaded_file is not None:
        try:
            # Load the data
            df = load_data(uploaded_file)
            
            # Convert object and complex columns to string for Arrow compatibility
            df = df.copy()
            for col in df.columns:
                if df[col].dtype == 'object' or isinstance(df[col].dtype, pd.DatetimeTZDtype):
                    df[col] = df[col].astype(str)
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(df[col].dtype, pd.CategoricalDtype):  # Updated to use isinstance instead of is_categorical_dtype
                    df[col] = df[col].astype(str)
                    
            # Store the dataframe in session state
            st.session_state['df'] = df
            
            # Perform basic analysis
            analysis = get_basic_analysis(df)
            
            # Display basic information
            st.subheader("📊 Dataset Overview")
            st.write(f"Number of rows: {analysis['row_count']}")
            st.write(f"Number of columns: {analysis['column_count']}")
            
            # Display sample data
            st.subheader("👀 Sample Data")
            st.dataframe(df.head())
            
            # Display column information
            st.subheader("📋 Column Information")
            col_info = pd.DataFrame({
                'Data Type': df.dtypes.astype(str),  # Convert dtype objects to strings
                'Missing Values': df.isnull().sum()
            })
            st.dataframe(col_info.astype(str))  # Convert all values to strings
            
            # Display numeric summary if available
            if analysis['numeric_summary']:
                st.subheader("📈 Numeric Column Statistics")
                for col, stats in analysis['numeric_summary'].items():
                    st.write(f"**{col}**")
                    st.write(f"Mean: {stats['mean']:.2f}")
                    st.write(f"Median: {stats['median']:.2f}")
                    st.write(f"Range: {stats['min']:.2f} to {stats['max']:.2f}")
            
            # Generate and display suggested visualizations
            st.subheader("📊 Recommended Visualizations")
            if 'viz_suggestions' not in st.session_state:
                st.session_state['viz_suggestions'] = suggest_visualizations(df)
            
            # Display visualization suggestions as expandable sections
            for i, viz in enumerate(st.session_state['viz_suggestions'], 1):
                with st.expander(f"📈 {viz['title']}", expanded=i==1):
                    st.write(f"**Insight**: {viz['explanation']}")
                    if st.button(f"Generate Visualization {i}", key=f"viz_{i}"):
                        with st.spinner('Generating visualization...'):
                            code = generate_visualization_code(viz, df)
                            fig = execute_viz_code(code, df)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
            
            # Generate and display suggested questions
            st.subheader("❓ Suggested Questions")
            if 'suggested_questions' not in st.session_state:
                st.session_state['suggested_questions'] = generate_suggested_questions(df)
            
            # Display questions as buttons
            for i, question in enumerate(st.session_state['suggested_questions'], 1):
                if st.button(f"Q{i}: {question}", key=f"q_{i}"):
                    with st.spinner('Analyzing your question...'):
                        process_question(df, question)
            
            # Custom question input
            st.subheader("🔍 Ask Your Own Question")
            custom_question = st.text_input("What would you like to know about your data?")
            
            if custom_question:
                with st.spinner('Analyzing your question...'):
                    process_question(df, custom_question)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()