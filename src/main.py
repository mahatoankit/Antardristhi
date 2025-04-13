from xml.etree.ElementInclude import include
import streamlit as st
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional  # Added Tuple
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
import re
import json
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg for non-interactive mode
import matplotlib.pyplot as plt

installer = PackageInstaller()

load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

async def ensure_dependencies():
    required_packages = ['pandas', 'numpy', 'plotly']
    for package in required_packages:
        await installer.install_package(package)

def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns using label and one-hot encoding.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with encoded categorical columns
    """
    try:
        df_encoded = df.copy()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            unique_count = df[col].nunique()
            missing_count = df[col].isnull().sum()
            
            # Skip columns with too many unique values or too many missing values
            if unique_count > len(df) * 0.5 or missing_count > len(df) * 0.5:
                continue
                
            if unique_count <= 2:  # Binary categorical variables
                # Use label encoding for binary
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df_encoded[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                
            elif unique_count <= 10:  # Categorical with few unique values
                # Use one-hot encoding for categorical with few unique values
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                
            else:  # Categorical with many unique values
                # Use label encoding for categorical with many unique values
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df_encoded[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
        
        return df_encoded
        
    except Exception as e:
        st.error(f"Error encoding categorical columns: {str(e)}")
        return df

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
        
        # Encode categorical columns
        df = encode_categorical_columns(df)
        
        # Store original column names for reference
        categorical_mappings = {}
        for col in df.columns:
            if col.endswith('_encoded') or '_' in col:
                original_col = col.split('_')[0]
                if original_col not in categorical_mappings:
                    categorical_mappings[original_col] = []
                categorical_mappings[original_col].append(col)
        
        # Store mappings in session state for later use
        st.session_state['categorical_mappings'] = categorical_mappings
        
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
    Keep it simple and conversational.
    Generate Python code to statistically analyze the relationship between two variables in a Pandas DataFrame named `df`.

Guidelines for code generation:

1. ✅ Use ONLY `pandas` and `matplotlib.pyplot`.
2. ✅ Check that required columns exist in `df.columns` before using them.
3. ✅ Drop NaN values from relevant columns before analysis.
4. ✅ If valid, group the data appropriately and compute a statistical summary (e.g., mean).
5. ✅ Plot using `matplotlib` with proper labels, `plt.tight_layout()`, and `plt.show()`.
6. ✅ Optionally calculate and print the Pearson correlation between the two variables.
7. ❌ Do NOT assume the column names—use only what is passed or checked.
8. ❌ Do NOT include file reading/writing, markdown explanations, or commentary.
9. ✅ Return a complete, ready-to-execute code block in Python.

Example request: "Does it seem like having close family relationships helps students get better grades?"
(Assume this maps to columns `'famrel'` and `'g3'`.)

Expected code output:
- Clean
- Executable
- Self-contained
- Column-safe
- Uses `pandas` for computation
- Uses `matplotlib.pyplot` for visualization

    """
    
    try:
        answer = generate_content_with_gemini(prompt)
        return answer.strip()
    except Exception as e:
        return f"Found {len(result_df)} matching records. {str(e)}"

def execute_analysis_code(df: pd.DataFrame, generated_code: str) -> Optional[plt.Figure]:
    """
    Execute generated analysis code and return the matplotlib figure.
    
    Args:
        df: Input DataFrame
        generated_code: Generated Python code string
        
    Returns:
        Optional[plt.Figure]: Generated plot or None if error occurs
    """
    try:
        # Create a clean namespace with only required imports
        namespace = {
            'pd': pd,
            'plt': plt,
            'df': df,
            'np': np
        }
        
        # Execute the code in isolated namespace
        exec(generated_code, namespace)
        
        # Get the current figure from matplotlib
        fig = plt.gcf()
        
        return fig
    except Exception as e:
        st.error(f"Error executing analysis code: {str(e)}")
        return None

def process_question(df: pd.DataFrame, question: str) -> None:
    """Process a question about the data and display results."""
    try:
        # Generate answer and code
        answer = generate_natural_language_answer(df, df, question)
        
        # Display the natural language answer
        st.write("💡 **Answer:**", answer)
        
        # Extract code block from the answer
        code_match = re.search(r'```python(.*?)```', answer, re.DOTALL)
        
        if code_match:
            analysis_code = code_match.group(1).strip()
            
            # Execute the code and display plot
            with st.spinner('Generating visualization...'):
                # Clear any existing plots
                plt.clf()
                
                # Create namespace with required imports
                namespace = {
                    'pd': pd,
                    'plt': plt,
                    'df': df,
                    'np': np
                }
                
                # Execute the code in isolated namespace
                exec(analysis_code, namespace)
                
                # Ensure proper layout
                plt.tight_layout()
                
                # Display the plot
                st.pyplot(plt.gcf())
                
                # Clean up
                plt.close()
                
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
    """Prepare data for visualization with encoding handling."""
    try:
        df_viz = get_sample_data(df, max_rows)
        
        # Check if we're dealing with encoded columns
        if 'categorical_mappings' in st.session_state:
            mappings = st.session_state['categorical_mappings']
            
            # Replace encoded columns with their numeric versions
            for col in columns:
                original_col = col.split('_')[0]
                if original_col in mappings:
                    encoded_cols = mappings[original_col]
                    # Use the first encoded version if available
                    if encoded_cols:
                        df_viz[col] = df_viz[encoded_cols[0]]
        
        return df_viz
        
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return df.head()

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

import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple

def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate if required columns exist in the DataFrame.
    
    Args:
        df: Input DataFrame
        required_columns: List of column names to validate
        
    Returns:
        bool: True if all columns exist, False otherwise
    """
    valid_columns = df.columns.tolist()
    missing_columns = [col for col in required_columns if col not in valid_columns]
    
    if missing_columns:
        print(f"Missing required columns: {', '.join(missing_columns)}")
        return False
    return True

def plot_distribution(df: pd.DataFrame, column: str) -> Optional[plt.Figure]:
    """
    Create histogram for numeric column distribution.
    
    Args:
        df: Input DataFrame
        column: Column name to plot
        
    Returns:
        matplotlib.Figure or None if error occurs
    """
    if not validate_columns(df, [column]):
        return None
        
    try:
        plt.figure(figsize=(10, 6))
        df[column].dropna().hist(bins=30)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.tight_layout()
        return plt.gcf()
    except Exception as e:
        print(f"Error plotting distribution: {str(e)}")
        return None

def plot_time_series(df: pd.DataFrame, time_column: str, value_column: str) -> Optional[plt.Figure]:
    """
    Create line plot for time series data.
    
    Args:
        df: Input DataFrame
        time_column: Column containing time data
        value_column: Column containing values to plot
        
    Returns:
        matplotlib.Figure or None if error occurs
    """
    if not validate_columns(df, [time_column, value_column]):
        return None
        
    try:
        # Ensure data is sorted by time
        plot_df = df.sort_values(time_column).dropna(subset=[time_column, value_column])
        
        plt.figure(figsize=(12, 6))
        plt.plot(plot_df[time_column], plot_df[value_column])
        plt.title(f'{value_column} Over Time')
        plt.xlabel(time_column)
        plt.ylabel(value_column)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()
    except Exception as e:
        print(f"Error plotting time series: {str(e)}")
        return None

def plot_bar_comparison(df: pd.DataFrame, category_column: str, value_column: str) -> Optional[plt.Figure]:
    """
    Create bar plot comparing values across categories.
    
    Args:
        df: Input DataFrame
        category_column: Column containing categories
        value_column: Column containing values to compare
        
    Returns:
        matplotlib.Figure or None if error occurs
    """
    if not validate_columns(df, [category_column, value_column]):
        return None
        
    try:
        # Group and aggregate data
        agg_df = df.groupby(category_column)[value_column].mean().dropna()
        
        plt.figure(figsize=(10, 6))
        agg_df.plot(kind='bar')
        plt.title(f'Average {value_column} by {category_column}')
        plt.xlabel(category_column)
        plt.ylabel(f'Average {value_column}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()
    except Exception as e:
        print(f"Error plotting bar comparison: {str(e)}")
        return None

def plot_scatter_correlation(df: pd.DataFrame, x_column: str, y_column: str) -> Optional[plt.Figure]:
    """
    Create scatter plot showing correlation between two numeric columns.
    
    Args:
        df: Input DataFrame
        x_column: Column for x-axis
        y_column: Column for y-axis
        
    Returns:
        matplotlib.Figure or None if error occurs
    """
    if not validate_columns(df, [x_column, y_column]):
        return None
        
    try:
        # Remove rows with missing values
        plot_df = df.dropna(subset=[x_column, y_column])
        
        plt.figure(figsize=(10, 6))
        plt.scatter(plot_df[x_column], plot_df[y_column], alpha=0.5)
        plt.title(f'Correlation: {x_column} vs {y_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.tight_layout()
        return plt.gcf()
    except Exception as e:
        print(f"Error plotting scatter correlation: {str(e)}")
        return None

def generate_visualization(df: pd.DataFrame, viz_config: Dict) -> Optional[plt.Figure]:
    """
    Generate visualization based on configuration.
    
    Args:
        df: Input DataFrame
        viz_config: Dictionary containing visualization parameters
        
    Returns:
        matplotlib.Figure or None if error occurs
    """
    try:
        plot_type = viz_config['plot_type'].lower()
        columns = viz_config['columns']
        
        if not validate_columns(df, columns):
            return None
            
        if plot_type == 'histogram':
            return plot_distribution(df, columns[0])
        elif plot_type == 'line':
            return plot_time_series(df, columns[0], columns[1])
        elif plot_type == 'bar':
            return plot_bar_comparison(df, columns[0], columns[1])
        elif plot_type == 'scatter':
            return plot_scatter_correlation(df, columns[0], columns[1])
        else:
            print(f"Unsupported plot type: {plot_type}")
            return None
            
    except Exception as e:
        print(f"Error generating visualization: {str(e)}")
        return None

def generate_generic_visualizations(df: pd.DataFrame) -> List[Dict]:
    """Generate visualization suggestions including encoded columns."""
    visualizations = []
    
    try:
        # Get encoded columns
        encoded_cols = [col for col in df.columns if col.endswith('_encoded')]
        numeric_cols = df.select_dtypes([np.number]).columns.tolist()
        
        # Add visualizations for encoded categorical columns
        for col in encoded_cols[:2]:
            visualizations.append({
                'title': f'Distribution of {col.replace("_encoded", "")}',
                'plot_type': 'histogram',
                'columns': [col],
                'explanation': f'Shows distribution of encoded {col.replace("_encoded", "")}'
            })
            
            # Add correlation plots between encoded categories and numeric columns
            for num_col in numeric_cols:
                if num_col not in encoded_cols:
                    visualizations.append({
                        'title': f'{col.replace("_encoded", "")} vs {num_col}',
                        'plot_type': 'scatter',
                        'columns': [col, num_col],
                        'explanation': f'Shows relationship between {col.replace("_encoded", "")} and {num_col}'
                    })
        
        # ... rest of existing visualization code ...
        
    except Exception as e:
        st.error(f"Error generating visualizations: {str(e)}")
        return []
    
    return visualizations

def plot_generic_visualization(df: pd.DataFrame, viz_config: Dict) -> Optional[plt.Figure]:
    """
    Create a generic visualization based on configuration.
    
    Args:
        df: Input DataFrame
        viz_config: Dictionary containing visualization configuration
        
    Returns:
        Optional[plt.Figure]: Generated matplotlib figure or None if error occurs
    """
    try:
        # Clear any existing plots
        plt.close('all')
        
        # Create new figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        plot_type = viz_config['plot_type'].lower()
        columns = viz_config['columns']
        title = viz_config['title']
        
        # Clean and prepare data
        plot_df = df[columns].copy()
        plot_df = plot_df.dropna()
        
        if len(plot_df) == 0:
            st.warning("No valid data to plot after cleaning")
            return None
        
        if plot_type == 'histogram':
            if pd.api.types.is_numeric_dtype(plot_df[columns[0]]):
                ax.hist(plot_df[columns[0]], bins=30, edgecolor='black')
                ax.set_xlabel(columns[0])
                ax.set_ylabel('Frequency')
            else:
                # Fall back to bar plot for categorical data
                value_counts = plot_df[columns[0]].value_counts()[:10]
                value_counts.plot(kind='bar', ax=ax)
                ax.set_xlabel(columns[0])
                ax.set_ylabel('Count')
                
        elif plot_type == 'bar':
            value_counts = plot_df[columns[0]].value_counts()[:10]
            value_counts.plot(kind='bar', ax=ax)
            ax.set_xlabel(columns[0])
            ax.set_ylabel('Count')
            
        elif plot_type == 'line':
            if pd.api.types.is_datetime64_any_dtype(plot_df[columns[0]]):
                plot_df = plot_df.sort_values(columns[0])
            ax.plot(plot_df[columns[0]], plot_df[columns[1]])
            ax.set_xlabel(columns[0])
            ax.set_ylabel(columns[1])
            plt.xticks(rotation=45)
            
        elif plot_type == 'scatter':
            if all(pd.api.types.is_numeric_dtype(plot_df[col]) for col in columns):
                ax.scatter(plot_df[columns[0]], plot_df[columns[1]], alpha=0.5)
                ax.set_xlabel(columns[0])
                ax.set_ylabel(columns[1])
                
                # Add correlation coefficient
                corr = plot_df[columns[0]].corr(plot_df[columns[1]])
                ax.text(0.05, 0.95, f'Correlation: {corr:.2f}', 
                       transform=ax.transAxes, 
                       bbox=dict(facecolor='white', alpha=0.8))
            else:
                st.warning("Scatter plot requires numeric columns")
                return None
        
        ax.set_title(title)
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
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
            visualizations = generate_generic_visualizations(df)

            if visualizations:
                for i, viz in enumerate(visualizations, 1):
                    with st.expander(f"📈 {viz['title']}", expanded=i==1):
                        st.write(f"**Insight**: {viz['explanation']}")
                        if st.button(f"Generate Visualization {i}", key=f"viz_{i}"):
                            with st.spinner('Generating visualization...'):
                                fig = plot_generic_visualization(df, viz)
                                if fig:
                                    st.pyplot(fig)
                                    plt.close(fig)  # Properly close the figure
            else:
                st.info("No automatic visualizations could be generated for this dataset")
            
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
            st.subheader("Ask Your Own Question")
            custom_question = st.text_input("What would you like to know about your data?")
            
            if custom_question:
                with st.spinner('Analyzing your question...'):
                    process_question(df, custom_question)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()