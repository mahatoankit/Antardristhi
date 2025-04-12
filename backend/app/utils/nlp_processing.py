import re
import logging
import joblib
import os
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if required libraries are available
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    # Download necessary NLTK data
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    NLTK_AVAILABLE = True
except ImportError:
    logger.warning("NLTK not available. Basic NLP functionality will be used.")
    NLTK_AVAILABLE = False

try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning(
        "Transformers library not available. Advanced NLP capabilities will be limited."
    )
    TRANSFORMERS_AVAILABLE = False

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    logger.warning(
        "Google Generative AI library not available. LLM-powered analysis will be limited."
    )
    GEMINI_AVAILABLE = False

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not available. Using environment variables directly.")

# Cache directory for NLP models
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "cache", "nlp"
)
os.makedirs(CACHE_DIR, exist_ok=True)

# Intent classification patterns
INTENT_PATTERNS = {
    "show_trends": [
        r"(?i)(?:show|display|view|analyze).*(?:trend|over time|timeseries|time series)",
        r"(?i)(?:how|what).*(?:trend|change|evolve|over time)",
        r"(?i)(?:forecast|predict|projection)",
    ],
    "summarize": [
        r"(?i)(?:summarize|summary|overview|describe|tell me about)",
        r"(?i)(?:statistics|stats|describe)",
        r"(?i)(?:what|how).*(?:summary|description)",
    ],
    "segment": [
        r"(?i)(?:segment|cluster|group|categorize)",
        r"(?i)(?:identify|find).*(?:segments|clusters|groups|categories)",
    ],
    "outliers": [
        r"(?i)(?:outlier|anomaly|unusual|abnormal|irregular)",
        r"(?i)(?:detect|find|identify).*(?:outlier|anomaly|unusual)",
    ],
    "comparison": [
        r"(?i)(?:compare|comparison|versus|vs|difference)",
        r"(?i)(?:how|what).*(?:compare|comparison|differ)",
    ],
    "correlation": [
        r"(?i)(?:correlate|correlation|relationship|association)",
        r"(?i)(?:how|what|is).*(?:related|correlated|associated)",
    ],
    "distribution": [
        r"(?i)(?:distribute|distribution|spread|range)",
        r"(?i)(?:how|what).*(?:distributed|spread)",
    ],
    "top_bottom": [
        r"(?i)(?:top|bottom|highest|lowest|best|worst).*\d+",
        r"(?i)(?:rank|ranking)",
    ],
}

# Entity extraction patterns
ENTITY_PATTERNS = {
    "time_period": [
        r"(?i)(?:last|past|previous|recent)\s+(\d+)\s+(?:day|week|month|year|quarter)s?",
        r"(?i)(?:from|between)\s+(\d{4}-\d{2}-\d{2})\s+(?:to|and|until)\s+(\d{4}-\d{2}-\d{2})",
        r"(?i)(?:since|after)\s+(\d{4}-\d{2}-\d{2})",
        r"(?i)(?:until|before)\s+(\d{4}-\d{2}-\d{2})",
        r"(?i)(?:in|during|for)\s+(?:the\s+)?(?:year|month)?\s*(\d{4})",
    ],
    "metrics": [
        r"(?i)(?:sum|total|aggregate)\s+(?:of\s+)?([a-zA-Z_]+(?:\s+[a-zA-Z_]+)*)",
        r"(?i)(?:average|avg|mean)\s+(?:of\s+)?([a-zA-Z_]+(?:\s+[a-zA-Z_]+)*)",
        r"(?i)(?:count|number)\s+(?:of\s+)?([a-zA-Z_]+(?:\s+[a-zA-Z_]+)*)",
    ],
    "dimensions": [
        r"(?i)(?:by|grouped by|segmented by|per|across)\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)*)",
    ],
    "filters": [
        r'(?i)(?:where|for|with)\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)*)\s+(?:is|=|equals|equal to)\s+([a-zA-Z0-9_"\']+)',
        r"(?i)(?:where|for|with)\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)*)\s+(?:>|greater than)\s+(\d+(?:\.\d+)?)",
        r"(?i)(?:where|for|with)\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)*)\s+(?:<|less than)\s+(\d+(?:\.\d+)?)",
    ],
    "number": [
        r"(?i)(?:top|bottom)\s+(\d+)",
        r"(\d+)\s+(?:percent|%)",
        r"(\d+(?:\.\d+)?)",
    ],
}


def preprocess_text(text: str) -> str:
    """
    Preprocess text by converting to lowercase, removing special characters, etc.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove special characters but keep spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def classify_intent(query: str) -> Dict[str, float]:
    """
    Classify the user's intent based on the query

    Args:
        query: User's natural language query

    Returns:
        Dictionary with intent types and confidence scores
    """
    query = preprocess_text(query)

    # Initialize scores
    intent_scores = {intent: 0.0 for intent in INTENT_PATTERNS.keys()}

    # Check for pattern matches
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query):
                intent_scores[intent] += 1.0

    # Normalize scores if any matches found
    total_score = sum(intent_scores.values())
    if total_score > 0:
        for intent in intent_scores:
            intent_scores[intent] /= total_score

    # If using transformers for advanced intent classification
    if TRANSFORMERS_AVAILABLE and total_score == 0:
        try:
            # Load or create text classification pipeline
            classifier_path = os.path.join(CACHE_DIR, "text_classifier")

            if os.path.exists(classifier_path):
                classifier = joblib.load(classifier_path)
            else:
                classifier = pipeline("zero-shot-classification")
                joblib.dump(classifier, classifier_path)

            # Classify using zero-shot classification
            result = classifier(query, list(INTENT_PATTERNS.keys()))

            # Update scores with model results
            for intent, score in zip(result["labels"], result["scores"]):
                intent_scores[intent] = score

        except Exception as e:
            logger.error(f"Error in transformer-based intent classification: {str(e)}")

    # Get primary intent (highest score)
    primary_intent = max(intent_scores, key=intent_scores.get)
    confidence = intent_scores[primary_intent]

    return {
        "primary_intent": primary_intent,
        "confidence": confidence,
        "all_intents": intent_scores,
    }


def extract_entities(query: str) -> Dict[str, Any]:
    """
    Extract entities like metrics, dimensions, time periods from query

    Args:
        query: User's natural language query

    Returns:
        Dictionary with extracted entities
    """
    entities = {
        "time_period": [],
        "metrics": [],
        "dimensions": [],
        "filters": [],
        "numbers": [],
    }

    # Apply regex patterns to extract entities
    for entity_type, patterns in ENTITY_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                if entity_type == "filters" and len(match.groups()) >= 2:
                    entities["filters"].append(
                        {
                            "column": match.group(1).strip(),
                            "value": match.group(2).strip().strip("\"'"),
                        }
                    )
                elif entity_type == "time_period" and len(match.groups()) >= 2:
                    # For date ranges with start and end
                    entities["time_period"].append(
                        {
                            "start": match.group(1),
                            "end": match.group(2) if len(match.groups()) > 1 else None,
                        }
                    )
                elif entity_type == "number":
                    entities["numbers"].append(float(match.group(1)))
                else:
                    # For other entity types
                    entity_value = match.group(1).strip()
                    if entity_type == "metrics":
                        entities["metrics"].append(entity_value)
                    elif entity_type == "dimensions":
                        entities["dimensions"].append(entity_value)
                    elif entity_type == "time_period":
                        entities["time_period"].append(entity_value)

    # Clean up entity lists (remove duplicates)
    for entity_type in entities:
        if entity_type != "filters":
            entities[entity_type] = (
                list(set(entities[entity_type]))
                if isinstance(entities[entity_type], list)
                else entities[entity_type]
            )

    return entities


def match_entities_to_columns(
    entities: Dict[str, Any], df_columns: List[str]
) -> Dict[str, Any]:
    """
    Match extracted entities to actual dataframe columns

    Args:
        entities: Extracted entities from query
        df_columns: List of column names in dataframe

    Returns:
        Dictionary with matched columns
    """
    matched = {
        "time_columns": [],
        "metric_columns": [],
        "dimension_columns": [],
        "filter_columns": [],
    }

    # Helper function to find closest matching column
    def find_closest_match(entity, columns):
        # Exact match
        for col in columns:
            if entity.lower() == col.lower():
                return col

        # Word in column
        for col in columns:
            if entity.lower() in col.lower().split("_"):
                return col

        # Substring match
        for col in columns:
            if entity.lower() in col.lower():
                return col

        return None

    # Match metrics to numeric columns
    for metric in entities.get("metrics", []):
        match = find_closest_match(metric, df_columns)
        if match and match not in matched["metric_columns"]:
            matched["metric_columns"].append(match)

    # Match dimensions
    for dimension in entities.get("dimensions", []):
        match = find_closest_match(dimension, df_columns)
        if match and match not in matched["dimension_columns"]:
            matched["dimension_columns"].append(match)

    # Match filters
    for filter_item in entities.get("filters", []):
        col_match = find_closest_match(filter_item["column"], df_columns)
        if col_match:
            matched["filter_columns"].append(
                {"column": col_match, "value": filter_item["value"]}
            )

    # Try to identify time columns
    time_keywords = ["date", "time", "year", "month", "day", "quarter"]
    for col in df_columns:
        for keyword in time_keywords:
            if keyword in col.lower() and col not in matched["time_columns"]:
                matched["time_columns"].append(col)
                break

    return matched


def generate_analysis_plan(
    intent: str, entities: Dict[str, Any], matched_columns: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a plan for analyzing the data based on intent and entities

    Args:
        intent: Detected primary intent
        entities: Extracted entities
        matched_columns: Matched dataframe columns

    Returns:
        Dictionary with analysis plan
    """
    plan = {
        "intent": intent,
        "analysis_type": None,
        "columns_to_use": [],
        "parameters": {},
        "description": "",
    }

    # Set analysis type based on intent
    if intent == "show_trends":
        plan["analysis_type"] = "time_series"

        # Get time column (date/time)
        if matched_columns["time_columns"]:
            plan["parameters"]["date_col"] = matched_columns["time_columns"][0]
            plan["columns_to_use"].append(matched_columns["time_columns"][0])

        # Get value column (metric)
        if matched_columns["metric_columns"]:
            plan["parameters"]["value_col"] = matched_columns["metric_columns"][0]
            plan["columns_to_use"].append(matched_columns["metric_columns"][0])

        # Set forecast periods
        plan["parameters"]["periods"] = (
            entities.get("numbers", [30])[0] if entities.get("numbers") else 30
        )

        plan["description"] = (
            f"Analyze time series trends for {plan['parameters'].get('value_col', 'metrics')} over time"
        )

    elif intent == "segment":
        plan["analysis_type"] = "clustering"

        # Use specified dimensions or metrics for segmentation
        if matched_columns["dimension_columns"]:
            plan["columns_to_use"].extend(matched_columns["dimension_columns"])

        if matched_columns["metric_columns"]:
            plan["columns_to_use"].extend(matched_columns["metric_columns"])

        # Number of clusters
        plan["parameters"]["n_clusters"] = (
            int(entities.get("numbers", [0])[0]) if entities.get("numbers") else None
        )

        plan["description"] = (
            f"Segment data using {', '.join(plan['columns_to_use'])} as features"
        )

    elif intent == "outliers":
        plan["analysis_type"] = "anomaly_detection"

        # Use specified metrics for outlier detection
        if matched_columns["metric_columns"]:
            plan["columns_to_use"].extend(matched_columns["metric_columns"])

        # Contamination parameter (proportion of outliers)
        contamination = 0.05  # Default
        if entities.get("numbers"):
            num = entities.get("numbers")[0]
            if num < 1:  # Assuming it's already a proportion
                contamination = num
            else:  # Assuming it's a percentage
                contamination = num / 100

        plan["parameters"]["contamination"] = contamination

        plan["description"] = f"Detect outliers in {', '.join(plan['columns_to_use'])}"

    elif intent == "summarize":
        plan["analysis_type"] = "summary"

        # Include all relevant columns
        if matched_columns["metric_columns"]:
            plan["columns_to_use"].extend(matched_columns["metric_columns"])

        if matched_columns["dimension_columns"]:
            plan["columns_to_use"].extend(matched_columns["dimension_columns"])

        if matched_columns["time_columns"]:
            plan["columns_to_use"].extend(matched_columns["time_columns"])

        plan["description"] = "Provide a summary of the data with basic statistics"

    elif intent == "correlation":
        plan["analysis_type"] = "correlation"

        # Include metrics
        if matched_columns["metric_columns"]:
            plan["columns_to_use"].extend(matched_columns["metric_columns"])

        plan["description"] = (
            f"Analyze correlations between {', '.join(plan['columns_to_use'])}"
        )

    elif intent == "distribution":
        plan["analysis_type"] = "distribution"

        # Column to analyze distribution
        if matched_columns["metric_columns"]:
            plan["columns_to_use"].append(matched_columns["metric_columns"][0])

        if matched_columns["dimension_columns"]:
            plan["parameters"]["group_by"] = matched_columns["dimension_columns"][0]
            plan["columns_to_use"].append(matched_columns["dimension_columns"][0])

        plan["description"] = f"Analyze distribution of {plan['columns_to_use'][0]}"

    elif intent == "top_bottom":
        plan["analysis_type"] = "ranking"

        # Metrics to rank by
        if matched_columns["metric_columns"]:
            plan["parameters"]["rank_by"] = matched_columns["metric_columns"][0]
            plan["columns_to_use"].append(matched_columns["metric_columns"][0])

        # Dimension to group by
        if matched_columns["dimension_columns"]:
            plan["parameters"]["group_by"] = matched_columns["dimension_columns"][0]
            plan["columns_to_use"].append(matched_columns["dimension_columns"][0])

        # Number of items
        plan["parameters"]["top_n"] = (
            int(entities.get("numbers", [10])[0]) if entities.get("numbers") else 10
        )

        # Direction (top or bottom)
        plan["parameters"]["ascending"] = "bottom" in entities.get("prompt", "").lower()

        plan["description"] = (
            f"Rank {plan['parameters'].get('group_by', 'items')} by {plan['parameters'].get('rank_by', 'value')}"
        )

    elif intent == "comparison":
        plan["analysis_type"] = "comparison"

        # Metrics to compare
        if matched_columns["metric_columns"]:
            plan["columns_to_use"].extend(matched_columns["metric_columns"])

        # Dimension to compare by
        if matched_columns["dimension_columns"]:
            plan["parameters"]["compare_by"] = matched_columns["dimension_columns"][0]
            plan["columns_to_use"].append(matched_columns["dimension_columns"][0])

        plan["description"] = (
            f"Compare {', '.join(matched_columns['metric_columns'])} by {plan['parameters'].get('compare_by', 'groups')}"
        )

    # Add filters
    if matched_columns["filter_columns"]:
        plan["parameters"]["filters"] = matched_columns["filter_columns"]

    return plan


def generate_llm_explanation(
    analysis_result: Dict[str, Any],
    plan: Dict[str, Any],
    prompt_template: Optional[str] = None,
) -> str:
    """
    Generate natural language explanation of analysis using LLM

    Args:
        analysis_result: Results from the analysis
        plan: Analysis plan
        prompt_template: Optional template for the LLM prompt

    Returns:
        Natural language explanation of results
    """
    # Default explanation (in case LLM is not available)
    default_explanation = f"Analysis of {plan['description']} completed. "

    # Add relevant stats based on analysis type
    if plan["analysis_type"] == "time_series":
        if "trend_direction" in analysis_result:
            default_explanation += (
                f"The overall trend is {analysis_result['trend_direction']}. "
            )
        if "percent_change" in analysis_result:
            default_explanation += f"There was a {analysis_result['percent_change']}% change over the period. "
        if "forecast" in analysis_result:
            default_explanation += f"A forecast for the next {len(analysis_result.get('forecast', []))} periods has been generated. "

    elif plan["analysis_type"] == "clustering":
        if "n_clusters" in analysis_result:
            default_explanation += (
                f"The data was grouped into {analysis_result['n_clusters']} segments. "
            )
        if "segment_descriptions" in analysis_result:
            default_explanation += "Key segments include: "
            for i, desc in enumerate(
                analysis_result.get("segment_descriptions", [])[:3]
            ):
                default_explanation += f"{desc} "

    elif plan["analysis_type"] == "anomaly_detection":
        if "outlier_count" in analysis_result:
            default_explanation += f"Found {analysis_result['outlier_count']} outliers "
            if "outlier_percentage" in analysis_result:
                default_explanation += (
                    f"({analysis_result['outlier_percentage']:.2f}% of the data). "
                )

    # Try to use Gemini for better explanation if available
    if GEMINI_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
        try:
            # Configure the Gemini API
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

            # Select the Gemini Flash model
            model = genai.GenerativeModel("gemini-1.5-flash")

            # Default prompt template if none provided
            if not prompt_template:
                prompt_template = """
                You are an expert data analyst explaining insights to a business user.
                
                Analysis type: {analysis_type}
                Description: {description}
                
                Here are the results:
                {results}
                
                Provide a clear, concise explanation of these results in 3-5 sentences.
                Focus on the most important business insights. Be specific about numbers and trends.
                """

            # Format the prompt
            prompt = prompt_template.format(
                analysis_type=plan["analysis_type"],
                description=plan["description"],
                results=json.dumps(analysis_result, indent=2),
            )

            # Call the Gemini API
            response = model.generate_content(prompt)

            # Extract and return the explanation
            if response and response.text:
                return response.text

        except Exception as e:
            logger.error(f"Error generating Gemini explanation: {str(e)}")

    # Return default explanation if Gemini fails or is not available
    return default_explanation


def process_natural_language_query(query: str, df_columns: List[str]) -> Dict[str, Any]:
    """
    Process a natural language query and return analysis plan

    Args:
        query: User's natural language query
        df_columns: List of column names in dataframe

    Returns:
        Dictionary with analysis plan
    """
    try:
        # Classify intent
        intent_result = classify_intent(query)
        primary_intent = intent_result["primary_intent"]

        # Extract entities
        entities = extract_entities(query)
        entities["prompt"] = query  # Include original prompt

        # Match entities to columns
        matched_columns = match_entities_to_columns(entities, df_columns)

        # Generate analysis plan
        plan = generate_analysis_plan(primary_intent, entities, matched_columns)

        return {
            "query": query,
            "intent_classification": intent_result,
            "entities": entities,
            "matched_columns": matched_columns,
            "analysis_plan": plan,
        }
    except Exception as e:
        logger.error(f"Error processing natural language query: {str(e)}")
        # Provide a fallback basic analysis plan
        fallback_plan = {
            "analysis_type": "summary",
            "columns_to_use": df_columns[:10],  # Use first 10 columns
            "parameters": {},
            "description": "Basic data summary",
        }

        return {
            "query": query,
            "intent_classification": {"primary_intent": "summarize", "confidence": 0.5},
            "entities": {"prompt": query},
            "matched_columns": {"metric_columns": df_columns[:5]},
            "analysis_plan": fallback_plan,
        }
