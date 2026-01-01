"""
LangChain utilities for AI-powered insights
Integrates with Google Gemini API for intelligent analysis
"""

import os
import json
import pandas as pd
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

# Initialize Hugging Face LLM
HF_API_KEY = os.getenv("HF_API_KEY")

if not HF_API_KEY:
    raise ValueError("HF_API_KEY not found in environment variables")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    huggingfacehub_api_token=HF_API_KEY,
    temperature=0.7,
    model_kwargs={"max_new_tokens": 256}
)

# ============ FEATURE #1: INTELLIGENT EDA ============

def intelligent_eda_analysis(df: pd.DataFrame, target_column: str) -> dict:
    """
    Feature #1: Intelligent Data Analysis
    AI analyzes the dataset and provides insights
    
    Returns:
        - recommendations: What to do with the data
        - problem_type: Regression or Classification
        - key_insights: Important patterns in the data
        - warning_flags: Issues to address
    """
    
    # Gather basic statistics
    basic_stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "target_column": target_column,
        "data_types": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_columns": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
    }
    
    # Create analysis prompt
    prompt_template = PromptTemplate(
        input_variables=["data_info"],
        template="""You are a data science expert. Analyze this dataset information and provide insights:

Dataset Information:
{data_info}

Provide a JSON response with:
1. "problem_type": "regression" or "classification" (based on target column)
2. "key_insights": [list of 2-3 important patterns or observations]
3. "warning_flags": [list of data quality issues if any]
4. "recommended_features": [top 3 most useful features for modeling]
5. "data_quality_score": score from 1-10
6. "next_steps": [actionable recommendations]

Respond ONLY with valid JSON, no markdown or code blocks."""
    )
    
    # Format data info for LLM
    data_info = f"""
    Total Rows: {basic_stats['total_rows']}
    Total Columns: {basic_stats['total_columns']}
    Target Column: {target_column}
    Target Data Type: {basic_stats['data_types'].get(target_column, 'Unknown')}
    
    Numeric Columns: {', '.join(basic_stats['numeric_columns'])}
    Categorical Columns: {', '.join(basic_stats['categorical_columns'])}
    
    Missing Values:
    {json.dumps(basic_stats['missing_values'], indent=2)}
    
    Sample Statistics:
    - Min rows in any column: {df.isnull().sum().min()}
    - Max missing in single column: {df.isnull().sum().max()}
    - Columns with no missing data: {sum(1 for x in df.isnull().sum() if x == 0)}
    """
    
    # Run LLM with prompt
    chain = prompt_template | llm
    response = chain.invoke({"data_info": data_info})
    
    # Extract text content from response
    response_text = response.content if hasattr(response, 'content') else str(response)
    
    try:
        # Parse JSON response
        analysis = json.loads(response_text)
        return {
            "success": True,
            "analysis": analysis,
            "basic_stats": basic_stats
        }
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return {
            "success": False,
            "error": "Could not parse AI response",
            "raw_response": response,
            "basic_stats": basic_stats
        }

# ============ FEATURE #3: DATA QUALITY REPORT ============

def generate_data_quality_report(df: pd.DataFrame) -> str:
    """
    Feature #3: Generate human-readable data quality report
    """
    
    stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_percent": round((df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100), 2),
        "columns_with_missing": sum(1 for x in df.isnull().sum() if x > 0),
        "duplicate_rows": len(df[df.duplicated()]),
    }
    
    prompt_template = PromptTemplate(
        input_variables=["stats"],
        template="""Create a friendly, non-technical data quality report based on these statistics:

{stats}

Write a 2-3 sentence summary explaining:
1. Overall data quality (excellent/good/fair/poor)
2. Main data quality issues
3. What action to take

Use plain language, avoid technical jargon."""
    )
    
    chain = prompt_template | llm
    response = chain.invoke({"stats": json.dumps(stats, indent=2)})
    report = response.content if hasattr(response, 'content') else str(response)
    
    return report

# ============ FEATURE #5: MODEL RECOMMENDATIONS ============

def recommend_models(df: pd.DataFrame, target_column: str, problem_type: str) -> dict:
    """
    Feature #5: Recommend best algorithms for the problem
    """
    
    dataset_info = {
        "rows": len(df),
        "columns": len(df.columns),
        "problem_type": problem_type,
        "target_column": target_column,
        "sample_size": "small" if len(df) < 1000 else "medium" if len(df) < 100000 else "large",
    }
    
    prompt_template = PromptTemplate(
        input_variables=["dataset_info"],
        template="""As an ML expert, recommend the best algorithm for this dataset:

Dataset Info:
{dataset_info}

Provide JSON with:
1. "recommended_algorithm": Best choice for this problem
2. "why": 2 sentence explanation
3. "expected_performance": Realistic accuracy/R² range
4. "alternatives": [2 other good options]

Respond ONLY with valid JSON."""
    )
    
    chain = prompt_template | llm
    response = chain.invoke({"dataset_info": json.dumps(dataset_info)})
    response_text = response.content if hasattr(response, 'content') else str(response)
    
    try:
        recommendations = json.loads(response_text)
        return {"success": True, "recommendations": recommendations}
    except json.JSONDecodeError:
        return {"success": False, "error": "Could not parse recommendations"}

# ============ FEATURE #6: CHAT ASSISTANT ============

def chat_with_assistant(question: str, context: dict = None) -> str:
    """
    Feature #6: Interactive chat assistant for model questions
    
    Args:
        question: User's question about the data/model
        context: Optional context about current dataset/model
    """
    
    context_text = ""
    if context:
        context_text = f"""
        Current Context:
        - Dataset rows: {context.get('rows', 'N/A')}
        - Target column: {context.get('target_column', 'N/A')}
        - Model type: {context.get('model_type', 'N/A')}
        - Current accuracy: {context.get('accuracy', 'N/A')}
        """
    
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful ML assistant. Answer the user's question about their ML project.
{context}

User Question: {question}

Provide a helpful, friendly answer. Be specific and actionable."""
    )
    
    chain = prompt_template | llm
    response = chain.invoke({"context": context_text, "question": question})
    answer = response.content if hasattr(response, 'content') else str(response)
    
    return answer

# ============ FEATURE #7: REPORT GENERATION ============

def generate_model_report(model_name: str, model_type: str, score: float, features: list) -> str:
    """
    Feature #7: Auto-generate model performance report
    """
    
    model_info = {
        "model_name": model_name,
        "model_type": model_type,
        "performance_score": score,
        "features_used": len(features),
        "feature_list": features[:5],  # Top 5 features
    }
    
    prompt_template = PromptTemplate(
        input_variables=["model_info"],
        template="""Generate a professional ML model report based on:

Model Info:
{model_info}

Create a 3-4 sentence report including:
1. Model performance assessment
2. Key insights about features
3. Recommendations for improvement

Write in a professional but friendly tone."""
    )
    
    chain = prompt_template | llm
    response = chain.invoke({"model_info": json.dumps(model_info, indent=2)})
    report = response.content if hasattr(response, 'content') else str(response)
    
    return report

# ============ FEATURE #10: QUERY PREDICTIONS ============

def explain_prediction(features: dict, prediction: float, prediction_type: str = "regression") -> str:
    """
    Feature #10: Explain what a prediction means
    
    Args:
        features: Dictionary of feature values used
        prediction: The predicted value or class
        prediction_type: "regression" or "classification"
    """
    
    prompt_template = PromptTemplate(
        input_variables=["features", "prediction", "pred_type"],
        template="""Explain this ML prediction in simple terms:

Input Features: {features}
Prediction: {prediction}
Prediction Type: {pred_type}

Provide:
1. What this prediction means
2. Why these features led to this prediction
3. Confidence level (high/medium/low)

Keep it non-technical and friendly."""
    )
    
    chain = prompt_template | llm
    response = chain.invoke(
        {
            "features": json.dumps(features),
            "prediction": str(prediction),
            "pred_type": prediction_type
        }
    )
    explanation = response.content if hasattr(response, 'content') else str(response)
    
    return explanation
