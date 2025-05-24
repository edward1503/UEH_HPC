from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import pickle
import os
import sys
from typing import List, Dict, Any, Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
import shap
import lime
import lime.lime_tabular
import json
from sklearn.utils import resample

# Add parent directory to path to import from src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.preprocessing import create_preprocessing_pipeline
from src.models.model_explainer import explain_with_shap, explain_with_lime

# Create FastAPI app
app = FastAPI(
    title="Credit Default Risk Prediction API",
    description="API for predicting credit default risk using machine learning models",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)  # Get parent directory (UEH_HPC)
DEFAULT_MODEL_DIR = os.path.join(PARENT_DIR, 'models')  # UEH_HPC/models
MODEL_DIR = os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)

# Load models
models = {}
preprocessor = None
expected_features = 14  # The number of features expected by models

class CreditData(BaseModel):
    """
    Input schema for credit data
    """
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., example=0.8)
    Age: int = Field(..., example=45)
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(..., example=0)
    DebtRatio: float = Field(..., example=0.2)
    MonthlyIncome: Optional[float] = Field(None, example=5000)
    NumberOfOpenCreditLinesAndLoans: int = Field(..., example=8)
    NumberOfTimes90DaysLate: int = Field(..., example=0)
    NumberRealEstateLoansOrLines: int = Field(..., example=1)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(..., example=0)
    NumberOfDependents: Optional[int] = Field(None, example=2)

class PredictionResponse(BaseModel):
    """
    Response schema for predictions
    """
    prediction: int
    probability: float
    risk_level: str
    model: str
    
    class Config:
        protected_namespaces = ()

class ExplanationResponse(BaseModel):
    """
    Response schema for model explanations
    """
    feature_importance: Dict[str, float]
    lime_importance: Optional[Dict[str, float]] = None
    prediction: int
    probability: float
    model: str
    visualizations: Dict[str, str]  # Base64 encoded plots
    class Config:
        protected_namespaces = ()

class ModelComparisonResponse(BaseModel):
    """
    Response schema for model comparisons
    """
    predictions: Dict[str, Dict[str, Any]]

@app.on_event("startup")
async def load_ml_models():
    """
    Load ML models on startup
    """
    global models, preprocessor, expected_features
    
    print("="*50)
    print(f"Loading models from directory: {MODEL_DIR}")
    print(f"Directory exists: {os.path.exists(MODEL_DIR)}")
    print("="*50)
    
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load preprocessor
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
    
    if os.path.exists(preprocessor_path):
        try:
            print(f"Loading preprocessor from {preprocessor_path}")
            with open(preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)
                
                # Test if preprocessor is fitted by transforming a dummy sample
                try:
                    dummy_data = pd.DataFrame({
                        'RevolvingUtilizationOfUnsecuredLines': [0.5],
                        'Age': [40],
                        'NumberOfTime30_59DaysPastDueNotWorse': [0],
                        'DebtRatio': [0.2],
                        'MonthlyIncome': [5000],
                        'NumberOfOpenCreditLinesAndLoans': [5],
                        'NumberOfTimes90DaysLate': [0],
                        'NumberRealEstateLoansOrLines': [1],
                        'NumberOfTime60_89DaysPastDueNotWorse': [0],
                        'NumberOfDependents': [2]
                    })
                    transformed = preprocessor.transform(dummy_data)
                    expected_features = transformed.shape[1]
                    print(f"Preprocessor successfully loaded and is fitted. Output shape: {transformed.shape}")
                except Exception as e:
                    print(f"WARNING: Preprocessor is not fitted, will be ignored: {str(e)}")
                    preprocessor = None
        except Exception as e:
            print(f"Error loading preprocessor: {str(e)}")
            preprocessor = None
    else:
        print(f"Preprocessor file not found at {preprocessor_path}")
        preprocessor = None
    
    # Load models if directory exists
    try:
        # List all files in the model directory
        all_files = os.listdir(MODEL_DIR)
        print(f"Files in model directory: {all_files}")
        
        model_files = [f for f in all_files if f.endswith(".pkl") and f != "preprocessor.pkl"]
        print(f"Found model files: {model_files}")
        
        for model_file in model_files:
            model_name = os.path.splitext(model_file)[0]
            model_path = os.path.join(MODEL_DIR, model_file)
            
            try:
                print(f"Loading model: {model_name} from {model_path}")
                with open(model_path, "rb") as f:
                    models[model_name] = pickle.load(f)
                print(f"Successfully loaded model: {model_name}")
                
                # Also register models with spaces replaced by underscores
                if " " in model_name:
                    underscore_name = model_name.replace(" ", "_")
                    if underscore_name not in models:
                        models[underscore_name] = models[model_name]
                        print(f"Registered alias: {underscore_name} -> {model_name}")
            except Exception as e:
                print(f"Error loading model {model_name}: {str(e)}")
        
        print(f"Loaded {len(models)} models: {list(models.keys())}")
    except FileNotFoundError:
        print(f"WARNING: Model directory {MODEL_DIR} not found. No models loaded.")
    except Exception as e:
        print(f"ERROR loading models: {str(e)}")
        import traceback
        traceback.print_exc()

def process_input(credit_data: CreditData):
    """
    Process input data for prediction
    """
    # Convert to DataFrame
    data_dict = credit_data.dict()
    df = pd.DataFrame([data_dict])
    
    # Preserve original feature names for explanation use
    feature_names = list(df.columns)
    
    # Handle missing values
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna('')
    
    # Apply preprocessing if available and working
    if preprocessor is not None:
        try:
            X = preprocessor.transform(df)
            return X, feature_names
        except Exception as e:
            print(f"Error using preprocessor: {str(e)}")
            print("Falling back to using raw data")
    
    # If preprocessor not available or failed, use raw data
    X = df.values
    
    # Handle feature mismatch - adapt features to match the expected count
    if X.shape[1] != expected_features:
        print(f"WARNING: Feature count mismatch. Got {X.shape[1]}, expected {expected_features}")
        if X.shape[1] < expected_features:
            # Pad with zeros if we have fewer features than expected
            padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
            X = np.hstack((X, padding))
            # Also extend feature names
            padding_features = [f"padding_{i}" for i in range(expected_features - len(feature_names))]
            feature_names.extend(padding_features)
            print(f"Padded features with zeros to match expected count: {X.shape}")
        else:
            # Trim if we have more features than expected
            X = X[:, :expected_features]
            feature_names = feature_names[:expected_features]
            print(f"Trimmed features to match expected count: {X.shape}")
    
    return X, feature_names

def get_risk_level(probability):
    """
    Convert probability to risk level
    """
    if probability < 0.2:
        return "Very Low Risk"
    elif probability < 0.4:
        return "Low Risk"
    elif probability < 0.6:
        return "Medium Risk"
    elif probability < 0.8:
        return "High Risk"
    else:
        return "Very High Risk"

@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "message": "Credit Default Risk Prediction API",
        "available_models": list(models.keys()),
        "docs_url": "/docs"
    }

@app.get("/models")
async def get_models():
    """
    Get available models
    """
    return {"available_models": list(models.keys())}

@app.post("/predict/{model_name}", response_model=PredictionResponse)
async def predict(model_name: str, credit_data: CreditData):
    """
    Make a prediction using the specified model
    """
    # Handle case where model name might have spaces
    if model_name not in models:
        # Try with underscores replaced by spaces
        space_name = model_name.replace("_", " ")
        if space_name in models:
            model_name = space_name
        else:
            available_models = list(models.keys())
            raise HTTPException(
                status_code=404, 
                detail=f"Model '{model_name}' not found. Available models: {available_models}"
            )
    
    # Process input data
    X, _ = process_input(credit_data)
    
    # Get prediction
    try:
        model = models[model_name]
        probability = model.predict_proba(X)[0, 1]
        prediction = 1 if probability >= 0.5 else 0
        risk_level = get_risk_level(probability)
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "risk_level": risk_level,
            "model": model_name
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction with model {model_name}: {str(e)}"
        )

@app.post("/compare_models", response_model=ModelComparisonResponse)
async def compare_models(credit_data: CreditData):
    """
    Make predictions using all available models
    """
    if not models:
        raise HTTPException(status_code=500, detail="No models available")
    
    # Process input data
    X, feature_names = process_input(credit_data)
    
    # Get predictions from all models
    results = {}
    for model_name, model in models.items():
        try:
            probability = model.predict_proba(X)[0, 1]
            prediction = 1 if probability >= 0.5 else 0
            risk_level = get_risk_level(probability)
            
            results[model_name] = {
                "prediction": int(prediction),
                "probability": float(probability),
                "risk_level": risk_level,
                "model": model_name
            }
        except Exception as e:
            # Skip models that fail
            print(f"Error with model {model_name}: {str(e)}")
            results[model_name] = {
                "prediction": None,
                "probability": None,
                "risk_level": "Error",
                "error": str(e),
                "model": model_name
            }
    
    if not results:
        raise HTTPException(status_code=500, detail="All models failed to make predictions")
    
    return {"predictions": results}

def create_shap_force_plot(model, X, feature_names):
    """Create SHAP force plot using the improved model_explainer function"""
    try:
        # Import from src/models/model_explainer to use improved functions
        from src.models.model_explainer import create_shap_plot
        print(f"Creating SHAP force plot with X shape {X.shape} and {len(feature_names)} feature names")
        return create_shap_plot(model, X, feature_names=feature_names, plot_type="force")
    except Exception as e:
        print(f"Error creating force plot: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try with fallback
        try:
            from src.models.model_explainer import create_fallback_plot
            return create_fallback_plot(model, X, feature_names=feature_names, 
                                      error_message=f"Error creating SHAP force plot: {str(e)}")
        except Exception as fallback_error:
            print(f"Fallback plot creation failed: {str(fallback_error)}")
            return None

def create_shap_summary_plot(model, X, feature_names):
    """Create SHAP summary plot using the improved model_explainer function"""
    try:
        # Import from src/models/model_explainer to use improved functions
        from src.models.model_explainer import create_shap_plot
        print(f"Creating SHAP summary plot with X shape {X.shape} and {len(feature_names)} feature names")
        return create_shap_plot(model, X, feature_names=feature_names, plot_type="summary")
    except Exception as e:
        print(f"Error creating summary plot: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try with fallback
        try:
            from src.models.model_explainer import create_fallback_plot
            return create_fallback_plot(model, X, feature_names=feature_names, 
                                      error_message=f"Error creating SHAP summary plot: {str(e)}")
        except Exception as fallback_error:
            print(f"Fallback plot creation failed: {str(fallback_error)}")
            return None

def create_shap_dependence_plot(model, X, feature_names, feature_idx=0):
    """Create SHAP dependence plot using the improved model_explainer function"""
    try:
        # Import from src/models/model_explainer to use improved functions
        from src.models.model_explainer import create_shap_plot
        print(f"Creating SHAP dependence plot for feature {feature_idx}")
        return create_shap_plot(model, X, feature_names=feature_names, plot_type="dependence")
    except Exception as e:
        print(f"Error creating dependence plot: {str(e)}")
        
        # Try with fallback
        try:
            from src.models.model_explainer import create_fallback_plot
            return create_fallback_plot(model, X, feature_names=feature_names, 
                                      error_message=f"Error creating SHAP dependence plot: {str(e)}")
        except Exception as fallback_error:
            print(f"Fallback plot creation failed: {str(fallback_error)}")
            return None

def create_lime_explanation_plot(model, X, feature_names, class_names=['No Default', 'Default']):
    """Create LIME explanation plot using the improved model_explainer function"""
    try:
        # Import from src/models/model_explainer to use improved functions
        from src.models.model_explainer import create_lime_plot
        # Use the same data for both training and testing to avoid dimensionality mismatch
        print(f"Creating LIME plot with X shape {X.shape} and {len(feature_names)} feature names")
        return create_lime_plot(model, X, X, feature_names=feature_names, class_names=class_names)
    except Exception as e:
        print(f"Error creating LIME plot: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try with fallback
        try:
            from src.models.model_explainer import create_fallback_plot
            return create_fallback_plot(model, X, feature_names=feature_names, 
                                      error_message=f"Error creating LIME plot: {str(e)}")
        except Exception as fallback_error:
            print(f"Fallback plot creation failed: {str(fallback_error)}")
            return None

@app.post("/explain/{model_name}", response_model=ExplanationResponse)
async def explain_prediction(model_name: str, credit_data: CreditData):
    """
    Explain prediction using SHAP values and LIME with additional visualizations
    """
    # Handle case where model name might have spaces
    if model_name not in models:
        # Try with underscores replaced by spaces
        space_name = model_name.replace("_", " ")
        if space_name in models:
            model_name = space_name
        else:
            available_models = list(models.keys())
            raise HTTPException(
                status_code=404, 
                detail=f"Model '{model_name}' not found. Available models: {available_models}"
            )
    
    # Process input data
    X, feature_names = process_input(credit_data)
    
    # Get model
    model = models[model_name]
    
    try:
        # Get prediction
        probability = model.predict_proba(X)[0, 1]
        prediction = 1 if probability >= 0.5 else 0
        
        # Initialize empty explanations
        shap_dict = {}
        lime_dict = {}
        visualizations = {}
        
        # Import the improved model_explainer functions
        try:
            from src.models.model_explainer import explain_with_shap, explain_with_lime
            print("Successfully imported model_explainer functions")
        except Exception as import_e:
            print(f"Error importing model_explainer functions: {str(import_e)}")
        
        # Generate SHAP visualizations - use improved error handling
        try:
            # Generate SHAP explanations
            shap_dict = explain_with_shap(model, X, feature_names=feature_names)
            
            # Force plot
            print(f"Attempting to create SHAP force plot for {model_name}...")
            force_plot = create_shap_force_plot(model, X, feature_names)
            if force_plot:
                print(f"SHAP force plot created successfully")
                visualizations['shap_force'] = force_plot
            
            # Summary plot
            print(f"Attempting to create SHAP summary plot for {model_name}...")
            summary_plot = create_shap_summary_plot(model, X, feature_names)
            if summary_plot:
                print(f"SHAP summary plot created successfully")
                visualizations['shap_summary'] = summary_plot
        except Exception as e:
            print(f"Error generating SHAP visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Generate LIME visualization - improved error handling
        try:
            # Generate LIME explanations
            lime_dict = explain_with_lime(model, X, X, feature_names=feature_names, 
                                         class_names=['No Default', 'Default'])
            
            print(f"Attempting to create LIME explanation plot for {model_name}...")
            lime_plot = create_lime_explanation_plot(model, X, feature_names)
            if lime_plot:
                print(f"LIME plot created successfully")
                visualizations['lime'] = lime_plot
        except Exception as e:
            print(f"Error generating LIME visualization: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # If no visualizations were generated, create synthetic ones
        if not visualizations:
            print("No visualizations were generated, creating synthetic ones")
            try:
                # Create a synthetic force plot
                plt.figure(figsize=(10, 3))
                if shap_dict:
                    values = list(shap_dict.values())
                    features = list(shap_dict.keys())
                    # Sort by absolute value
                    sorted_indices = sorted(range(len(values)), key=lambda i: abs(values[i]), reverse=True)
                    # Take top 10 or fewer
                    top_n = min(10, len(sorted_indices))
                    top_indices = sorted_indices[:top_n]
                    
                    plt.barh(
                        [features[i] for i in top_indices],
                        [values[i] for i in top_indices]
                    )
                    plt.title("Feature Contribution (Synthetic Plot)")
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    plt.close()
                    buf.seek(0)
                    visualizations['shap_force'] = base64.b64encode(buf.getvalue()).decode('utf-8')
                
                # Create a synthetic LIME plot
                if lime_dict:
                    plt.figure(figsize=(10, 6))
                    lime_features = list(lime_dict.keys())
                    lime_values = list(lime_dict.values())
                    
                    # Sort by absolute value
                    sorted_indices = sorted(range(len(lime_values)), key=lambda i: abs(lime_values[i]), reverse=True)
                    # Take top 10 or fewer
                    top_n = min(10, len(sorted_indices))
                    top_indices = sorted_indices[:top_n]
                    
                    plt.barh(
                        [lime_features[i] for i in top_indices],
                        [lime_values[i] for i in top_indices]
                    )
                    plt.title("LIME Explanation (Synthetic Plot)")
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    plt.close()
                    buf.seek(0)
                    visualizations['lime'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            except Exception as e:
                print(f"Error creating synthetic visualizations: {str(e)}")
        
        print(f"Final visualization keys: {list(visualizations.keys())}")
        
        return {
            "feature_importance": shap_dict,
            "lime_importance": lime_dict or None,  # Return None if empty
            "prediction": int(prediction),
            "probability": float(probability),
            "model": model_name,
            "visualizations": visualizations
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating explanation for model {model_name}: {str(e)}"
        )

@app.get("/explain/visualization/{model_name}/{plot_type}", response_class=HTMLResponse)
async def get_visualization(model_name: str, plot_type: str, credit_data: CreditData):
    """
    Get a specific visualization for the explanation
    """
    if model_name not in models:
        space_name = model_name.replace("_", " ")
        if space_name in models:
            model_name = space_name
        else:
            available_models = list(models.keys())
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Available models: {available_models}"
            )
    
    # Process input data
    X, feature_names = process_input(credit_data)
    model = models[model_name]
    
    try:
        if plot_type == 'shap_force':
            plot_data = create_shap_force_plot(model, X, feature_names)
        elif plot_type == 'shap_summary':
            plot_data = create_shap_summary_plot(model, X, feature_names)
        elif plot_type.startswith('shap_dependence_'):
            feature = plot_type.replace('shap_dependence_', '')
            if feature in feature_names:
                feature_idx = feature_names.index(feature)
                plot_data = create_shap_dependence_plot(model, X, feature_names, feature_idx)
            else:
                raise HTTPException(status_code=404, detail=f"Feature '{feature}' not found")
        elif plot_type == 'lime':
            plot_data = create_lime_explanation_plot(model, X, feature_names)
        else:
            raise HTTPException(status_code=404, detail=f"Plot type '{plot_type}' not found")
        
        if plot_data:
            return f"""
            <html>
                <body>
                    <img src="data:image/png;base64,{plot_data}" alt="{plot_type}">
                </body>
            </html>
            """
        else:
            raise HTTPException(status_code=500, detail="Failed to generate visualization")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating visualization: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 