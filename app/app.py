import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import time
import base64
import pickle

# Set API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Base directory for loading evaluation results
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Set page configuration
st.set_page_config(
    page_title="Credit Default Risk Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",  # Start with collapsed sidebar
)

# Sidebar with contributor info
st.sidebar.title("Contributors")
st.sidebar.markdown("""
- **Nguyễn Đôn Đức**
- **Đỗ Nhật Phương**
- **Bùi Tiến Hiếu**
- **Lê Đình Anh Khoa**
""")

# Main header
st.title("Credit Default Risk Prediction")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_models():
    """Get available models from API - cached to improve performance"""
    try:
        response = requests.get(f"{API_URL}/models")
        return response.json()["available_models"]
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return []

def predict_risk(data, model_name):
    """Send prediction request to API"""
    try:
        response = requests.post(
            f"{API_URL}/predict/{model_name}",
            json=data
        )
        return response.json()
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def compare_models(data):
    """Compare predictions from all models"""
    try:
        response = requests.post(
            f"{API_URL}/compare_models",
            json=data
        )
        return response.json()["predictions"]
    except Exception as e:
        st.error(f"Error comparing models: {e}")
        return None

def explain_prediction(data, model_name):
    """Get model explanation for prediction"""
    try:
        # Add retry logic with increased timeout
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{API_URL}/explain/{model_name}",
                    json=data,
                    timeout=60  # Increased timeout to 60 seconds
                )
                if response.status_code == 200:
                    explanation_data = response.json()
                    
                    # Debug info
                    if "visualizations" in explanation_data:
                        st.session_state["debug_vis_keys"] = list(explanation_data["visualizations"].keys())
                    else:
                        st.session_state["debug_vis_keys"] = "No visualizations found"
                    
                    return explanation_data
                else:
                    st.error(f"Error fetching explanation: {response.status_code} - {response.text}")
                    if attempt < 2:
                        time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s
                    else:
                        return None
            except requests.exceptions.Timeout:
                if attempt < 2:
                    st.warning(f"Request timed out, retrying (attempt {attempt+1}/3)...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    st.error("Explanation request timed out after multiple attempts")
                    return None
            except Exception as e:
                st.error(f"Error getting explanation: {e}")
                return None
        
        return None
    except Exception as e:
        st.error(f"Error getting explanation: {e}")
        return None

def display_model_explanations(response_data, col):
    """Display model explanations and visualizations"""
    if not response_data:
        col.warning("No explanation data available")
        return

    # Check if visualizations are available
    has_visualizations = "visualizations" in response_data and any([
        "shap_force" in response_data["visualizations"],
        "shap_summary" in response_data["visualizations"],
        "lime" in response_data["visualizations"]
    ])

    if not has_visualizations:
        col.warning("No visualizations are available for this model prediction")
        
        # Display feature importance as fallback
        if "feature_importance" in response_data:
            col.subheader("Feature Importance")
            
            # Create a DataFrame for better visualization
            importance_df = pd.DataFrame({
                'Feature': list(response_data["feature_importance"].keys()),
                'Importance': list(response_data["feature_importance"].values())
            })
            
            # Handle the case with no valid importances
            if importance_df.empty or importance_df['Importance'].isna().all():
                col.warning("No valid feature importance values available")
                return
                
            # Sort by absolute importance for better visualization
            importance_df['Abs_Importance'] = importance_df['Importance'].abs()
            importance_df = importance_df.sort_values('Abs_Importance', ascending=False).drop(columns=['Abs_Importance'])
            
            # Plot using plotly
            fig = px.bar(importance_df, 
                         x='Importance', 
                         y='Feature',
                         orientation='h',
                         title='Feature Importance',
                         color='Importance',
                         color_continuous_scale=['red', 'lightgray', 'blue'])
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            col.plotly_chart(fig, use_container_width=True)
        
        return

    # Create tabs for different visualizations within the column
    tabs = col.tabs(["SHAP Force", "SHAP Summary", "LIME", "Feature Importance"])

    # Display SHAP Force Plot
    with tabs[0]:
        if "visualizations" in response_data and "shap_force" in response_data["visualizations"]:
            try:
                img_data = base64.b64decode(response_data["visualizations"]["shap_force"])
                st.image(img_data, caption="SHAP Force Plot", use_column_width=True)
            except Exception as e:
                st.error(f"Error displaying SHAP Force Plot: {str(e)}")
                st.warning("SHAP Force Plot could not be displayed")
        else:
            st.warning("SHAP Force Plot not available")

    # Display SHAP Summary Plot
    with tabs[1]:
        if "visualizations" in response_data and "shap_summary" in response_data["visualizations"]:
            try:
                img_data = base64.b64decode(response_data["visualizations"]["shap_summary"])
                st.image(img_data, caption="SHAP Summary Plot", use_column_width=True)
            except Exception as e:
                st.error(f"Error displaying SHAP Summary Plot: {str(e)}")
                st.warning("SHAP Summary Plot could not be displayed")
        else:
            st.warning("SHAP Summary Plot not available")

    # Display LIME Plot
    with tabs[2]:
        if "visualizations" in response_data and "lime" in response_data["visualizations"]:
            try:
                img_data = base64.b64decode(response_data["visualizations"]["lime"])
                st.image(img_data, caption="LIME Explanation Plot", use_column_width=True)
            except Exception as e:
                st.error(f"Error displaying LIME Plot: {str(e)}")
                st.warning("LIME Plot could not be displayed")
                
                # Retry loading LIME visualization
                if st.button("Retry LIME Visualization"):
                    st.info("This would normally trigger a new API call for LIME explanation")
        else:
            st.warning("LIME Plot not available")

    # Display Feature Importance
    with tabs[3]:
        if "feature_importance" in response_data:
            # Create a DataFrame for better visualization
            importance_df = pd.DataFrame({
                'Feature': list(response_data["feature_importance"].keys()),
                'Importance': list(response_data["feature_importance"].values())
            })
            
            # Handle the case with no valid importances
            if importance_df.empty or importance_df['Importance'].isna().all():
                st.warning("No valid feature importance values available")
                return
                
            # Sort by absolute importance for better visualization
            importance_df['Abs_Importance'] = importance_df['Importance'].abs()
            importance_df = importance_df.sort_values('Abs_Importance', ascending=False).drop(columns=['Abs_Importance'])
            
            # Plot using plotly
            fig = px.bar(importance_df, 
                         x='Importance', 
                         y='Feature',
                         orientation='h',
                         title='Feature Importance',
                         color='Importance',
                         color_continuous_scale=['red', 'lightgray', 'blue'])
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature importance data not available")

@st.cache_data(ttl=300)
def load_model_hyperparameters():
    """Load model hyperparameters from training results"""
    try:
        # Try to load the evaluation results file
        results_path = os.path.join(MODELS_DIR, "evaluation_results.pkl")
        if os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                return pickle.load(f)
        return None
    except Exception as e:
        st.error(f"Error loading model hyperparameters: {e}")
        return None

def create_input_form():
    """Create a compact input form for credit data"""
    # Initialize session state if not exists
    if 'revolving_utilization' not in st.session_state:
        st.session_state.revolving_utilization = 0.5
    if 'debt_ratio' not in st.session_state:
        st.session_state.debt_ratio = 0.3
    if 'monthly_income' not in st.session_state:
        st.session_state.monthly_income = 5000
    if 'num_open_credit_lines' not in st.session_state:
        st.session_state.num_open_credit_lines = 8
    if 'num_real_estate_loans' not in st.session_state:
        st.session_state.num_real_estate_loans = 1
    if 'num_times_30_59_days_late' not in st.session_state:
        st.session_state.num_times_30_59_days_late = 0
    if 'num_times_60_89_days_late' not in st.session_state:
        st.session_state.num_times_60_89_days_late = 0
    if 'num_times_90_days_late' not in st.session_state:
        st.session_state.num_times_90_days_late = 0
    if 'age' not in st.session_state:
        st.session_state.age = 40
    if 'num_dependents' not in st.session_state:
        st.session_state.num_dependents = 0
        
    # Create compact form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Financial Details")
            
            revolving_utilization = st.slider(
                "Revolving Utilization",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.revolving_utilization,
                step=0.01,
                help="Total balance on credit cards and personal lines of credit divided by the sum of credit limits",
                key="form_revolving_utilization"
            )
            
            debt_ratio = st.slider(
                "Debt Ratio",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.debt_ratio,
                step=0.01,
                help="Monthly debt payments, alimony, living costs divided by monthly gross income",
                key="form_debt_ratio"
            )
            
            monthly_income = st.number_input(
                "Monthly Income ($)",
                min_value=0,
                max_value=100000,
                value=st.session_state.monthly_income,
                step=100,
                help="Monthly income",
                key="form_monthly_income"
            )
            
            # Personal information
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=st.session_state.age,
                help="Age of borrower in years",
                key="form_age"
            )
            
            num_dependents = st.number_input(
                "Dependents",
                min_value=0,
                max_value=10,
                value=st.session_state.num_dependents,
                help="Number of dependents in family",
                key="form_num_dependents"
            )
        
        with col2:
            st.subheader("Credit History")
            
            # Credit history with 2-column layout
            num_open_credit_lines = st.number_input(
                "Open Credit Lines",
                min_value=0,
                max_value=30,
                value=st.session_state.num_open_credit_lines,
                help="Number of open loans and lines of credit",
                key="form_num_open_credit_lines"
            )
            
            num_real_estate_loans = st.number_input(
                "Real Estate Loans",
                min_value=0,
                max_value=10,
                value=st.session_state.num_real_estate_loans,
                help="Number of mortgage and real estate loans",
                key="form_num_real_estate_loans"
            )
            
            st.subheader("Payment History")
            # Past due history in a 3-column layout
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                num_times_30_59_days_late = st.number_input(
                    "30-59 Days Late",
                    min_value=0,
                    max_value=10,
                    value=st.session_state.num_times_30_59_days_late,
                    help="Number of times 30-59 days past due",
                    key="form_num_times_30_59_days_late"
                )
            with col_b:
                num_times_60_89_days_late = st.number_input(
                    "60-89 Days Late",
                    min_value=0,
                    max_value=10,
                    value=st.session_state.num_times_60_89_days_late,
                    help="Number of times 60-89 days past due",
                    key="form_num_times_60_89_days_late"
                )
            with col_c:
                num_times_90_days_late = st.number_input(
                    "90+ Days Late",
                    min_value=0,
                    max_value=10,
                    value=st.session_state.num_times_90_days_late,
                    help="Number of times 90+ days past due",
                    key="form_num_times_90_days_late"
                )
        
        # Model selection in its own row
        st.subheader("Select Prediction Model")
        available_models = get_available_models()
        
        if available_models:
            model_options = available_models + ["Compare All Models"]
            selected_model = st.selectbox(
                "Model",
                options=model_options,
                index=0 if available_models else 0,
                key="form_selected_model"
            )
        else:
            st.error("No models available. Please check API connection.")
            return None, None, False
        
        # Submit button - centered and prominent
        submitted = st.form_submit_button("Predict Risk", use_container_width=True)
        
        # Prepare input data
        input_data = {
            "RevolvingUtilizationOfUnsecuredLines": revolving_utilization,
            "Age": age,
            "NumberOfTime30_59DaysPastDueNotWorse": num_times_30_59_days_late,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": monthly_income,
            "NumberOfOpenCreditLinesAndLoans": num_open_credit_lines,
            "NumberOfTimes90DaysLate": num_times_90_days_late,
            "NumberRealEstateLoansOrLines": num_real_estate_loans,
            "NumberOfTime60_89DaysPastDueNotWorse": num_times_60_89_days_late,
            "NumberOfDependents": num_dependents
        }
        
        # Update session state with form values if submitted
        if submitted:
            st.session_state.revolving_utilization = revolving_utilization
            st.session_state.debt_ratio = debt_ratio
            st.session_state.monthly_income = monthly_income
            st.session_state.num_open_credit_lines = num_open_credit_lines
            st.session_state.num_real_estate_loans = num_real_estate_loans
            st.session_state.num_times_30_59_days_late = num_times_30_59_days_late
            st.session_state.num_times_60_89_days_late = num_times_60_89_days_late
            st.session_state.num_times_90_days_late = num_times_90_days_late
            st.session_state.age = age
            st.session_state.num_dependents = num_dependents
        
        return input_data, selected_model, submitted

def predictor_tab():
    """Prediction tab with input form and results display"""
    st.header("Credit Default Risk Prediction")
    
    # Create a collapsible expander for the input form
    with st.expander("Customer Information Input", expanded=False):
        st.markdown("Enter customer information to predict credit default risk.")
        # Create the input form
        input_data, selected_model, submitted = create_input_form()
    
    # Initialize session state for prediction results if it doesn't exist
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    if 'prediction_explanation' not in st.session_state:
        st.session_state.prediction_explanation = None
    if 'prediction_model' not in st.session_state:
        st.session_state.prediction_model = None
    
    # If new submission, clear previous results
    if submitted:
        st.session_state.prediction_results = None
        st.session_state.prediction_explanation = None
        st.session_state.prediction_model = None
    
    # Process submission or use cached results
    if submitted and input_data and selected_model:
        # Show progress bar
        progress_bar = st.progress(0)
        
        # Display results in a container
        results_container = st.container()
        
        with st.spinner("Processing prediction..."):
            progress_bar.progress(25)
            time.sleep(0.1)  # Small delay for UI feedback
            
            if selected_model == "Compare All Models":
                # Compare all models
                progress_bar.progress(50)
                results = compare_models(input_data)
                progress_bar.progress(75)
                time.sleep(0.1)  # Small delay for UI feedback
                
                if results:
                    progress_bar.progress(100)
                    st.session_state.prediction_results = results
                    st.session_state.prediction_model = "Compare All Models"
                    
                    with results_container:
                        display_model_comparison(results)
                else:
                    progress_bar.progress(100)
                    results_container.error("Failed to get model comparisons")
            else:
                # Single model prediction
                progress_bar.progress(50)
                result = predict_risk(input_data, selected_model)
                progress_bar.progress(60)
                
                if result:
                    # Get explanation with longer timeout
                    with st.spinner("Generating explanations (this may take a moment)..."):
                        explanation = explain_prediction(input_data, selected_model)
                    
                    progress_bar.progress(90)
                    time.sleep(0.1)
                    progress_bar.progress(100)
                    
                    # Store results in session state for reuse
                    st.session_state.prediction_results = result
                    st.session_state.prediction_explanation = explanation
                    st.session_state.prediction_model = selected_model
                    
                    with results_container:
                        display_prediction_results(result, explanation, selected_model)
                else:
                    progress_bar.progress(100)
                    results_container.error("Failed to get prediction result")
    # If we have cached results, display them
    elif st.session_state.prediction_results is not None:
        if st.session_state.prediction_model == "Compare All Models":
            display_model_comparison(st.session_state.prediction_results)
        else:
            display_prediction_results(
                st.session_state.prediction_results,
                st.session_state.prediction_explanation,
                st.session_state.prediction_model
            )
    else:
        # No results yet, show instructions
        st.info("👈 Click the expander above to input customer information and make a prediction")

def display_model_comparison(results):
    """Display model comparison results"""
    st.success("Prediction Complete!")
    st.subheader("Model Comparison")
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, result in results.items():
        # Check if result has the expected keys
        if result and "probability" in result and "risk_level" in result and "prediction" in result:
            comparison_data.append({
                "Model": model_name,
                "Probability": result["probability"] if result["probability"] is not None else 0,
                "Risk Level": result["risk_level"],
                "Default Prediction": "Yes" if result["prediction"] == 1 else "No"
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create two columns for table and chart
        col1, col2 = st.columns(2)
        
        # Display as table
        with col1:
            st.subheader("Results Table")
            st.dataframe(comparison_df, use_container_width=True)
        
        # Create bar chart
        with col2:
            st.subheader("Risk Probability by Model")
            fig = px.bar(
                comparison_df,
                x="Model",
                y="Probability",
                color="Risk Level",
                text="Probability",
                color_discrete_map={
                    "Very Low Risk": "green",
                    "Low Risk": "lightgreen",
                    "Medium Risk": "yellow", 
                    "High Risk": "orange",
                    "Very High Risk": "red",
                    "Error": "gray"
                }
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(
                yaxis_title="Default Probability",
                yaxis=dict(range=[0, 1]),
                xaxis_title="",
                legend_title="Risk Level"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No valid comparison data available")

def display_prediction_results(result, explanation, selected_model):
    """Display prediction results and explanations"""
    st.success("Prediction Complete!")
    
    # Create two columns for results and explanations
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Prediction Results")
        
        # Display probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["probability"],
            number={"suffix": "", "font": {"size": 24}},
            title={"text": "Default Risk Probability"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "darkgray"},
                "steps": [
                    {"range": [0, 0.2], "color": "green"},
                    {"range": [0.2, 0.4], "color": "lightgreen"},
                    {"range": [0.4, 0.6], "color": "yellow"},
                    {"range": [0.6, 0.8], "color": "orange"},
                    {"range": [0.8, 1], "color": "red"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 0.5
                }
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display prediction result
        prediction_text = "High risk of default" if result["prediction"] == 1 else "Low risk of default"
        prediction_color = "red" if result["prediction"] == 1 else "green"
        
        st.markdown(f"""
        <div style="padding: 10px; background-color: {prediction_color}; color: white; border-radius: 5px; text-align: center; margin-bottom: 20px;">
            <h3>{prediction_text}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-bottom: 10px;">
            <b>Risk Level:</b> {result["risk_level"]}
        </div>
        <div style="padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-bottom: 10px;">
            <b>Model Used:</b> {result["model"]}
        </div>
        """, unsafe_allow_html=True)
        
        # Display model information including hyperparameters
        with st.expander("Model Information", expanded=False):
            st.subheader(f"Model: {selected_model}")
            
            # Try to load hyperparameters
            hyperparams = load_model_hyperparameters()
            if hyperparams and selected_model in hyperparams:
                model_data = hyperparams[selected_model]
                
                if "best_params" in model_data:
                    st.subheader("Model Hyperparameters")
                    params_df = pd.DataFrame({
                        "Parameter": list(model_data["best_params"].keys()),
                        "Value": list(model_data["best_params"].values())
                    })
                    st.dataframe(params_df, use_container_width=True)
                
                if "metrics" in model_data:
                    st.subheader("Training Metrics")
                    metrics_df = pd.DataFrame({
                        "Metric": list(model_data["metrics"].keys()),
                        "Value": list(model_data["metrics"].values())
                    })
                    # Filter out non-numeric metrics like confusion matrix
                    metrics_df = metrics_df[~metrics_df["Metric"].isin(["confusion_matrix", "classification_report"])]
                    st.dataframe(metrics_df, use_container_width=True)
    
    # Display explanation in the second column
    with col2:
        if explanation:
            st.subheader("Model Explanations")
            
            # Check if visualizations are available
            has_visualizations = "visualizations" in explanation and any([
                "shap_force" in explanation["visualizations"],
                "shap_summary" in explanation["visualizations"],
                "lime" in explanation["visualizations"]
            ])
            
            visualization_tabs = st.tabs(["SHAP Force", "SHAP Summary", "LIME", "Feature Importance"])
            
            # Display SHAP Force Plot
            with visualization_tabs[0]:
                if has_visualizations and "shap_force" in explanation["visualizations"]:
                    try:
                        img_data = base64.b64decode(explanation["visualizations"]["shap_force"])
                        st.image(img_data, caption="SHAP Force Plot", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error displaying SHAP Force Plot: {str(e)}")
                        # Try to show raw data as fallback
                        if "feature_importance" in explanation:
                            display_fallback_feature_importance(explanation["feature_importance"], "SHAP Values")
                else:
                    st.warning("SHAP Force Plot not available")
                    # Show feature importance as fallback
                    if "feature_importance" in explanation:
                        display_fallback_feature_importance(explanation["feature_importance"], "SHAP Values")
            
            # Display SHAP Summary Plot
            with visualization_tabs[1]:
                if has_visualizations and "shap_summary" in explanation["visualizations"]:
                    try:
                        img_data = base64.b64decode(explanation["visualizations"]["shap_summary"])
                        st.image(img_data, caption="SHAP Summary Plot", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error displaying SHAP Summary Plot: {str(e)}")
                        # Try to show raw data as fallback
                        if "feature_importance" in explanation:
                            display_fallback_feature_importance(explanation["feature_importance"], "SHAP Values")
                else:
                    st.warning("SHAP Summary Plot not available")
                    # Show feature importance as fallback
                    if "feature_importance" in explanation:
                        display_fallback_feature_importance(explanation["feature_importance"], "SHAP Values")
            
            # Display LIME Plot
            with visualization_tabs[2]:
                if has_visualizations and "lime" in explanation["visualizations"]:
                    try:
                        img_data = base64.b64decode(explanation["visualizations"]["lime"])
                        st.image(img_data, caption="LIME Explanation Plot", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error displaying LIME Plot: {str(e)}")
                        # Try to show raw data as fallback
                        if "lime_importance" in explanation and explanation["lime_importance"]:
                            display_fallback_feature_importance(explanation["lime_importance"], "LIME Values")
                else:
                    st.warning("LIME Plot not available")
                    # Show feature importance as fallback
                    if "lime_importance" in explanation and explanation["lime_importance"]:
                        display_fallback_feature_importance(explanation["lime_importance"], "LIME Values")
            
            # Display Feature Importance
            with visualization_tabs[3]:
                if "feature_importance" in explanation:
                    display_fallback_feature_importance(explanation["feature_importance"], "Feature Importance")
                else:
                    st.warning("Feature importance data not available")
            
            # Display debug info with raw visualization data if there are issues
            with st.expander("Debug Visualization Info", expanded=False):
                st.subheader("Visualization Debug Information")
                if "visualizations" in explanation:
                    st.write("Available visualization keys:", list(explanation["visualizations"].keys()))
                    # Show the first 100 chars of each visualization to check if valid
                    for vis_key in explanation["visualizations"]:
                        vis_data = explanation["visualizations"][vis_key]
                        if vis_data:
                            st.write(f"{vis_key} data preview: {vis_data[:100]}...")
                else:
                    st.write("No visualizations available in response")
        else:
            st.warning("Could not generate explanations for this prediction")
            
            # Show a potential fix to the user
            if st.button("Retry Explanation Generation"):
                with st.spinner("Retrying explanation generation..."):
                    explanation = explain_prediction(input_data, selected_model)
                if explanation:
                    st.session_state.prediction_explanation = explanation
                    st.experimental_rerun()  # Rerun the app to refresh with new data
                else:
                    st.error("Still unable to generate explanations")

def display_fallback_feature_importance(importance_dict, title="Feature Importance"):
    """Display feature importance as a fallback when visualizations fail"""
    if not importance_dict:
        st.warning("No feature importance data available")
        return
        
    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': list(importance_dict.keys()),
        'Importance': list(importance_dict.values())
    })
    
    # Handle the case with no valid importances
    if importance_df.empty or importance_df['Importance'].isna().all():
        st.warning("No valid feature importance values available")
        return
        
    # Sort by absolute importance for better visualization
    importance_df['Abs_Importance'] = importance_df['Importance'].abs()
    importance_df = importance_df.sort_values('Abs_Importance', ascending=False).drop(columns=['Abs_Importance'])
    
    # Plot using plotly
    fig = px.bar(importance_df, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title=title,
                color='Importance',
                color_continuous_scale=['red', 'lightgray', 'blue'])
    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

def performance_metrics_tab():
    """Tab to display model performance metrics"""
    st.header("Model Performance Metrics")
    st.markdown("Compare metrics and performance across different models.")
    
    # Load model evaluation results
    evaluation_results = load_model_hyperparameters()
    
    if not evaluation_results:
        st.error("No model evaluation results available.")
        return
    
    # Display saved performance plot if available
    performance_plot_path = os.path.join(MODELS_DIR, "model_performance.png")
    if os.path.exists(performance_plot_path):
        st.subheader("Model Performance Comparison")
        st.image(performance_plot_path, use_column_width=True)
    
    # Create tabs for metrics, hyperparameters
    metrics_tab, hyperparams_tab = st.tabs(["Model Metrics", "Hyperparameters"])
    
    with metrics_tab:
        # Create a consolidated metrics table
        metrics_data = []
        metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        
        for model_name, results in evaluation_results.items():
            if "metrics" in results:
                metrics = results["metrics"]
                model_metrics = {"Model": model_name}
                for key in metric_keys:
                    if key in metrics:
                        model_metrics[key.replace("_", " ").title()] = round(metrics[key], 4)
                metrics_data.append(model_metrics)
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Create radar chart for model comparison
            fig = go.Figure()
            
            for i, row in metrics_df.iterrows():
                model = row["Model"]
                values = [row[col] for col in metrics_df.columns if col != "Model"]
                categories = [col for col in metrics_df.columns if col != "Model"]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=model
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Model Performance Comparison",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No metrics data available")
    
    with hyperparams_tab:
        # Select a model to see its hyperparameters
        model_options = list(evaluation_results.keys())
        selected_model = st.selectbox("Select Model", model_options)
        
        if selected_model in evaluation_results and "best_params" in evaluation_results[selected_model]:
            st.subheader(f"Hyperparameters for {selected_model}")
            params = evaluation_results[selected_model]["best_params"]
            params_df = pd.DataFrame({
                "Parameter": list(params.keys()),
                "Value": list(params.values())
            })
            st.dataframe(params_df, use_container_width=True)
        else:
            st.warning(f"No hyperparameters available for {selected_model}")

# Main app with tabs
tab1, tab2 = st.tabs(["Predict Credit Risk", "Model Performance"])

with tab1:
    predictor_tab()

with tab2:
    performance_metrics_tab()

# Footer with copyright notice
st.markdown("""
---
### Credit Default Risk Prediction System | Powered by Docker Swarm
""") 