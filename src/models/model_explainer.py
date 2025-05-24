import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
import joblib
import os
import base64
import io
from sklearn.utils import resample
from typing import List, Dict, Any, Tuple, Union, Optional

def explain_with_shap(
    model, 
    X: np.ndarray, 
    feature_names: List[str] = None, 
    class_names: List[str] = None,
    sample_idx: int = 0
) -> Dict[str, float]:
    """
    Generate SHAP explanations for a model prediction
    
    Parameters
    ----------
    model : trained model with predict_proba method
        The machine learning model to explain
    X : numpy.ndarray
        Feature array for the instances to explain
    feature_names : list of str, optional
        Names of features (used for visualization)
    class_names : list of str, optional
        Names of classes (used for visualization)
    sample_idx : int, default=0
        Index of the sample to explain
        
    Returns
    -------
    dict
        Dictionary mapping feature names to SHAP values
    """
    # Determine model type and use appropriate explainer
    is_tree_model = hasattr(model, 'tree_') or type(model).__name__ in [
        'XGBClassifier', 'XGBRegressor',
        'LGBMClassifier', 'LGBMRegressor',
        'RandomForestClassifier', 'RandomForestRegressor'
    ]
    
    try:
        # Convert X to numpy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X_values = X.values
        else:
            X_values = X
            
        # Ensure we have proper feature names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_values.shape[1])]
            
        if is_tree_model:
            explainer = shap.TreeExplainer(model)
        else:
            # For non-tree models
            explainer = shap.KernelExplainer(model.predict_proba, X_values)
        
        # Get SHAP values
        shap_values = explainer.shap_values(X_values)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # For binary classification, use the positive class (index 1)
            shap_values_to_use = shap_values[1]
        else:
            shap_values_to_use = shap_values
        
        # Create a dictionary mapping feature names to SHAP values
        # Ensure we only use the specified sample index
        shap_dict = {
            feature: float(value) 
            for feature, value in zip(feature_names, shap_values_to_use[sample_idx])
        }
        
        return shap_dict
    except Exception as e:
        print(f"SHAP explanation error: {str(e)}")
        return {}

def explain_with_lime(
    model, 
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    feature_names: List[str] = None, 
    class_names: List[str] = None,
    sample_idx: int = 0, 
    num_features: int = 10
) -> Dict[str, float]:
    """
    Generate LIME explanations for a model prediction
    
    Parameters
    ----------
    model : trained model with predict_proba method
        The machine learning model to explain
    X_train : numpy.ndarray
        Training data for LIME to learn the locality
    X_test : numpy.ndarray
        Test instances to explain
    feature_names : list of str, optional
        Names of features (used for visualization)
    class_names : list of str, optional
        Names of classes (used for visualization)
    sample_idx : int, default=0
        Index of the sample to explain
    num_features : int, default=10
        Maximum number of features to include in explanation
        
    Returns
    -------
    dict
        Dictionary mapping feature names to importance values
    """
    try:
        # Convert to numpy arrays if they're DataFrames
        if isinstance(X_train, pd.DataFrame):
            if feature_names is None:
                feature_names = X_train.columns.tolist()
            X_train_values = X_train.values
        else:
            X_train_values = X_train
            
        if isinstance(X_test, pd.DataFrame):
            X_test_values = X_test.values
        else:
            X_test_values = X_test
        
        # Set feature names and class names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_train_values.shape[1])]
        
        if class_names is None:
            class_names = ["Class 0", "Class 1"]
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train_values,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            discretize_continuous=True,
            verbose=False
        )
        
        # Get prediction function
        if hasattr(model, 'predict_proba'):
            predict_fn = model.predict_proba
        else:
            # Create a wrapper function for models without predict_proba
            def predict_fn(x):
                preds = model.predict(x)
                return np.vstack([(1-preds), preds]).T
        
        # Generate explanation
        explanation = explainer.explain_instance(
            X_test_values[sample_idx],
            predict_fn,
            num_features=min(num_features, X_train_values.shape[1]),
            top_labels=1
        )
        
        # Convert explanation to dictionary
        if explanation:
            # Get explanation for the positive class (index 1)
            exp_list = explanation.as_list(label=1)
            return {feature_name: float(importance) for feature_name, importance in exp_list}
        else:
            return {}
    except Exception as e:
        print(f"LIME explanation error: {str(e)}")
        return {}

def create_shap_plot(
    model,
    X: np.ndarray, 
    feature_names: List[str] = None,
    plot_type: str = "force",
    sample_idx: int = 0
) -> str:
    """
    Create a SHAP plot and return it as a base64-encoded image
    
    Parameters
    ----------
    model : trained model
        The model to explain
    X : numpy.ndarray or pd.DataFrame
        Data for explanation
    feature_names : list of str, optional
        Names of the features
    plot_type : str, default="force"
        Type of plot to create: "force", "summary", "dependence"
    sample_idx : int, default=0
        Index of the sample to explain
    
    Returns
    -------
    str
        Base64-encoded string of the plot image
    """
    try:
        # Convert X to numpy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X_values = X.values
        else:
            X_values = X
            
        # Make sure X is not empty
        if X_values is None or X_values.shape[0] == 0 or X_values.shape[1] == 0:
            print(f"Invalid data shape: {X_values.shape if X_values is not None else 'None'}")
            return None
            
        # Ensure we have valid feature names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_values.shape[1])]
            
        # Make sure sample_idx is valid
        if sample_idx >= X_values.shape[0]:
            print(f"Sample index {sample_idx} out of range, using first sample")
            sample_idx = 0
            
        # Use only the selected sample for individual plots
        X_to_explain = X_values[sample_idx:sample_idx+1] if plot_type == "force" else X_values
            
        # Check if model has predict_proba method
        predict_proba_fn = None
        if hasattr(model, 'predict_proba'):
            predict_proba_fn = model.predict_proba
            print("Model has predict_proba method")
        # Create a wrapper if model only has predict
        elif hasattr(model, 'predict'):
            def predict_proba_wrapper(X):
                predictions = model.predict(X)
                return np.column_stack((1-predictions, predictions))
            predict_proba_fn = predict_proba_wrapper
            print("Created predict_proba wrapper")
        else:
            print("Model has no prediction method")
            return None
        
        # Determine model type and use appropriate explainer
        is_tree_model = hasattr(model, 'tree_') or type(model).__name__ in [
            'XGBClassifier', 'XGBRegressor',
            'LGBMClassifier', 'LGBMRegressor',
            'RandomForestClassifier', 'RandomForestRegressor'
        ]
        
        print(f"Creating {type(model).__name__} explainer...")
        if is_tree_model:
            explainer = shap.TreeExplainer(model)
        else:
            # For non-tree models, use a small subset of data for background
            background = X_values[:min(100, X_values.shape[0])]
            try:
                explainer = shap.KernelExplainer(predict_proba_fn, background)
            except Exception as e:
                print(f"Error creating KernelExplainer: {str(e)}")
                # Try a more direct approach - just use the model's predict method
                try:
                    explainer = shap.Explainer(model, background)
                except Exception as e2:
                    print(f"Error creating Explainer: {str(e2)}")
                    return create_fallback_plot(model, X_values, feature_names, sample_idx, 
                                              error_message=f"Error creating SHAP explainer: {str(e2)}")
    
        # Get SHAP values
        print(f"Calculating SHAP values for {plot_type} plot...")
        try:
            shap_values = explainer.shap_values(X_to_explain)
        except Exception as sv_error:
            print(f"Error calculating SHAP values: {str(sv_error)}")
            return create_fallback_plot(model, X_values, feature_names, sample_idx, 
                                      error_message=f"Error calculating SHAP values: {str(sv_error)}")
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # For binary classification, use the positive class (index 1)
            shap_values_to_plot = shap_values[1]
            expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
        else:
            shap_values_to_plot = shap_values
            expected_value = explainer.expected_value
        
        # Create appropriate plot
        plt.figure(figsize=(10, 6))
        
        try:
            print(f"Creating {plot_type} plot...")
            if plot_type == "force":
                # Try the newer API first
                try:
                    plt.title('SHAP Force Plot')
                    # Use the updated SHAP API for force plot
                    shap.plots.force(
                        explainer.expected_value, 
                        shap_values_to_plot[0], 
                        features=X_to_explain[0],
                        feature_names=feature_names,
                        matplotlib=True,
                        show=False
                    )
                except Exception as e:
                    print(f"Error with SHAP force plot: {str(e)}")
                    # Try alternate API for older versions
                    try:
                        shap.force_plot(
                            explainer.expected_value, 
                            shap_values_to_plot[0], 
                            features=X_to_explain[0],
                            feature_names=feature_names,
                            matplotlib=True,
                            show=False
                        )
                    except Exception:
                        # Create a simpler bar chart as fallback
                        plt.title('Feature Contribution (SHAP Values)')
                        importance = shap_values_to_plot[0]
                        # Convert to Python list to avoid indexing issues
                        importance_list = importance.tolist() if hasattr(importance, 'tolist') else list(importance)
                        feature_names_list = list(feature_names)
                        
                        # Create indices sorted by absolute importance
                        try:
                            sorted_indices = sorted(range(len(importance_list)), key=lambda i: abs(importance_list[i]) if not isinstance(importance_list[i], list) else abs(importance_list[i][0]), reverse=True)
                            # Take top 10 or fewer
                            top_n = min(10, len(sorted_indices))
                            top_indices = sorted_indices[:top_n]
                            
                            plt.barh(
                                [feature_names_list[i] for i in top_indices],
                                [importance_list[i] if not isinstance(importance_list[i], list) else importance_list[i][0] for i in top_indices]
                            )
                            plt.xlabel('SHAP Value (Impact on Prediction)')
                        except Exception as sort_error:
                            print(f"Error sorting importance values: {sort_error}")
                            # Ultra fallback - just plot raw values
                            plt.barh(range(min(10, len(feature_names_list))), 
                                    [0] * min(10, len(feature_names_list)))
                            plt.yticks(range(min(10, len(feature_names_list))), 
                                    feature_names_list[:min(10, len(feature_names_list))])
            
            elif plot_type == "summary":
                # Try summary plot - simpler and more reliable
                try:
                    shap.summary_plot(
                        shap_values_to_plot,
                        X_to_explain,
                        feature_names=feature_names,
                        plot_type="bar",
                        show=False
                    )
                except Exception as e:
                    print(f"Error with SHAP summary plot: {str(e)}")
                    # Create simple bar chart as fallback
                    plt.title('Feature Importance')
                    importance = np.abs(shap_values_to_plot).mean(axis=0)
                    
                    # Convert to Python list to avoid indexing issues
                    importance_list = importance.tolist() if hasattr(importance, 'tolist') else list(importance)
                    feature_names_list = list(feature_names)
                    
                    # Create indices sorted by absolute importance
                    sorted_indices = sorted(range(len(importance_list)), key=lambda i: importance_list[i], reverse=True)
                    # Take top 10 or fewer
                    top_n = min(10, len(sorted_indices))
                    top_indices = sorted_indices[:top_n]
                    
                    plt.barh(
                        [feature_names_list[i] for i in top_indices],
                        [importance_list[i] for i in top_indices]
                    )
            
            elif plot_type == "dependence":
                # Only attempt if we have features
                if feature_names and len(feature_names) > 0:
                    # Get the most important feature
                    feature_importance = np.abs(shap_values_to_plot).mean(axis=0)
                    # Convert to list and find index of max
                    feature_importance_list = feature_importance.tolist() if hasattr(feature_importance, 'tolist') else list(feature_importance)
                    most_important_feature = feature_importance_list.index(max(feature_importance_list))
                    
                    try:
                        shap.dependence_plot(
                            most_important_feature, 
                            shap_values_to_plot, 
                            X_to_explain,
                            feature_names=feature_names, 
                            show=False
                        )
                    except Exception as e:
                        print(f"Error with SHAP dependence plot: {str(e)}")
                        # Simple scatter plot as fallback
                        plt.title(f'Dependence Plot for {feature_names[most_important_feature]}')
                        plt.scatter(
                            X_to_explain[:, most_important_feature], 
                            shap_values_to_plot[:, most_important_feature]
                        )
                        plt.xlabel(feature_names[most_important_feature])
                        plt.ylabel('SHAP Value')
            
        except Exception as plot_error:
            print(f"Error creating plot: {str(plot_error)}")
            # Create a simpler visualization as fallback
            if plot_type in ["force", "summary"]:
                plt.title(f'Feature Importance for {plot_type.title()} Plot')
                
                # Handle different data types safely
                if plot_type == "force" and shap_values_to_plot.shape[0] == 1:
                    importance = np.abs(shap_values_to_plot[0])
                else:
                    importance = np.abs(shap_values_to_plot).mean(axis=0) 
                    
                # Convert to Python list to avoid indexing issues
                importance_list = importance.tolist() if hasattr(importance, 'tolist') else list(importance)
                feature_names_list = list(feature_names)
                
                # Create indices sorted by absolute importance
                sorted_indices = sorted(range(len(importance_list)), key=lambda i: importance_list[i], reverse=True)
                # Take top 10 or fewer
                top_n = min(10, len(sorted_indices))
                top_indices = sorted_indices[:top_n]
                
                plt.barh(
                    [feature_names_list[i] for i in top_indices],
                    [importance_list[i] for i in top_indices]
                )
        
        # Save plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    except Exception as e:
        print(f"Error creating SHAP plot: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return create_fallback_plot(model, X, feature_names, sample_idx, error_message=f"Error creating SHAP plot: {str(e)}")

def create_fallback_plot(model, X, feature_names=None, sample_idx=0, error_message=None):
    """Create a simple fallback plot when SHAP or LIME fails"""
    try:
        print("Creating fallback plot...")
        plt.figure(figsize=(10, 6))
        
        if error_message:
            plt.text(0.5, 0.95, error_message, ha='center', va='top', wrap=True, fontsize=12, color='red')
        
        # Convert X to numpy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X_values = X.values
        else:
            X_values = X
            
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_values.shape[1])]
        
        # Extract important features if possible
        try:
            # Check if model has feature_importances_
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                title = "Model Feature Importances"
            # Try coef_ for linear models
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if len(coef.shape) > 1:
                    importances = coef[0]  # For binary classification
                else:
                    importances = coef
                title = "Model Coefficients"
            # Use feature values as a last resort
            else:
                sample = X_values[sample_idx] if sample_idx < X_values.shape[0] else X_values[0]
                importances = np.abs(sample)
                title = "Feature Values"
                
            # Convert to list
            importances_list = importances.tolist() if hasattr(importances, 'tolist') else list(importances)
            
            # Sort by absolute importance
            sorted_indices = sorted(range(len(importances_list)), key=lambda i: abs(importances_list[i]), reverse=True)
            top_n = min(10, len(sorted_indices))
            top_indices = sorted_indices[:top_n]
            
            plt.barh(
                [feature_names[i] for i in top_indices],
                [float(importances_list[i]) for i in top_indices]
            )
            plt.title(f"{title} (Fallback Visualization)")
            plt.xlabel("Importance")
            
        except Exception as e:
            print(f"Error creating feature importance plot: {str(e)}")
            plt.text(0.5, 0.5, "Could not generate feature importance plot", 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
        
        # Save plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error creating fallback plot: {str(e)}")
        try:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Visualization Failed: {str(e)}", 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except:
            return None

def create_lime_plot(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str] = None,
    class_names: List[str] = None,
    sample_idx: int = 0
) -> str:
    """
    Create a LIME explanation plot and return it as a base64-encoded image
    
    Parameters
    ----------
    model : trained model
        The model to explain
    X_train : numpy.ndarray or pd.DataFrame
        Training data for LIME to learn the locality
    X_test : numpy.ndarray or pd.DataFrame
        Test instances to explain
    feature_names : list of str, optional
        Names of features
    class_names : list of str, optional
        Names of classes
    sample_idx : int, default=0
        Index of the sample to explain
    
    Returns
    -------
    str
        Base64-encoded string of the plot image
    """
    try:
        print(f"Starting LIME explanation for sample {sample_idx}")
        print(f"X_train type: {type(X_train)}, X_test type: {type(X_test)}")
        
        # Convert to numpy arrays if they're DataFrames
        if isinstance(X_train, pd.DataFrame):
            if feature_names is None:
                feature_names = X_train.columns.tolist()
                print(f"Using DataFrame columns as feature names: {feature_names[:5]}...")
            X_train_values = X_train.values
        else:
            X_train_values = X_train
            
        if isinstance(X_test, pd.DataFrame):
            X_test_values = X_test.values
        else:
            X_test_values = X_test
        
        print(f"X_train_values shape: {X_train_values.shape}, X_test_values shape: {X_test_values.shape}")
        
        # Data validation
        if X_train_values is None or X_train_values.shape[0] == 0 or X_train_values.shape[1] == 0:
            print(f"Invalid training data shape: {X_train_values.shape if X_train_values is not None else 'None'}")
            return None
            
        if X_test_values is None or X_test_values.shape[0] == 0 or X_test_values.shape[1] == 0:
            print(f"Invalid test data shape: {X_test_values.shape if X_test_values is not None else 'None'}")
            return None
            
        if sample_idx >= X_test_values.shape[0]:
            print(f"Sample index {sample_idx} out of range for test data with shape {X_test_values.shape}")
            sample_idx = 0
        
        # Set feature names and class names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_train_values.shape[1])]
            print(f"Created default feature names: {feature_names[:5]}...")
            
        # Feature dimension validation
        if len(feature_names) != X_train_values.shape[1]:
            print(f"Feature names count ({len(feature_names)}) doesn't match data dimension ({X_train_values.shape[1]})")
            print("Creating new feature names to match data dimensions")
            feature_names = [f"feature_{i}" for i in range(X_train_values.shape[1])]
            print(f"New feature names: {feature_names[:5]}...")
        
        if class_names is None:
            class_names = ["No Default", "Default"]
            
        # Check if model has predict_proba method
        if not hasattr(model, 'predict_proba'):
            print(f"Model {type(model).__name__} does not have predict_proba method")
            # Create a wrapper if model only has predict
            if hasattr(model, 'predict'):
                original_predict = model.predict
                def predict_proba_wrapper(X):
                    predictions = original_predict(X)
                    return np.column_stack((1-predictions, predictions))
                model.predict_proba = predict_proba_wrapper
                print("Created predict_proba wrapper")
            else:
                print("Model has no prediction method")
                return None
        
        # Test the predict_proba method
        try:
            print("Testing predict_proba function...")
            test_pred = model.predict_proba(X_test_values[0:1])
            print(f"Prediction shape: {test_pred.shape}, values: {test_pred}")
            if test_pred.shape[1] != 2:
                print(f"Warning: predict_proba doesn't return two classes, found {test_pred.shape[1]}")
        except Exception as pred_error:
            print(f"Error testing predict_proba: {str(pred_error)}")
            # Continue anyway as LIME might handle this
        
        # Create a copy of the data for LIME to avoid modification issues
        X_train_for_lime = X_train_values.copy()
        
        # Get LIME explanation
        print("Creating LIME explainer...")
        explainer = None
        try:
            # Use try/except blocks at each step to identify where the error occurs
            print(f"Setting up LimeTabularExplainer with training data shape {X_train_for_lime.shape}")
            
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train_for_lime,
                feature_names=feature_names,
                class_names=class_names,
                mode='classification',
                discretize_continuous=True,
                verbose=True
            )
            print("LimeTabularExplainer created successfully")
        except Exception as lime_init_error:
            print(f"Error creating LIME explainer: {str(lime_init_error)}")
            import traceback
            traceback.print_exc()
            
            # Try with a simplified approach
            try:
                print("Attempting simplified LIME explainer...")
                # Use a small random sample for training data
                np.random.seed(42)
                sample_indices = np.random.choice(X_train_values.shape[0], min(100, X_train_values.shape[0]), replace=False)
                simple_train = X_train_values[sample_indices]
                
                # Create with minimal parameters
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    simple_train,
                    mode='classification'
                )
                print("Simplified LIME explainer created")
            except Exception as simple_lime_error:
                print(f"Simplified LIME explainer failed: {str(simple_lime_error)}")
                traceback.print_exc()
        
        if explainer is None:
            print("Failed to create a LIME explainer")
            # Create a simple visualization as fallback
            return create_fallback_feature_plot(model, X_test_values, sample_idx, feature_names)
        
        print(f"Explaining instance {sample_idx}...")
        explanation = None
        try:
            test_instance = X_test_values[sample_idx].reshape(1, -1)
            print(f"Test instance shape: {test_instance.shape}")
            
            # Create predict function that properly handles input format
            def predict_fn(x):
                # Ensure x has the right shape (n_samples, n_features)
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                return model.predict_proba(x)
            
            explanation = explainer.explain_instance(
                X_test_values[sample_idx],
                predict_fn,
                num_features=min(10, X_train_values.shape[1]),
                top_labels=1
            )
            print("LIME explanation generated successfully")
        except Exception as explain_error:
            print(f"Error in explain_instance: {str(explain_error)}")
            import traceback
            traceback.print_exc()
            # Fall back to simpler method
            return create_fallback_feature_plot(model, X_test_values, sample_idx, feature_names)
        
        if not explanation:
            print("No explanation generated")
            return create_fallback_feature_plot(model, X_test_values, sample_idx, feature_names)
        
        # Create plot
        print("Creating LIME plot...")
        plt.figure(figsize=(10, 6))
        try:
            print("Converting explanation to pyplot figure")
            explanation.as_pyplot_figure(label=0)  # Use label=0 instead of 1 to avoid KeyError
            print("LIME pyplot figure created successfully")
        except Exception as plot_error:
            print(f"Error creating pyplot figure: {str(plot_error)}")
            import traceback
            traceback.print_exc()
            
            # Try manually creating the plot from explanation data
            try:
                print("Attempting manual plot creation...")
                exp_list = explanation.as_list(label=0)  # Use label=0 instead of 1
                features = [x[0] for x in exp_list]
                values = [x[1] for x in exp_list]
                
                plt.title("LIME Explanation (Manual Plot)")
                plt.barh(features, values)
                plt.xlabel("Feature Contribution")
                print("Manual LIME visualization created")
            except Exception as manual_error:
                print(f"Manual plotting failed: {str(manual_error)}")
                traceback.print_exc()
                return create_fallback_feature_plot(model, X_test_values, sample_idx, feature_names)
        
        # Save plot to base64 string
        print("Converting plot to base64 string...")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        base64_string = base64.b64encode(buf.getvalue()).decode('utf-8')
        print("LIME plot successfully created and encoded")
        return base64_string
    
    except Exception as e:
        print(f"Error creating LIME plot: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Create a simple fallback plot with an error message
        try:
            return create_fallback_feature_plot(model, X_test_values, sample_idx, feature_names, 
                                           error_message=f"Error creating LIME plot: {str(e)}")
        except Exception:
            # Ultra-simple error message as last resort
            try:
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, f"Visualization Failed: {str(e)}", 
                        ha='center', va='center', fontsize=14)
                plt.axis('off')
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close()
                buf.seek(0)
                return base64.b64encode(buf.getvalue()).decode('utf-8')
            except:
                return None

def create_fallback_feature_plot(model, X, sample_idx=0, feature_names=None, error_message=None):
    """Create a simple feature importance plot as fallback when LIME fails"""
    try:
        print("Creating fallback feature plot...")
        # Convert to numpy if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X_values = X.values
        else:
            X_values = X
            
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_values.shape[1])]
        
        plt.figure(figsize=(10, 6))
        
        if error_message:
            plt.text(0.5, 0.95, error_message, ha='center', va='top', wrap=True, fontsize=12, color='red')
            
        # Extract the feature values
        if sample_idx < X_values.shape[0]:
            feature_values = X_values[sample_idx]
        else:
            feature_values = X_values[0]
            
        # Get prediction if model is available
        pred_text = ""
        if model and hasattr(model, 'predict_proba'):
            try:
                pred = model.predict_proba(feature_values.reshape(1, -1))
                pred_text = f"(Prediction probability: {pred[0, 1]:.4f})"
            except:
                pass
        
        # Sort by absolute value
        feature_values_list = feature_values.tolist() if hasattr(feature_values, 'tolist') else list(feature_values)
        feature_names_list = list(feature_names)[:len(feature_values_list)]  # Ensure they match
        
        # Create indices sorted by absolute feature values
        sorted_indices = sorted(range(len(feature_values_list)), key=lambda i: abs(feature_values_list[i]), reverse=True)
        # Take top 10 or fewer
        top_n = min(10, len(sorted_indices))
        top_indices = sorted_indices[:top_n]
        
        plt.title(f"Feature Values {pred_text}\n(Fallback for LIME visualization)")
        plt.barh(
            [feature_names_list[i] for i in top_indices],
            [float(feature_values_list[i]) for i in top_indices]
        )
        plt.xlabel("Feature Value")
        
        # Save plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        print("Fallback visualization created successfully")
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error creating fallback visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Create an ultra-simple error message
        try:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Visualization Failed: {str(e)}", 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except:
            return None

def plot_shap_summary(shap_values, feature_names, X, plot_type='bar', max_display=10, save_path=None):
    """
    Plot SHAP summary plots
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values for each sample and feature
    feature_names : list
        List of feature names
    X : pd.DataFrame or np.ndarray
        The data used to generate SHAP values
    plot_type : str
        Type of plot to generate ('bar', 'beeswarm', or 'violin')
    max_display : int
        Maximum number of features to display
    save_path : str
        Path to save the plot. If None, plot will be displayed.
    """
    if isinstance(X, pd.DataFrame):
        X_display = X
    else:
        X_display = pd.DataFrame(X, columns=feature_names)
    
    plt.figure(figsize=(10, 8))
    
    if plot_type == 'bar':
        shap.summary_plot(shap_values, X_display, plot_type='bar', max_display=max_display, show=False)
    elif plot_type == 'beeswarm':
        shap.summary_plot(shap_values, X_display, max_display=max_display, show=False)
    elif plot_type == 'violin':
        shap.summary_plot(shap_values, X_display, plot_type='violin', max_display=max_display, show=False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_shap_dependence(shap_values, X, feature_idx, interaction_idx=None, feature_names=None, save_path=None):
    """
    Plot SHAP dependence plots for a specific feature
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values for each sample and feature
    X : pd.DataFrame or np.ndarray
        The data used to generate SHAP values
    feature_idx : int or str
        Index or name of the feature to plot
    interaction_idx : int or str
        Index or name of the interaction feature
    feature_names : list
        List of feature names
    save_path : str
        Path to save the plot. If None, plot will be displayed.
    """
    if isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = X.columns.tolist()
        
        # Convert feature name to index if needed
        if isinstance(feature_idx, str):
            feature_idx = feature_names.index(feature_idx)
        
        # Convert interaction feature name to index if needed
        if interaction_idx is not None and isinstance(interaction_idx, str):
            interaction_idx = feature_names.index(interaction_idx)
    else:
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
    
    plt.figure(figsize=(10, 7))
    
    if interaction_idx is not None:
        shap.dependence_plot(feature_idx, shap_values, X, interaction_index=interaction_idx, show=False)
    else:
        shap.dependence_plot(feature_idx, shap_values, X, show=False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def save_explanations(shap_values, feature_names, output_dir='explanations'):
    """
    Save SHAP values and feature names to disk
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values for each sample and feature
    feature_names : list
        List of feature names
    output_dir : str
        Directory to save the explanations
    """
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save SHAP values
    np.save(os.path.join(output_dir, 'shap_values.npy'), shap_values)
    
    # Save feature names
    joblib.dump(feature_names, os.path.join(output_dir, 'feature_names.pkl'))

def load_explanations(input_dir='explanations'):
    """
    Load SHAP values and feature names from disk
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the saved explanations
        
    Returns:
    --------
    shap_values : np.ndarray
        SHAP values for each sample and feature
    feature_names : list
        List of feature names
    """
    # Load SHAP values
    shap_values = np.load(os.path.join(input_dir, 'shap_values.npy'))
    
    # Load feature names
    feature_names = joblib.load(os.path.join(input_dir, 'feature_names.pkl'))
    
    return shap_values, feature_names 