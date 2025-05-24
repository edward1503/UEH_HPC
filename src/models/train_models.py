import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import uniform, randint
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import logging
from typing import Dict, Tuple, Any, List
import optuna
from optuna.integration import LightGBMPruningCallback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class to handle model training with proper handling of imbalanced data"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.evaluation_results = {}
        
        # Define model configurations
        self.model_configs = {
            'LightGBM': {
                'model': lgb.LGBMClassifier(random_state=random_state),
                'param_distributions': {
                    'model__n_estimators': [100, 200, 300, 500],
                    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'model__num_leaves': [31, 63, 127],
                    'model__max_depth': [3, 5, 7, 9],
                    'model__min_child_samples': [20, 50, 100],
                    'model__subsample': [0.8, 0.9, 1.0],
                    'model__colsample_bytree': [0.8, 0.9, 1.0],
                    'model__reg_alpha': [0, 0.1, 0.5],
                    'model__reg_lambda': [0, 0.1, 0.5],
                    'model__class_weight': ['balanced', None]
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=random_state),
                'param_distributions': {
                    'model__n_estimators': [100, 200, 300, 500],
                    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'model__max_depth': [3, 5, 7, 9],
                    'model__min_child_weight': [1, 3, 5],
                    'model__subsample': [0.8, 0.9, 1.0],
                    'model__colsample_bytree': [0.8, 0.9, 1.0],
                    'model__gamma': [0, 0.1, 0.2],
                    'model__scale_pos_weight': [1, 3, 5]
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=random_state),
                'param_distributions': {
                    'model__n_estimators': [100, 200, 300, 500],
                    'model__max_depth': [3, 5, 7, 9, None],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4],
                    'model__max_features': ['sqrt', 'log2'],
                    'model__class_weight': ['balanced', 'balanced_subsample', None]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=random_state),
                'param_distributions': {
                    'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'model__penalty': ['l1', 'l2', 'elasticnet'],
                    'model__solver': ['liblinear', 'saga'],
                    'model__class_weight': ['balanced', None]
                }
            }
        }

    def create_pipeline(self, model_name: str) -> Pipeline:
        """Create a pipeline with preprocessing and model"""
        model_config = self.model_configs[model_name]
        return Pipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=self.random_state)),
            ('model', model_config['model'])
        ])

    def objective(self, trial: optuna.Trial, model_name: str, X: np.ndarray, y: np.ndarray) -> float:
        """Optuna objective function for hyperparameter optimization"""
        model_config = self.model_configs[model_name]
        param_distributions = model_config['param_distributions']
        
        # Sample hyperparameters
        params = {}
        for param_name, param_values in param_distributions.items():
            if isinstance(param_values, list):
                params[param_name] = trial.suggest_categorical(param_name, param_values)
            elif isinstance(param_values, tuple):
                if isinstance(param_values[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_values[0], param_values[1])
                else:
                    params[param_name] = trial.suggest_float(param_name, param_values[0], param_values[1])
        
        # Create pipeline with sampled parameters
        pipeline = self.create_pipeline(model_name)
        pipeline.set_params(**params)
        
        # Use stratified k-fold cross validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            score = f1_score(y_val, y_pred)
            scores.append(score)
        
        return np.mean(scores)

    def optimize_hyperparameters(self, model_name: str, X: np.ndarray, y: np.ndarray, 
                               n_trials: int = 100) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        logger.info(f"Optimizing hyperparameters for {model_name}")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.objective(trial, model_name, X, y), 
                      n_trials=n_trials)
        
        best_params = study.best_params
        logger.info(f"Best parameters for {model_name}: {best_params}")
        return best_params

    def train_model(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Tuple[Any, Dict[str, float]]:
        """Train a single model with optimized hyperparameters"""
        logger.info(f"Training {model_name}")
        
        # Optimize hyperparameters
        best_params = self.optimize_hyperparameters(model_name, X, y)
        
        # Create and train final pipeline
        pipeline = self.create_pipeline(model_name)
        pipeline.set_params(**best_params)
        pipeline.fit(X, y)
        
        # Evaluate model
        y_pred = pipeline.predict(X)
        y_pred_proba = pipeline.predict_proba(X)[:, 1]
        
        evaluation = {
            'f1_score': f1_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        self.models[model_name] = pipeline
        self.best_params[model_name] = best_params
        self.evaluation_results[model_name] = evaluation
        
        logger.info(f"{model_name} evaluation results: {evaluation}")
        return pipeline, evaluation

    def train_all_models(self, X: np.ndarray, y: np.ndarray, 
                        save_dir: str = 'models') -> Dict[str, Any]:
        """Train all models and save results"""
        results = {}
        
        for model_name in self.model_configs.keys():
            try:
                model, evaluation = self.train_model(model_name, X, y)
                
                # Save model
                model_path = f"{save_dir}/{model_name}.pkl"
                joblib.dump(model, model_path)
                logger.info(f"Saved {model_name} to {model_path}")
                
                results[model_name] = {
                    'model_path': model_path,
                    'evaluation': evaluation,
                    'best_params': self.best_params[model_name]
                }
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Save training results
        results_path = f"{save_dir}/training_results.json"
        with open(results_path, 'w') as f:
            import json
            json.dump(results, f, indent=4)
        logger.info(f"Saved training results to {results_path}")
        
        return results

def train_all_models(X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray, y_val: np.ndarray,
                    save_dir: str = 'models') -> Dict[str, Any]:
    """Main function to train all models"""
    trainer = ModelTrainer()
    return trainer.train_all_models(X_train, y_train, save_dir)

def train_random_forest(X_train, y_train, params=None):
    """
    Train a Random Forest model with hyperparameter tuning
    """
    # Define parameter grid
    if params is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    else:
        param_grid = params
    
    # Use RandomizedSearchCV for efficiency
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_grid,
        n_iter=20,
        cv=3,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the search
    try:
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        print(f"Best params for random forest: {best_params}")
    except Exception as e:
        print(f"Error during hyperparameter search: {e}")
        print("Falling back to default model")
        best_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        best_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt'
        }
        best_model.fit(X_train, y_train)
    
    # Return best model and best parameters
    return best_model, best_params

def train_xgboost(X_train, y_train, params=None):
    """
    Train an XGBoost model with hyperparameter tuning
    """
    # Define parameter distributions for randomized search
    if params is None:
        param_distributions = {
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 10),
            'min_child_weight': randint(1, 10),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5),
            'gamma': uniform(0, 1),
            'n_estimators': randint(50, 200)
        }
    else:
        param_distributions = params
    
    # Use RandomizedSearchCV
    search = RandomizedSearchCV(
        xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            use_label_encoder=False
        ),
        param_distributions=param_distributions,
        n_iter=20,
        cv=3,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the search
    try:
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        print(f"Best params for XGBoost: {best_params}")
    except Exception as e:
        print(f"Error during hyperparameter search: {e}")
        print("Falling back to default model")
        best_model = xgb.XGBClassifier(
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=100,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            use_label_encoder=False
        )
        best_params = {
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 100
        }
        best_model.fit(X_train, y_train)
    
    # Return best model and best parameters
    return best_model, best_params

def train_lightgbm(X_train, y_train, params=None):
    """
    Train a LightGBM model with hyperparameter tuning
    """
    # Define parameter distributions for randomized search
    if params is None:
        param_distributions = {
            'learning_rate': uniform(0.01, 0.3),
            'num_leaves': randint(20, 150),
            'max_depth': randint(3, 10),
            'min_child_samples': randint(5, 50),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5),
            'n_estimators': randint(50, 200)
        }
    else:
        param_distributions = params
    
    # Use RandomizedSearchCV
    search = RandomizedSearchCV(
        lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            boost_from_average=True,
            random_state=42,
            verbose=-1
        ),
        param_distributions=param_distributions,
        n_iter=20,
        cv=3,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the search - explicitly pass feature names to avoid any issues
    try:
        if isinstance(X_train, pd.DataFrame):
            X_train_copy = X_train.copy()
            search.fit(X_train_copy, y_train)
        else:
            search.fit(X_train, y_train)
            
        best_model = search.best_estimator_
        best_params = search.best_params_
        print(f"Best params for LightGBM: {best_params}")
    except Exception as e:
        print(f"Error during hyperparameter search: {e}")
        print("Falling back to default model")
        best_model = lgb.LGBMClassifier(
            learning_rate=0.1,
            num_leaves=31,
            max_depth=5,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=100,
            objective='binary',
            metric='auc',
            boost_from_average=True,
            random_state=42,
            verbose=-1
        )
        best_params = {
            'learning_rate': 0.1,
            'num_leaves': 31,
            'max_depth': 5,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 100
        }
        
        if isinstance(X_train, pd.DataFrame):
            X_train_copy = X_train.copy()
            best_model.fit(X_train_copy, y_train, feature_name='auto')
        else:
            best_model.fit(X_train, y_train)
    
    # Return best model and best parameters
    return best_model, best_params

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test data
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    return metrics

def save_model(model, model_name, model_dir='models'):
    """
    Save trained model to disk
    """
    # Create directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save model
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model_path

def load_model(model_path):
    """
    Load trained model from disk
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

def train_all_models(X_train, y_train, X_test=None, y_test=None, save_dir='models'):
    """
    Train all models and evaluate them if test data is provided
    """
    models = {
        'random_forest': train_random_forest,
        'xgboost': train_xgboost,
        'lightgbm': train_lightgbm
    }
    
    trained_models = {}
    model_paths = {}
    evaluation_results = {}
    
    # Sanitize feature names to avoid LightGBM errors
    # LightGBM doesn't support special characters in feature names
    if isinstance(X_train, pd.DataFrame):
        # Create a mapping of original to sanitized feature names
        feature_name_mapping = {}
        sanitized_columns = []
        
        for col in X_train.columns:
            # Replace special characters with underscore
            sanitized_col = ''.join(c if c.isalnum() else '_' for c in str(col))
            # Ensure the name starts with a letter or underscore
            if not (sanitized_col[0].isalpha() or sanitized_col[0] == '_'):
                sanitized_col = 'f_' + sanitized_col
            # Avoid duplicate names by adding an index if needed
            i = 0
            original_sanitized = sanitized_col
            while sanitized_col in sanitized_columns:
                i += 1
                sanitized_col = f"{original_sanitized}_{i}"
            
            feature_name_mapping[col] = sanitized_col
            sanitized_columns.append(sanitized_col)
        
        # Rename columns in the dataframes
        X_train = X_train.rename(columns=feature_name_mapping)
        if X_test is not None and isinstance(X_test, pd.DataFrame):
            X_test = X_test.rename(columns=feature_name_mapping)
        
        print(f"Sanitized {len(feature_name_mapping)} feature names for model compatibility")
    
    for model_name, train_func in models.items():
        print(f"Training {model_name}...")
        model, best_params = train_func(X_train, y_train)
        trained_models[model_name] = model
        
        # Save model
        model_path = save_model(model, model_name, save_dir)
        model_paths[model_name] = model_path
        
        # Evaluate model if test data is provided
        if X_test is not None and y_test is not None:
            metrics = evaluate_model(model, X_test, y_test)
            evaluation_results[model_name] = {
                'metrics': metrics,
                'best_params': best_params
            }
    
    return trained_models, model_paths, evaluation_results 