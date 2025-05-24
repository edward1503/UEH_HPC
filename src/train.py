import pandas as pd
import numpy as np
import os
import joblib
import logging
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, Tuple, List

# Silence the physical cores warning
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Set to a reasonable number
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

# Import modules from project
from data.preprocessing import prepare_data_for_training
from models.train_models import train_all_models, evaluate_model, save_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train credit default risk models')
    
    parser.add_argument(
        '--train_data',
        type=str,
        default='data/raw/cs-training.csv',
        help='Path to training data CSV'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models',
        help='Directory to save trained models and results'
    )
    
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random state for reproducibility'
    )
    
    parser.add_argument(
        '--val_size',
        type=float,
        default=0.2,
        help='Proportion of data to use for validation'
    )
    
    return parser.parse_args()

def main():
    """Main function to train and evaluate models"""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Check if training file exists
    if not os.path.exists(args.train_data):
        logger.error(f"Training data file not found: {args.train_data}")
        return
    
    # Prepare data
    logger.info("Preparing data for training...")
    try:
        # Use the updated prepare_data_for_training function with train/val split
        X_train, y_train, X_val, y_val, preprocessor = prepare_data_for_training(
            args.train_data, 
            test_size=args.val_size,
            random_state=args.random_state
        )
        
        logger.info(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        logger.info(f"Validation data: {X_val.shape[0]} samples, {X_val.shape[1]} features")
        
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        return
    
    # Save preprocessor
    preprocessor_path = os.path.join(args.output_dir, 'preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    # Train models
    logger.info("Training models...")
    try:
        trained_models, model_paths, evaluation_results = train_all_models(
            X_train, y_train, X_val, y_val, save_dir=args.output_dir
        )
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        return
    
    # Log evaluation results
    logger.info("Model evaluation results:")
    for model_name, result in evaluation_results.items():
        logger.info(f"Model: {model_name}")
        logger.info(f"Best parameters: {result['best_params']}")
        logger.info(f"Metrics:")
        for metric, value in result['metrics'].items():
            if metric not in ['confusion_matrix', 'classification_report']:
                logger.info(f"  {metric}: {value:.4f}")
    
    # Save evaluation results
    results_path = os.path.join(args.output_dir, 'evaluation_results.pkl')
    joblib.dump(evaluation_results, results_path)
    logger.info(f"Evaluation results saved to {results_path}")
    
    # Create performance comparison plot
    create_performance_plot(evaluation_results, os.path.join(args.output_dir, 'model_performance.png'))
    
    logger.info("Training and evaluation completed successfully!")

def create_performance_plot(evaluation_results, save_path):
    """Create and save model performance comparison plot"""
    # Extract metrics
    models = []
    accuracy = []
    precision = []
    recall = []
    f1 = []
    roc_auc = []
    
    for model_name, result in evaluation_results.items():
        metrics = result['metrics']
        models.append(model_name)
        accuracy.append(metrics['accuracy'])
        precision.append(metrics['precision'])
        recall.append(metrics['recall'])
        f1.append(metrics['f1'])
        roc_auc.append(metrics['roc_auc'])
    
    # Create dataframe
    df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    })
    
    # Melt dataframe for plotting
    df_melted = pd.melt(df, id_vars=['Model'], var_name='Metric', value_name='Score')
    
    # Create plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted)
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    main() 