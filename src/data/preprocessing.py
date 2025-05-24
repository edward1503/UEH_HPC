import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(train_path, test_size=0.2, random_state=42):
    """
    Load training data and split into train and validation sets
    
    Parameters:
    -----------
    train_path : str
        Path to training data CSV
    test_size : float
        Proportion of data to use for validation
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    train_data : pd.DataFrame
        Training data
    val_data : pd.DataFrame
        Validation data
    """
    try:
        # Load data
        data = pd.read_csv(train_path)
        logger.info(f"Loaded data with shape {data.shape}")
        
        # Check if target exists
        if 'SeriousDlqin2yrs' not in data.columns:
            raise ValueError("Target column 'SeriousDlqin2yrs' not found in data")
        
        # Split data into train and validation sets
        train_data, val_data = train_test_split(
            data, 
            test_size=test_size, 
            random_state=random_state,
            stratify=data['SeriousDlqin2yrs']  # Ensure balanced splits
        )
        
        logger.info(f"Split data into train ({train_data.shape}) and validation ({val_data.shape})")
        
        return train_data, val_data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def clean_data(df):
    """
    Clean data by handling missing values and outliers
    """
    try:
        # Make a copy to avoid modifying the original dataframe
        df_clean = df.copy()
        
        # Fill NaN values in numerical columns with median
        num_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            if col != 'SeriousDlqin2yrs':  # Don't modify the target variable
                missing_count = df_clean[col].isna().sum()
                if missing_count > 0:
                    logger.info(f"Filling {missing_count} missing values in column {col}")
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Ensure target column doesn't have NaN values
        if 'SeriousDlqin2yrs' in df_clean.columns:
            missing_target = df_clean['SeriousDlqin2yrs'].isna().sum()
            if missing_target > 0:
                logger.warning(f"Dropping {missing_target} rows with missing target values")
                df_clean = df_clean.dropna(subset=['SeriousDlqin2yrs'])
        
        # Remove rows with too many missing values (e.g., more than 50%)
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(thresh=len(df.columns) // 2)
        dropped_rows = initial_rows - len(df_clean)
        logger.info(f"Dropped {dropped_rows} rows with too many missing values")
        
        # Handle outliers for numerical columns using capping
        for col in num_cols:
            if col != 'SeriousDlqin2yrs':  # Don't modify the target variable
                q1 = df_clean[col].quantile(0.01)
                q3 = df_clean[col].quantile(0.99)
                outliers = ((df_clean[col] < q1) | (df_clean[col] > q3)).sum()
                if outliers > 0:
                    logger.info(f"Capping {outliers} outliers in column {col}")
                    df_clean[col] = df_clean[col].clip(q1, q3)
        
        return df_clean
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        raise

def create_preprocessing_pipeline(df, scaler_type='standard'):
    """
    Create a preprocessing pipeline for feature engineering
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    scaler_type : str
        Type of scaler to use ('standard', 'robust', 'minmax', 'power')
        
    Returns:
    --------
    preprocessor : ColumnTransformer
        Preprocessing pipeline
    """
    try:
        # Identify numerical and categorical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Numerical columns: {numerical_cols}")
        logger.info(f"Categorical columns: {categorical_cols}")
        
        # Remove target column if present
        if 'SeriousDlqin2yrs' in numerical_cols:
            numerical_cols.remove('SeriousDlqin2yrs')
        
        # Select the appropriate scaler based on scaler_type
        if scaler_type == 'robust':
            scaler = RobustScaler()
            logger.info("Using RobustScaler for numerical features")
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
            logger.info("Using MinMaxScaler for numerical features")
        elif scaler_type == 'power':
            scaler = PowerTransformer(method='yeo-johnson')
            logger.info("Using PowerTransformer for numerical features")
        else:  # default to standard
            scaler = StandardScaler()
            logger.info("Using StandardScaler for numerical features")
            
        # Create preprocessing pipelines
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', scaler)
        ])
        
        # Create categorical transformer if there are categorical columns
        transformers = [
            ('num', numerical_transformer, numerical_cols)
        ]
        
        if categorical_cols:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            transformers.append(('cat', categorical_transformer, categorical_cols))
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Remove columns not specified
        )
        
        return preprocessor
    except Exception as e:
        logger.error(f"Error creating preprocessing pipeline: {str(e)}")
        raise

def process_features(df, target_col='SeriousDlqin2yrs'):
    """
    Extract features and target from dataframe
    """
    try:
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found in dataframe")
            X = df
            y = None
        else:
            # Check for NaN values in target
            if df[target_col].isna().any():
                logger.warning(f"Found NaN values in target column. Dropping these rows.")
                df = df.dropna(subset=[target_col])
            
            # Convert target to int to avoid any float issues
            df[target_col] = df[target_col].astype(int)
            
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
        logger.info(f"Processed features with shape {X.shape}")
        if y is not None:
            logger.info(f"Target distribution: {y.value_counts(normalize=True)}")
            
        return X, y
    except Exception as e:
        logger.error(f"Error processing features: {str(e)}")
        raise

def feature_engineering(df):
    """
    Create new features that might be useful for credit risk prediction
    """
    try:
        # Make a copy to avoid modifying the original dataframe
        df_fe = df.copy()
        
        # Calculate debt-to-income ratio
        if 'DebtRatio' in df_fe.columns and 'MonthlyIncome' in df_fe.columns:
            # Avoid division by zero
            df_fe['DebtToIncomeRatio'] = df_fe['DebtRatio'] / df_fe['MonthlyIncome'].replace(0, np.nan)
            # Fill NaN values with median
            median_value = df_fe['DebtToIncomeRatio'].median()
            if np.isnan(median_value):
                median_value = 0  # Fallback if median is NaN
            df_fe['DebtToIncomeRatio'] = df_fe['DebtToIncomeRatio'].fillna(median_value)
            logger.info(f"Created feature 'DebtToIncomeRatio'")
        
        # Create numerical utilization levels
        if 'RevolvingUtilizationOfUnsecuredLines' in df_fe.columns:
            # Create new numerical feature based on bins
            bins = [0, 0.2, 0.4, 0.6, 0.8, float('inf')]
            # Use a safer approach without categorical intermediate step
            df_fe['UtilizationLevel'] = np.digitize(
                df_fe['RevolvingUtilizationOfUnsecuredLines'].values, bins=bins
            )
            logger.info(f"Created feature 'UtilizationLevel'")
        
        # Combine late payment features
        payment_cols = [col for col in df_fe.columns if 'NumberOfTime' in col and 'DaysPastDue' in col]
        if payment_cols:
            df_fe['TotalTimesLate'] = df_fe[payment_cols].sum(axis=1)
            logger.info(f"Created feature 'TotalTimesLate' from {len(payment_cols)} columns")
        
        # Log feature creation
        new_features = ['DebtToIncomeRatio', 'UtilizationLevel', 'TotalTimesLate']
        actual_new = [f for f in new_features if f in df_fe.columns]
        logger.info(f"Added {len(actual_new)} new features: {actual_new}")
        
        return df_fe
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise

def prepare_data_for_training(train_path, test_size=0.2, random_state=42, scaler_type='robust'):
    """
    Full data preparation pipeline using only training data, split into train and validation
    
    Parameters:
    -----------
    train_path : str
        Path to training data CSV
    test_size : float
        Proportion of data to use for validation
    random_state : int
        Random seed for reproducibility
    scaler_type : str
        Type of scaler to use ('standard', 'robust', 'minmax', 'power')
        
    Returns:
    --------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation target
    preprocessor : ColumnTransformer
        Preprocessing pipeline
    """
    try:
        logger.info("Loading and splitting data...")
        train_data, val_data = load_data(train_path, test_size, random_state)
        
        logger.info("Cleaning training data...")
        train_data = clean_data(train_data)
        
        logger.info("Cleaning validation data...")
        val_data = clean_data(val_data)
        
        logger.info("Performing feature engineering on training data...")
        train_data = feature_engineering(train_data)
        
        logger.info("Performing feature engineering on validation data...")
        val_data = feature_engineering(val_data)
        
        logger.info("Processing training features...")
        X_train, y_train = process_features(train_data)
        
        logger.info("Processing validation features...")
        X_val, y_val = process_features(val_data)
        
        logger.info(f"Creating preprocessing pipeline with {scaler_type} scaler...")
        preprocessor = create_preprocessing_pipeline(train_data, scaler_type=scaler_type)
        
        return X_train, y_train, X_val, y_val, preprocessor
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise 