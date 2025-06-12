import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib # For saving model and components
import os # For checking directory existence
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
# Define file paths and model parameters centrally for easier management
PROCESSED_DATA_PATH = "data/final_processed_dataset_stage1.pkl"
MODEL_OUTPUT_DIR = "models"
MODEL_FILENAME = "best_random_forest_model.pkl"
COMPONENTS_FILENAME = "model_eval_components.pkl"
RANDOM_STATE = 42 # Ensures reproducibility

def train_evaluate_model(df_processed_stage1: pd.DataFrame):
    """
    Orchestrates Stage 2: Model Training.
    This function selects features/target, splits data, trains an initial model,
    performs hyperparameter optimization, and prepares components for evaluation.

    Args:
        df_processed_stage1 (pd.DataFrame): The fully processed DataFrame from Stage 1.

    Returns:
        tuple: (best_rf_model, model_components) if successful, otherwise (None, None).
               best_rf_model: The trained RandomForestClassifier with optimal hyperparameters.
               model_components: A dictionary containing data splits, the model, and feature names.
    """
    print("=" * 70)
    print("STAGE 2: TRAIN A MACHINE LEARNING MODEL (Expert Implementation)")
    print("=" * 70)

    if not isinstance(df_processed_stage1, pd.DataFrame) or df_processed_stage1.empty:
        print("❌ CRITICAL ERROR: Input DataFrame is invalid or empty. Halting Stage 2.")
        return None, None

    # --- 1. Feature and Target Selection ---
    print("\n[1.0] Selecting Features (X) and Target (y)...")
    target_column = 'booking_complete'
    if target_column not in df_processed_stage1.columns:
        print(f"❌ CRITICAL ERROR: Target column '{target_column}' not found. Verify Stage 1 output.")
        return None, None
        
    X = df_processed_stage1.drop(target_column, axis=1)
    y = df_processed_stage1[target_column]
    print(f"   ✓ Features (X) shape: {X.shape}")
    print(f"   ✓ Target (y) shape: {y.shape}")

    # Validate feature set: Ensure all features are numeric
    non_numeric_cols = X.select_dtypes(exclude=np.number).columns
    if not non_numeric_cols.empty:
        print(f"❌ CRITICAL ERROR: Non-numeric columns found in features (X): {list(non_numeric_cols)}")
        print("     All features must be numeric for RandomForestClassifier. Review Stage 1 encoding steps.")
        return None, None
    print("   ✓ All features are confirmed numeric.")

    # --- 2. Data Splitting ---
    print("\n[2.0] Splitting data into training and testing sets (80% train / 20% test)...")
    # Stratification is crucial for imbalanced datasets to maintain class proportions in splits.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.20, 
            random_state=RANDOM_STATE, 
            stratify=y 
        )
        print(f"   ✓ X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"   ✓ X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        print(f"   ✓ Training target distribution (normalized): \n{y_train.value_counts(normalize=True).to_string()}")
        print(f"   ✓ Testing target distribution (normalized): \n{y_test.value_counts(normalize=True).to_string()}")
        print("   ✓ Stratification applied successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR during data splitting: {e}")
        return None, None

    # --- 3. Initial Model Training (Baseline) ---
    print("\n[3.0] Training Initial RandomForestClassifier (Baseline)...")
    # Using 'class_weight='balanced'' to help with class imbalance.
    # n_jobs=-1 utilizes all available CPU cores for training.
    rf_initial = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
    
    try:
        rf_initial.fit(X_train, y_train)
        print("   ✓ Initial RandomForestClassifier trained successfully.")
        
        # Quick baseline evaluation on the test set
        y_pred_initial_test = rf_initial.predict(X_test)
        initial_accuracy = accuracy_score(y_test, y_pred_initial_test)
        initial_roc_auc = roc_auc_score(y_test, rf_initial.predict_proba(X_test)[:, 1])
        print(f"   Baseline Model - Test Accuracy: {initial_accuracy:.4f}")
        print(f"   Baseline Model - Test ROC AUC: {initial_roc_auc:.4f}")
    except Exception as e:
        print(f"❌ ERROR during initial model training: {e}")
        # Decide if to proceed or halt. For now, we'll halt if baseline fails.
        return None, None

    # --- 4. Hyperparameter Optimization (RandomizedSearchCV) ---
    print("\n[4.0] Performing Hyperparameter Optimization with RandomizedSearchCV...")
    
    # Define a comprehensive yet manageable parameter distribution for RandomizedSearch
    # These ranges are common starting points for RandomForest.
    param_dist = {
        'n_estimators': [100, 200, 300],       # Number of trees
        'max_features': ['sqrt', 'log2', 0.5], # Number of features to consider at every split
        'max_depth': [10, 20, 30, None],       # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],       # Minimum samples required to split a node
        'min_samples_leaf': [1, 2, 4],         # Minimum samples required at each leaf node
        'bootstrap': [True, False],            # Method of selecting samples for training each tree
        'class_weight': ['balanced', 'balanced_subsample'] # Handles imbalanced classes
    }

    # RandomizedSearchCV is often more efficient than GridSearchCV for larger search spaces.
    # n_iter: Number of parameter settings that are sampled. Higher is better but slower.
    # cv: Number of cross-validation folds. 5 is a common choice.
    # scoring='roc_auc': Excellent metric for imbalanced binary classification.
    # refit=True: Automatically refits the best estimator on the whole training set.
    rf_random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=25,  # Increased iterations for a more thorough search (adjust based on time)
        cv=5,       # 5-fold cross-validation
        verbose=2,  # Higher verbosity for more detailed output
        random_state=RANDOM_STATE,
        n_jobs=-1,  # Utilize all available cores for search
        scoring='roc_auc' 
    )

    try:
        print("   Starting RandomizedSearchCV fitting (this may take some time)...")
        rf_random_search.fit(X_train, y_train)
        print("   ✓ RandomizedSearchCV fitting complete.")

        print("\n   Optimal Parameters found by RandomizedSearchCV:")
        # Nicely format best_params_
        for param, value in rf_random_search.best_params_.items():
            print(f"     - {param}: {value}")
            
        print(f"\n   Best Cross-Validated ROC AUC Score from RandomizedSearchCV:")
        print(f"     {rf_random_search.best_score_:.4f}")

        best_rf_model = rf_random_search.best_estimator_
        
    except Exception as e:
        print(f"❌ ERROR during RandomizedSearchCV: {e}")
        print("   ⚠️ WARNING: Hyperparameter tuning failed. Falling back to the initial baseline model.")
        best_rf_model = rf_initial 

    # --- 5. Final Model Preparation ---
    # The best_rf_model is already trained on the full X_train, y_train due to refit=True in RandomizedSearchCV.
    print("\n[5.0] Final model prepared (best estimator from RandomizedSearchCV or baseline).")
    
    # Package components for Stage 3 (Evaluation & Interpretation)
    model_components = {
        'X_train': X_train, 'X_test': X_test, 
        'y_train': y_train, 'y_test': y_test,
        'model': best_rf_model,
        'feature_names': list(X.columns) 
    }
    
    print("\n✅ STAGE 2: MODEL TRAINING AND HYPERPARAMETER OPTIMIZATION COMPLETED SUCCESSFULLY!")
    return best_rf_model, model_components

# --- Helper function for DUMMY data (fallback if Stage 1 output is not found) ---
def load_dummy_processed_data(num_samples=5000, num_features_approx=900):
    """Generates a plausible dummy DataFrame mimicking Stage 1 output."""
    print("\n--- LOADING DUMMY PROCESSED DATA (Fallback for Stage 2 demonstration) ---")
    # Create a base of random features
    data = np.random.rand(num_samples, num_features_approx - 4) # Reserve space for specific dummy features
    columns = [f'feature_ohe_{i}' for i in range(num_features_approx - 4)]
    df = pd.DataFrame(data, columns=columns)
    
    # Simulate key engineered/encoded features from Stage 1
    df['route_popularity'] = np.random.randint(1, 250, num_samples)
    df['flight_day_numeric'] = np.random.randint(1, 8, num_samples)
    df['origin_route_pair_encoded'] = np.random.randint(0, 3000, num_samples) # Simulating label encoded
    df['lead_time_Medium_Lead'] = np.random.randint(0,2, num_samples)
    df['lead_time_Long_Lead'] = np.random.randint(0,2, num_samples)
    # ... add more dummy OHE columns if needed to match expected dimensionality
    
    # Add target variable 'booking_complete' (imbalanced)
    # Create a slightly more complex probability for the target
    prob = 0.05 + \
           0.10 * (df['feature_ohe_0'] > 0.6) + \
           0.05 * (df['route_popularity'] / 250) + \
           0.05 * (df['origin_route_pair_encoded'] / 3000) + \
           0.03 * (df['lead_time_Medium_Lead'])
    df['booking_complete'] = np.random.binomial(1, np.clip(prob, 0.02, 0.5), num_samples) # Ensure some imbalance
    
    print(f"   Dummy data shape: {df.shape}")
    print(f"   Dummy data target distribution (normalized):\n{df['booking_complete'].value_counts(normalize=True).to_string()}")
    print("--- DUMMY DATA LOADED ---")
    return df

# --- Main execution block ---
if __name__ == "__main__":
    print("🚀 Initiating CDAS2.py - Model Training Stage 🚀")

    # Ensure model output directory exists
    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)
        print(f"   ✓ Created directory for saving models: '{MODEL_OUTPUT_DIR}'")

    # Load processed data from Stage 1
    try:
        df_processed_stage1 = pd.read_pickle(PROCESSED_DATA_PATH)
        print(f"✓ Successfully loaded processed data from Stage 1: '{PROCESSED_DATA_PATH}'")
    except FileNotFoundError:
        print(f"❌ ERROR: Processed data file ('{PROCESSED_DATA_PATH}') not found.")
        print("     Please ensure Stage 1 (CDAS.py or equivalent) has been run successfully and the output file exists.")
        print("     Proceeding with DUMMY DATA for demonstration purposes ONLY.")
        df_processed_stage1 = load_dummy_processed_data(num_samples=50000, num_features_approx=917) # Match expected shape
    except Exception as e:
        print(f"❌ ERROR loading processed data: {e}")
        print("     Proceeding with DUMMY DATA for demonstration purposes ONLY.")
        df_processed_stage1 = load_dummy_processed_data(num_samples=50000, num_features_approx=917)

    if df_processed_stage1 is not None:
        trained_model, model_eval_components = train_evaluate_model(df_processed_stage1)

        if trained_model and model_eval_components:
            print("\n--- Persisting Trained Model and Evaluation Components ---")
            try:
                model_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME)
                joblib.dump(trained_model, model_path)
                print(f"   ✓ Trained model saved to: '{model_path}'")

                components_path = os.path.join(MODEL_OUTPUT_DIR, COMPONENTS_FILENAME)
                joblib.dump(model_eval_components, components_path)
                print(f"   ✓ Model evaluation components saved to: '{components_path}'")
                
                print("\n🏁 Model training stage (CDAS2.py) complete. Artifacts saved.")
                print("   Ready for Stage 3: Model Evaluation and Interpretation.")
            except Exception as e:
                print(f"❌ ERROR saving model/components: {e}")
        else:
            print("\n❌ Model training stage (CDAS2.py) did not complete successfully. Artifacts not saved.")
    else:
        print("\n❌ Halting CDAS2.py as no data was available for training.")
