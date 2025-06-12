import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import os # For path joining

# --- Configuration ---
MODEL_DIR = "models"
MODEL_FILENAME = "best_random_forest_model.pkl"
COMPONENTS_FILENAME = "model_eval_components.pkl"

# Set a consistent plotting style for professional-looking visuals
plt.style.use('seaborn-v0_8-whitegrid') # A clean and modern seaborn style
sns.set_palette("husl", n_colors=8) # Using a vibrant and distinct color palette

def evaluate_model():
    """
    Orchestrates Stage 3: Model Evaluation.
    Loads the trained model and data components from Stage 2,
    makes predictions on the test set, calculates comprehensive metrics,
    and visualizes performance.
    """
    print("=" * 70)
    print("STAGE 3: EVALUATE THE TRAINED MACHINE LEARNING MODEL")
    print("=" * 70)

    # --- A. Setup and Load Artifacts ---
    print("\n[3.1] Loading model and evaluation components from Stage 2...")
    try:
        model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
        eval_components_path = os.path.join(MODEL_DIR, COMPONENTS_FILENAME)

        if not os.path.exists(model_path) or not os.path.exists(eval_components_path):
            print(f"❌ ERROR: Model ('{model_path}') or evaluation components ('{eval_components_path}') file not found.")
            print("     Ensure Stage 2 (CDAS2.py) completed successfully and artifacts were saved.")
            return # Halt script

        model = joblib.load(model_path)
        eval_components = joblib.load(eval_components_path)

        X_test = eval_components.get('X_test')
        y_test = eval_components.get('y_test')
        # feature_names = eval_components.get('feature_names') # For Stage 4
        
        if X_test is None or y_test is None:
            print("❌ ERROR: X_test or y_test not found in loaded evaluation components.")
            return

        print(f"   ✓ Model loaded successfully from: '{model_path}'")
        print(f"   ✓ Evaluation components loaded successfully from: '{eval_components_path}'")
        print(f"   ✓ X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    except Exception as e:
        print(f"❌ CRITICAL ERROR loading artifacts: {e}")
        return # Halt script

    # --- B. Make Predictions on the Test Set ---
    print("\n[3.2] Making predictions on the unseen test set...")
    try:
        y_pred_test = model.predict(X_test)
        # Probabilities for the positive class (booking_complete=1) are needed for ROC-AUC and PR Curve
        y_pred_proba_test = model.predict_proba(X_test)[:, 1] 
        print("   ✓ Predictions and class probabilities generated for the test set.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR during prediction on test set: {e}")
        return # Halt script

    # --- C. Report Cross-Validation Performance (from RandomizedSearchCV in Stage 2) ---
    print("\n[3.3] Cross-Validation Performance (from Stage 2 Hyperparameter Tuning):")
    # This value was logged by CDAS2.py from RandomizedSearchCV's best_score_ attribute.
    # If this score was saved in eval_components, it could be loaded directly.
    # For this script, we'll use the value you noted from CDAS2.py's output.
    cv_roc_auc_from_tuning = 0.7882 # As per CDAS2.py output from previous run
    print(f"   Best Cross-Validated ROC AUC (during RandomizedSearchCV with cv=5): {cv_roc_auc_from_tuning:.4f}")
    print("      Note: This metric reflects the model's average performance on validation folds")
    print("            during the training and hyperparameter tuning phase in Stage 2.")

    # --- D. Calculate and Report Evaluation Metrics on the Test Set ---
    print("\n[3.4] Calculating Evaluation Metrics on the Unseen Test Set...")
    try:
        accuracy_test = accuracy_score(y_test, y_pred_test)
        precision_test = precision_score(y_test, y_pred_test, zero_division=0) # For positive class
        recall_test = recall_score(y_test, y_pred_test, zero_division=0)       # For positive class
        f1_test = f1_score(y_test, y_pred_test, zero_division=0)               # For positive class
        roc_auc_test = roc_auc_score(y_test, y_pred_proba_test)

        print("\n   --- Key Performance Metrics (Test Set) ---")
        print(f"   Accuracy:                             {accuracy_test:.4f}")
        print(f"      Interpretation: Overall percentage of correct predictions.")
        print(f"   Precision (Positive Class - Booking): {precision_test:.4f}")
        print(f"      Interpretation: Of all instances the model PREDICTED as 'Booking', {precision_test*100:.2f}% actually were bookings.")
        print(f"   Recall (Sensitivity - Positive Class - Booking): {recall_test:.4f}")
        print(f"      Interpretation: Of all ACTUAL 'Booking' instances, the model correctly identified {recall_test*100:.2f}%.")
        print(f"   F1 Score (Positive Class - Booking):  {f1_test:.4f}")
        print(f"      Interpretation: Harmonic mean of Precision and Recall; a balanced measure, useful for imbalanced classes.")
        print(f"   ROC-AUC Score:                        {roc_auc_test:.4f}")
        print(f"      Interpretation: Model's ability to distinguish between 'Booking' and 'No Booking' classes across all thresholds.")
        print("      (A score of 0.5 is random, 1.0 is perfect).")

        # --- D.1. Display Confusion Matrix ---
        print("\n   --- Confusion Matrix (Test Set) ---")
        cm_test = confusion_matrix(y_test, y_pred_test)
        
        # For clear labels in the heatmap
        class_labels = ['No Booking (0)', 'Booking (1)']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=class_labels, yticklabels=class_labels,
                    annot_kws={"size": 14}) # Increase annotation font size
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.title('Confusion Matrix - Test Set Performance', fontsize=16)
        plt.show()

        # Detailed breakdown of Confusion Matrix components
        tn, fp, fn, tp = cm_test.ravel()
        print(f"      True Negatives (TN): {tn:6d} (Correctly predicted 'No Booking')")
        print(f"      False Positives (FP): {fp:5d} (Incorrectly predicted 'Booking' - Type I Error)")
        print(f"      False Negatives (FN): {fn:5d} (Incorrectly predicted 'No Booking' - Type II Error)")
        print(f"      True Positives (TP): {tp:6d} (Correctly predicted 'Booking')")

        # --- D.2. Display Classification Report ---
        print("\n   --- Classification Report (Test Set) ---")
        # Provides precision, recall, f1-score for each class.
        report_test = classification_report(y_test, y_pred_test, target_names=class_labels, zero_division=0)
        print(report_test)

    except Exception as e:
        print(f"❌ CRITICAL ERROR calculating or displaying metrics: {e}")
        return # Halt script

    # --- E. Visualize Performance Curves (on Test Set) ---
    print("\n[3.5] Generating Performance Curve Visualizations (Test Set)...")
    try:
        # --- E.1. ROC Curve ---
        fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba_test)
        
        plt.figure(figsize=(9, 7)) # Slightly larger for better readability
        plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC curve (AUC = {roc_auc_test:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance Level (AUC = 0.50)')
        plt.xlim([-0.02, 1.02]) # Adjusted limits for aesthetics
        plt.ylim([-0.02, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
        plt.ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC) Curve - Test Set', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.show()
        print("   ✓ ROC Curve generated.")

        # --- E.2. Precision-Recall Curve ---
        precision_pr, recall_pr, thresholds_pr = precision_recall_curve(y_test, y_pred_proba_test)
        avg_precision = average_precision_score(y_test, y_pred_proba_test)
        
        plt.figure(figsize=(9, 7))
        plt.plot(recall_pr, precision_pr, color='dodgerblue', lw=2.5, label=f'Precision-Recall curve (AP = {avg_precision:.4f})')
        # Plotting a reference line representing a no-skill classifier for PR curve
        # This is the ratio of positives in the dataset
        no_skill = len(y_test[y_test==1]) / len(y_test)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='grey', label=f'No Skill (AP = {no_skill:.2f})')
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.ylim([-0.02, 1.05])
        plt.xlim([-0.02, 1.02])
        plt.title('Precision-Recall Curve - Test Set', fontsize=16)
        plt.legend(loc="upper right", fontsize=12) # Adjusted legend location
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.show()
        print(f"   ✓ Precision-Recall Curve generated. Average Precision (AP): {avg_precision:.4f}")
        print("      Note: AP summarizes the PR curve and is a good metric for imbalanced datasets.")

    except Exception as e:
        print(f"❌ ERROR generating performance curve visualizations: {e}")
        # Continue even if plotting fails, as metrics are more critical.

    # --- F. Concluding Stage 3 ---
    print("\n" + "=" * 70)
    print("✅ STAGE 3: MODEL EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("   Review the metrics and visualizations above to gain a comprehensive understanding")
    print("   of the trained model's performance on the unseen test data.")
    print("   This provides a strong basis for deciding if the model is suitable for its intended purpose")
    print("   or if further iterations (e.g., more feature engineering, different model types) are needed.")

if __name__ == "__main__":
    evaluate_model()