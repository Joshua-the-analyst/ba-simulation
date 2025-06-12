import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
import os
import warnings
warnings.filterwarnings('ignore')

# Set consistent plotting style to match your previous stages
plt.style.use('default')
sns.set_palette("husl", n_colors=8)

# --- Configuration ---
MODEL_DIR = "models"
MODEL_FILENAME = "best_random_forest_model.pkl"
COMPONENTS_FILENAME = "model_eval_components.pkl"

def visualize_feature_contributions():
    """
    Stage 4: Visualize Feature Contribution
    Analyzes and visualizes which features are most important for the model's predictions.
    """
    print("=" * 70)
    print("STAGE 4: VISUALIZE FEATURE CONTRIBUTION")
    print("=" * 70)

    # --- A. Setup and Load Artifacts ---
    print("\n[4.1] Loading model and evaluation components from Stage 2...")
    try:
        model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
        eval_components_path = os.path.join(MODEL_DIR, COMPONENTS_FILENAME)

        if not os.path.exists(model_path) or not os.path.exists(eval_components_path):
            print(f"❌ ERROR: Model ('{model_path}') or evaluation components ('{eval_components_path}') file not found.")
            print("     Ensure Stage 2 (CDAS2.py) completed successfully and artifacts were saved.")
            return # Halt script

        model = joblib.load(model_path)
        eval_components = joblib.load(eval_components_path)

        X_train = eval_components.get('X_train')
        X_test = eval_components.get('X_test')
        feature_names = eval_components.get('feature_names')
        
        if X_train is None or feature_names is None:
            print("❌ ERROR: X_train or feature_names not found in loaded evaluation components.")
            return

        print(f"   ✓ Model loaded successfully from: '{model_path}'")
        print(f"   ✓ Evaluation components loaded successfully from: '{eval_components_path}'")
        print(f"   ✓ X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        print(f"   ✓ Number of features: {len(feature_names)}")

    except Exception as e:
        print(f"❌ CRITICAL ERROR loading artifacts: {e}")
        return # Halt script

    # --- B. Method 1: RandomForest Built-in Feature Importances ---
    print("\n[4.2] Visualizing RandomForest Built-in Feature Importances...")
    try:
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        top_n = 20 # Number of top features to display
        print(f"\n   Top {top_n} most important features (according to RandomForest):")
        print(feature_importance_df.head(top_n))

        plt.figure(figsize=(12, 8)) # Adjusted for better readability of feature names
        sns.barplot(x='importance', y='feature', data=feature_importance_df.head(top_n),
                    palette='viridis_r') # Using a reverse viridis palette
        plt.title(f'Top {top_n} Feature Importances from RandomForest', fontsize=16)
        plt.xlabel('Importance Score', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.yticks(fontsize=10) # Adjust y-tick font size if names are long
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        plt.show()
        print("   ✓ Bar chart of feature importances generated.")
        print("      Interpretation: Features with higher scores contribute more to the model's decision-making process (based on Gini impurity or similar criteria).")
    except Exception as e:
        print(f"❌ ERROR generating RandomForest feature importances: {e}")

    # --- C. Method 2: Partial Dependence Plots (PDP) - FIXED VERSION ---
    print("\n[4.3] Visualizing Partial Dependence Plots (PDPs)...")
    
    # Get top features from importance ranking and filter for PDP suitability
    pdp_candidate_features = feature_importance_df['feature'].head(15).tolist() # Take top 15 candidates
    
    # Filter features suitable for PDP:
    # 1. Must exist in X_train
    # 2. Must have more than 1 unique value
    # 3. Prefer features that aren't pure binary (0/1) for better visualization
    features_for_pdp = []
    for f in pdp_candidate_features:
        if f in X_train.columns:
            unique_vals = X_train[f].nunique()
            if unique_vals > 1:
                # Prioritize non-binary features for better PDP visualization
                if unique_vals > 2:
                    features_for_pdp.append(f)
                elif len(features_for_pdp) < 3:  # Include some binary if we need more features
                    features_for_pdp.append(f)
        if len(features_for_pdp) >= 6:  # Limit to 6 features max
            break
    
    print(f"   Selected {len(features_for_pdp)} features for PDP analysis: {features_for_pdp}")

    if len(features_for_pdp) == 0:
        print("   ⚠️ No suitable features found for PDP generation. Skipping PDP.")
    else:
        try:
            # Using a sample of X_train for faster PDP generation
            X_pdp_background = X_train.sample(n=min(1000, X_train.shape[0]), random_state=42)
            
            # FIXED: Handle different numbers of features properly
            if len(features_for_pdp) == 1:
                print(f"   Generating single PDP for: {features_for_pdp[0]}")
                PartialDependenceDisplay.from_estimator(
                    model,
                    X_pdp_background,
                    features_for_pdp,
                    kind='average',
                    grid_resolution=50,
                    line_kw={"color": "dodgerblue", "linewidth": 2.5}
                )
                plt.suptitle("Partial Dependence Plot - Average Effect on Prediction", fontsize=16)
                plt.tight_layout()
                plt.show()
                
            elif len(features_for_pdp) == 2:
                print(f"   Generating PDPs for 2 features: {features_for_pdp}")
                PartialDependenceDisplay.from_estimator(
                    model,
                    X_pdp_background,
                    features_for_pdp,
                    kind='average',
                    n_cols=2,  # Safe to use 2 columns for 2 features
                    grid_resolution=50,
                    line_kw={"color": "dodgerblue", "linewidth": 2.5}
                )
                fig = plt.gcf()
                fig.suptitle("Partial Dependence Plots - Average Effect on Prediction", fontsize=16, y=1.03)
                plt.tight_layout(rect=[0, 0, 1, 0.98])
                plt.show()
                
            else:  # 3 or more features
                print(f"   Generating PDPs for {len(features_for_pdp)} features")
                # Limit to first 6 features to avoid overcrowding
                features_to_plot = features_for_pdp[:6]
                
                # Calculate appropriate number of columns
                n_cols = min(3, len(features_to_plot))  # Max 3 columns
                
                PartialDependenceDisplay.from_estimator(
                    model,
                    X_pdp_background,
                    features_to_plot,
                    kind='average',
                    n_cols=n_cols,
                    grid_resolution=50,
                    line_kw={"color": "dodgerblue", "linewidth": 2.5}
                )
                fig = plt.gcf()
                fig.suptitle("Partial Dependence Plots - Average Effect on Prediction", fontsize=16, y=1.03)
                # Dynamic figure size based on number of features
                n_rows = (len(features_to_plot) + n_cols - 1) // n_cols
                fig.set_size_inches(12, 4 * n_rows)
                plt.tight_layout(rect=[0, 0, 1, 0.98])
                plt.show()
                
            print("   ✓ One-way PDPs generated successfully.")
            print("      Interpretation: Shows the marginal effect of a feature on the model's prediction,")
            print("                    averaging out the effects of other features. Useful for understanding")
            print("                    how the model responds to changes in a single feature's value.")
            
        except Exception as e:
            print(f"❌ ERROR generating PDPs: {e}")
            print(f"   Debug info: features_for_pdp length = {len(features_for_pdp)}")
            print(f"   Features: {features_for_pdp}")

    # --- D. Method 3: Two-Way PDPs (Interaction Plots) - IMPROVED ---
    print("\n[4.4] Visualizing Feature Interactions with Two-Way PDPs...")
    
    if len(features_for_pdp) >= 2:
        # Select the top 2 most important features for interaction analysis
        interaction_features = features_for_pdp[:2]
        features_for_interaction_pdp = [(interaction_features[0], interaction_features[1])]
        
        print(f"   Generating Interaction PDP for: {interaction_features[0]} × {interaction_features[1]}")
        try:
            PartialDependenceDisplay.from_estimator(
                model, 
                X_pdp_background, 
                features_for_interaction_pdp, 
                kind='average',
                grid_resolution=20  # Lower resolution for 2D plots to improve speed
            )
            plt.suptitle(f"Interaction Partial Dependence Plot: {interaction_features[0]} × {interaction_features[1]}", 
                        fontsize=16, y=1.03)
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            plt.show()
            print("   ✓ Interaction PDP generated successfully.")
            print("      Interpretation: Shows how two features jointly affect the model's predictions,")
            print("                    revealing potential interaction effects between features.")
        except Exception as e:
            print(f"❌ ERROR generating interaction PDPs: {e}")
    else:
        print("   ⚠️ Not enough suitable features for interaction PDP. Skipping.")

    # --- E. Feature Correlation Analysis ---
    print("\n[4.5] Analyzing Feature Correlations...")
    try:
        # Get top features for correlation analysis
        top_features = feature_importance_df['feature'].head(15).tolist()
        top_features_in_data = [f for f in top_features if f in X_train.columns]
        
        if len(top_features_in_data) > 1:
            # Calculate correlation matrix
            corr_matrix = X_train[top_features_in_data].corr()
            
            # Plot correlation heatmap
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask for upper triangle
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f',
                       linewidths=0.5, cbar_kws={"shrink": .8})
            plt.title('Correlation Matrix of Top Important Features', fontsize=16)
            plt.tight_layout()
            plt.show()
            
            # Find highly correlated feature pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.5:  # Threshold for high correlation
                        high_corr_pairs.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j]
                        ))
            
            if high_corr_pairs:
                print("\n   Highly correlated feature pairs (|correlation| > 0.5):")
                for feat1, feat2, corr in high_corr_pairs:
                    print(f"     • {feat1} ↔ {feat2}: {corr:.3f}")
                print("\n      Note: High correlation between important features may indicate redundancy.")
                print("            Consider feature selection or dimensionality reduction techniques.")
            else:
                print("\n   No highly correlated pairs found among top features (threshold: |correlation| > 0.5).")
                print("      This suggests the top features provide complementary information to the model.")
        else:
            print("   ⚠️ Not enough features available for correlation analysis.")
    except Exception as e:
        print(f"❌ ERROR in feature correlation analysis: {e}")

    # --- F. Feature Importance Summary ---
    print("\n[4.6] Feature Importance Summary and Cumulative Contribution...")
    try:
        # Calculate cumulative importance
        top_n_summary = 30  # Analyze more features for the summary
        summary_df = feature_importance_df.head(top_n_summary).copy()
        summary_df['importance_percentage'] = summary_df['importance'] / summary_df['importance'].sum() * 100
        summary_df['cumulative_importance'] = summary_df['importance_percentage'].cumsum()
        
        # Find how many features contribute to 80% of importance
        features_for_80pct = len(summary_df[summary_df['cumulative_importance'] <= 80]) + 1
        
        # Plot cumulative importance
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(summary_df) + 1), summary_df['cumulative_importance'], 
                marker='o', linestyle='-', linewidth=2, markersize=8, color='darkblue')
        plt.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% Threshold')
        plt.xlabel('Number of Features', fontsize=12)
        plt.ylabel('Cumulative Importance (%)', fontsize=12)
        plt.title('Cumulative Feature Importance', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        print(f"\n   Key Insights:")
        print(f"   • The top {features_for_80pct} features account for 80% of the model's predictive power.")
        print(f"   • Most important feature: '{summary_df.iloc[0]['feature']}' ({summary_df.iloc[0]['importance_percentage']:.2f}%)")
        
        # Print top 10 features with their contribution
        print("\n   Top 10 Features by Importance:")
        print("   " + "-" * 70)
        print(f"   {'Rank':<5}{'Feature':<35}{'Importance %':<15}{'Cumulative %':<15}")
        print("   " + "-" * 70)
        for i, (_, row) in enumerate(summary_df.head(10).iterrows()):
            print(f"   {i+1:<5}{row['feature']:<35}{row['importance_percentage']:.2f}%{row['cumulative_importance']:.2f}%")
    except Exception as e:
        print(f"❌ ERROR in feature importance summary: {e}")

    # --- G. Additional Analysis: Feature Type Breakdown ---
    print("\n[4.7] Feature Type Analysis...")
    try:
        # Categorize features by type based on naming patterns
        feature_categories = {
            'Route Features': [f for f in feature_names if 'route' in f.lower()],
            'Sales Channel Features': [f for f in feature_names if 'sales_channel' in f.lower()],
            'Trip Type Features': [f for f in feature_names if 'trip_type' in f.lower()],
            'Lead Time Features': [f for f in feature_names if 'lead_time' in f.lower()],
            'Booking Origin Features': [f for f in feature_names if 'booking_origin' in f.lower()],
            'Flight Features': [f for f in feature_names if any(x in f.lower() for x in ['flight', 'duration', 'hour'])],
            'Passenger Preference Features': [f for f in feature_names if any(x in f.lower() for x in ['wants', 'baggage', 'seat', 'meal'])],
            'Other Features': []
        }
        
        # Assign uncategorized features to 'Other'
        categorized_features = set()
        for category_features in feature_categories.values():
            categorized_features.update(category_features)
        
        feature_categories['Other Features'] = [f for f in feature_names if f not in categorized_features]
        
        # Calculate importance by category
        category_importance = {}
        for category, features in feature_categories.items():
            if features:
                category_features_df = feature_importance_df[feature_importance_df['feature'].isin(features)]
                total_importance = category_features_df['importance'].sum()
                category_importance[category] = {
                    'total_importance': total_importance,
                    'feature_count': len(features),
                    'avg_importance': total_importance / len(features) if len(features) > 0 else 0
                }
        
        # Sort categories by total importance
        sorted_categories = sorted(category_importance.items(), 
                                 key=lambda x: x[1]['total_importance'], reverse=True)
        
        print("\n   Feature Category Analysis:")
        print("   " + "-" * 80)
        print(f"   {'Category':<30}{'Total Imp.':<12}{'Avg Imp.':<12}{'# Features':<12}")
        print("   " + "-" * 80)
        for category, stats in sorted_categories:
            if stats['feature_count'] > 0:
                print(f"   {category:<30}{stats['total_importance']:<12.4f}{stats['avg_importance']:<12.4f}{stats['feature_count']:<12}")
        
        print("\n   Interpretation: This breakdown shows which types of features")
        print("                 contribute most to the model's predictive power.")
        
    except Exception as e:
        print(f"❌ ERROR in feature type analysis: {e}")

    # --- H. Concluding Stage 4 ---
    print("\n" + "=" * 70)
    print("✅ STAGE 4: FEATURE CONTRIBUTION VISUALIZATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("   Review the visualizations and insights above to understand which features")
    print("   are driving your model's predictions and how they influence booking completion.")
    print("\n   Key Takeaways:")
    print("   • Feature importance rankings identify the most influential predictors")
    print("   • Partial dependence plots show how specific feature values affect predictions")
    print("   • Feature interactions reveal how combinations of features impact the model")
    print("   • Correlation analysis helps identify potential feature redundancies")
    print("   • Feature category analysis shows which types of information are most valuable")
    print("\n   Next Steps:")
    print("   • Use these insights to refine feature engineering in future iterations")
    print("   • Consider feature selection to create a more parsimonious model")
    print("   • Share key feature insights with business stakeholders")
    print("   • Validate that the important features align with domain knowledge")

if __name__ == "__main__":
    visualize_feature_contributions()