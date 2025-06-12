import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder # Added for origin_route_pair
import os # For directory creation and file saving
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default') 
sns.set_palette("husl")

# --- Configuration ---
RAW_DATA_FILE = 'data/customer_booking.csv'
PROCESSED_OUTPUT_FILE = "data/final_processed_dataset_stage1.pkl"
OUTPUT_DATA_DIR = "data"


def load_and_inspect_data(file_path: str) -> pd.DataFrame | None:
    """Load dataset and perform initial inspection."""
    print("=" * 60)
    print("STAGE 1.1: DATA LOADING AND INITIAL INSPECTION")
    print("=" * 60)
    
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        print(f"✓ Dataset loaded successfully from '{file_path}'!")
        print(f"  Dataset shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at '{file_path}'.")
        print(f"  Please ensure the CSV file is in the '{os.path.dirname(file_path)}/' subdirectory or update the path.")
        return None
    
    print("\n" + "-" * 40 + "\nFirst 5 Rows:\n" + "-" * 40)
    print(df.head())
    
    print("\n" + "-" * 40 + "\nDataset Info:\n" + "-" * 40)
    df.info()
    
    print("\n" + "-" * 40 + "\nStatistical Summary (Numerical Features):\n" + "-" * 40)
    print(df.describe())

    print("\n" + "-" * 40 + "\nStatistical Summary (Object/Categorical Features):\n" + "-" * 40)
    print(df.describe(include=['object']))
    
    print("\n" + "-" * 40 + "\nMissing Values (per column):\n" + "-" * 40)
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(missing_values[missing_values > 0])
    else:
        print("  No missing values found in any column. Excellent!")
    
    return df

def clean_and_preprocess_data(df_input: pd.DataFrame) -> pd.DataFrame:
    """Clean, preprocess, and One-Hot Encode categorical features, ensuring integer dtypes."""
    print("\n" + "=" * 60)
    print("STAGE 1.2: DATA CLEANING, PREPROCESSING & ENCODING")
    print("=" * 60)
    
    df_processed = df_input.copy() 
    
    # --- Flight Day Mapping ---
    print("\n[1.2.1] Mapping 'flight_day' to numerical values (1-7)...")
    print(f"  Unique 'flight_day' values BEFORE mapping: {df_processed['flight_day'].unique()}")
    day_mapping = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7}
    df_processed['flight_day_numeric'] = df_processed['flight_day'].map(day_mapping)
    
    if df_processed['flight_day_numeric'].isnull().any():
        unmapped_days = df_processed[df_processed['flight_day_numeric'].isnull()]['flight_day'].unique()
        print(f"  ⚠ WARNING: Some 'flight_day' values were not mapped and resulted in NaN: {unmapped_days}")
        print(f"    Consider updating 'day_mapping' or implementing NaN handling for 'flight_day_numeric'.")
    else:
        print(f"  ✓ 'flight_day' successfully mapped to 'flight_day_numeric'.")
    print(f"  Unique 'flight_day_numeric' values AFTER mapping: {sorted(df_processed['flight_day_numeric'].dropna().unique())}")
    
    # --- One-Hot Encoding with dtype=int FIX ---
    print("\n[1.2.2] One-Hot Encoding categorical variables (ensuring integer dtype)...")
    
    low_cardinality_cols = ['sales_channel', 'trip_type']
    for col in low_cardinality_cols:
        print(f"  Encoding '{col}' (Unique values: {df_processed[col].nunique()})...")
        dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True, dtype=int) # FIX: dtype=int
        df_processed = pd.concat([df_processed, dummies], axis=1)
        print(f"    ✓ Created {dummies.shape[1]} new features for '{col}' (as int).")

    high_cardinality_cols = ['route', 'booking_origin'] 
    for col in high_cardinality_cols:
        unique_count = df_processed[col].nunique()
        print(f"  Encoding '{col}' (Unique values: {unique_count})...")
        if unique_count > 70: # Adjusted threshold for warning
             print(f"    ⚠ Note: '{col}' has high cardinality ({unique_count}). OHE will add many features.")
        dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True, dtype=int) # FIX: dtype=int
        df_processed = pd.concat([df_processed, dummies], axis=1)
        print(f"    ✓ Created {dummies.shape[1]} new features for '{col}' (as int).")
        
    # --- Boolean Column Type Confirmation ---
    print("\n[1.2.3] Confirming boolean-like columns are integer type (0/1)...")
    boolean_cols = ['wants_extra_baggage', 'wants_preferred_seat', 'wants_in_flight_meals', 'booking_complete']
    for col in boolean_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(int) # This was generally correct
            print(f"  ✓ '{col}' confirmed/converted to int. Unique values: {sorted(df_processed[col].unique())}")
        else:
            print(f"  ⚠ Warning: Expected boolean column '{col}' not found in DataFrame.")
            
    print(f"\n✓ Initial cleaning, mapping, and OHE complete.")
    print(f"  Dataset shape at this stage: {df_processed.shape}")
    print(f"  Note: Original categorical columns (e.g., 'route', 'flight_day') are still present for Feature Engineering.")
    return df_processed

def feature_engineering_and_drops(df_processed: pd.DataFrame, df_original_copy: pd.DataFrame) -> pd.DataFrame: 
    """Create new features from existing data and drop original/intermediate columns."""
    print("\n" + "=" * 60)
    print("STAGE 1.3: FEATURE ENGINEERING & FINAL COLUMN DROPS")
    print("=" * 60)
    
    # --- Route Popularity ---
    print("\n[1.3.1] Creating 'route_popularity'...")
    if 'route' in df_original_copy.columns: 
        route_counts = df_original_copy['route'].value_counts()
        df_processed['route_popularity'] = df_original_copy['route'].map(route_counts) # Use original for mapping
        print(f"  ✓ 'route_popularity' created. Range: {df_processed['route_popularity'].min()}-{df_processed['route_popularity'].max()}")
    else:
        print("  ⚠ 'route' column not found in original df for popularity calculation.")

    # --- Purchase Lead Binning & Encoding ---
    print("\n[1.3.2] Binning 'purchase_lead' and One-Hot Encoding (ensuring integer dtype)...")
    if 'purchase_lead' in df_processed.columns:
        lead_quantiles = df_processed['purchase_lead'].quantile([0.33, 0.66]).values
        bins = [0, lead_quantiles[0], lead_quantiles[1], df_processed['purchase_lead'].max() + 1] 
        labels = ['Short_Lead', 'Medium_Lead', 'Long_Lead']
        
        df_processed['purchase_lead_category'] = pd.cut(
            df_processed['purchase_lead'], 
            bins=bins, 
            labels=labels, 
            right=False, # Intervals like [0, q1), [q1, q2), ...
            include_lowest=True
        )
        print(f"  ✓ 'purchase_lead' binned into categories: {df_processed['purchase_lead_category'].value_counts().to_dict()}")
        
        lead_dummies = pd.get_dummies(df_processed['purchase_lead_category'], prefix='lead_time', drop_first=True, dtype=int) # FIX: dtype=int
        df_processed = pd.concat([df_processed, lead_dummies], axis=1)
        print(f"    ✓ Created {lead_dummies.shape[1]} new features for binned purchase lead (as int).")
        
        df_processed.drop(columns=['purchase_lead', 'purchase_lead_category'], axis=1, inplace=True)
        print(f"    ✓ Dropped original 'purchase_lead' and intermediate 'purchase_lead_category'.")
    else:
        print("  ⚠ 'purchase_lead' column not found for binning.")

    # --- Origin-Route Pair Creation ---
    print("\n[1.3.3] Creating 'origin_route_pair' (as a single string feature)...")
    if 'booking_origin' in df_original_copy.columns and 'route' in df_original_copy.columns: 
        df_processed['origin_route_pair'] = df_original_copy['booking_origin'] + '_to_' + df_original_copy['route']
        unique_pairs = df_processed['origin_route_pair'].nunique()
        print(f"  ✓ 'origin_route_pair' created with {unique_pairs} unique pairs.")
        print(f"    Note: This feature is kept as a single categorical string column (to be LabelEncoded later).")
    else:
        print("  ⚠ 'booking_origin' or 'route' column not found in original df for pair creation.")

    # --- Final Column Drops ---
    print("\n[1.3.4] Dropping original categorical columns used for OHE or mapping...")
    # These are the original string columns that have now been processed into new features.
    columns_to_drop_final = ['flight_day', 'sales_channel', 'trip_type', 'route', 'booking_origin']
    
    columns_to_drop_existing = [col for col in columns_to_drop_final if col in df_processed.columns]
    if columns_to_drop_existing:
        df_processed.drop(columns=columns_to_drop_existing, axis=1, inplace=True)
        print(f"  ✓ Dropped after Feature Engineering: {columns_to_drop_existing}")
    else:
        print("  ✓ No specified original categorical columns found to drop (likely already handled or not present).")

    print(f"\n✓ Feature Engineering and final column drops complete.")
    print(f"  Dataset shape at this stage: {df_processed.shape}")
    return df_processed


def apply_label_encoding(df_input: pd.DataFrame) -> pd.DataFrame:
    """Applies Label Encoding to high-cardinality 'origin_route_pair'."""
    print("\n" + "=" * 60)
    print("STAGE 1.4: LABEL ENCODING HIGH-CARDINALITY FEATURE")
    print("=" * 60)
    df_processed = df_input.copy()

    if 'origin_route_pair' in df_processed.columns:
        print("\n[1.4.1] Applying Label Encoding to 'origin_route_pair'...")
        le = LabelEncoder()
        df_processed['origin_route_pair_encoded'] = le.fit_transform(df_processed['origin_route_pair'])
        df_processed.drop(columns=['origin_route_pair'], inplace=True)
        print("  ✓ 'origin_route_pair' label encoded to 'origin_route_pair_encoded'.")
        print(f"  New column 'origin_route_pair_encoded' created.")
        print(f"  Number of unique encoded values: {df_processed['origin_route_pair_encoded'].nunique()}")
    else:
        print("\n  ✓ 'origin_route_pair' column not found for label encoding (might have been dropped or not created).")
    
    print(f"\n✓ Label Encoding step complete. Dataset shape: {df_processed.shape}")
    return df_processed


def exploratory_data_analysis(df_processed: pd.DataFrame):
    """Perform comprehensive EDA with visualizations on the processed data."""
    print("\n" + "=" * 60)
    print("STAGE 1.5: EXPLORATORY DATA ANALYSIS (Post-Processing)")
    print("=" * 60)
    
    if df_processed is None:
        print("  No data available for EDA.")
        return

    # --- Target Variable Distribution ---
    print("\n[1.5.1] Visualizing target variable 'booking_complete' distribution...")
    if 'booking_complete' in df_processed.columns:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        booking_counts = df_processed['booking_complete'].value_counts()
        booking_counts.plot(kind='bar', color=['lightcoral', 'lightblue'], edgecolor='black')
        plt.title('Booking Completion Distribution'); plt.ylabel('Count'); plt.xlabel('Booking Status (0=No, 1=Yes)')
        plt.xticks(ticks=[0,1], labels=['Not Completed (0)', 'Completed (1)'], rotation=0)
        
        plt.subplot(1, 2, 2)
        plt.pie(booking_counts, labels=['Not Completed', 'Completed'], 
               autopct='%1.2f%%', colors=['lightcoral', 'lightblue'], startangle=90)
        plt.title('Booking Completion Class Balance')
        plt.tight_layout(); plt.show()
        print(f"  ✓ Booking completion rate: {df_processed['booking_complete'].mean():.2%}")
        print(f"  ✓ Class balance (Counts): {booking_counts.to_dict()}")
    else:
        print("  ⚠ 'booking_complete' target variable not found for EDA.")

    # --- Numerical Features Distributions ---
    print("\n[1.5.2] Examining distributions of key numerical features...")
    # Define key numerical columns for focused EDA (avoiding too many OHE features in general plots)
    key_numerical_cols = ['num_passengers', 'length_of_stay', 
                           'flight_hour', 'flight_duration', 'flight_day_numeric', 
                           'route_popularity'] 
    if 'origin_route_pair_encoded' in df_processed.columns: # Add if it was created
        key_numerical_cols.append('origin_route_pair_encoded')
    
    # Also include OHE features derived from 'purchase_lead_category'
    lead_time_ohe_cols = [col for col in df_processed.columns if col.startswith('lead_time_')]
    numerical_eda_cols = [col for col in key_numerical_cols + lead_time_ohe_cols if col in df_processed.columns]

    if numerical_eda_cols:
        num_plots = len(numerical_eda_cols); num_cols_plot = 3
        num_rows_plot = (num_plots + num_cols_plot - 1) // num_cols_plot # Calculate rows needed
        
        print("  Generating Histograms for numerical features...")
        fig, axes = plt.subplots(num_rows_plot, num_cols_plot, figsize=(15, 5 * num_rows_plot))
        axes = axes.flatten() # Flatten in case of single row/col
        for i, col in enumerate(numerical_eda_cols):
            sns.histplot(df_processed[col], kde=True, ax=axes[i], bins=30, color='skyblue')
            axes[i].set_title(f'Distribution of {col}')
        for j in range(i + 1, len(axes)): axes[j].set_visible(False) # Hide unused subplots
        plt.tight_layout(); plt.show()

        if 'booking_complete' in df_processed.columns:
            print("  Generating Box Plots for numerical features vs. Booking Completion...")
            fig, axes = plt.subplots(num_rows_plot, num_cols_plot, figsize=(15, 5 * num_rows_plot))
            axes = axes.flatten()
            for i, col in enumerate(numerical_eda_cols):
                sns.boxplot(x='booking_complete', y=col, data=df_processed, ax=axes[i], palette=['lightcoral', 'lightblue'])
                axes[i].set_title(f'{col} vs Booking Completion')
            for j in range(i + 1, len(axes)): axes[j].set_visible(False)
            plt.tight_layout(); plt.show()
    else:
        print("  No specific numerical columns identified for distribution plots.")

    # --- Correlation Heatmap ---
    print("\n[1.5.3] Creating Correlation Heatmap for key numerical features...")
    # Use the same 'key_numerical_cols' for the heatmap for readability
    heatmap_display_cols = key_numerical_cols[:] # Create a copy
    if 'booking_complete' in df_processed.columns and 'booking_complete' not in heatmap_display_cols:
        heatmap_display_cols.append('booking_complete')
    
    if len(heatmap_display_cols) > 1:
        # Ensure all columns for heatmap are actually numeric in the DataFrame
        numeric_heatmap_cols = df_processed[heatmap_display_cols].select_dtypes(include=np.number).columns.tolist()
        if len(numeric_heatmap_cols) >= 2 :
            correlation_matrix = df_processed[numeric_heatmap_cols].corr()
            plt.figure(figsize=(max(10, len(numeric_heatmap_cols)*0.8), max(8, len(numeric_heatmap_cols)*0.6))) # Dynamic size
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 8})
            plt.title('Correlation Heatmap of Key Numerical Features')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout(); plt.show()

            if 'booking_complete' in correlation_matrix.columns:
                print("\n  Correlations with 'booking_complete':")
                print(correlation_matrix['booking_complete'].sort_values(ascending=False).to_string())
        else:
             print("  Not enough numeric columns found from the selected key features for heatmap.")
    else:
        print("  Not enough key numerical columns selected for a correlation heatmap.")
        
    print(f"\n✓ EDA visualizations generated.")


def main_stage1_pipeline():
    """Orchestrates the full Stage 1 data processing pipeline."""
    print("🚀 INITIATING CUSTOMER BOOKING ML PIPELINE - STAGE 1 (v5 - DTYPE FIXES)")
    print("   Data Exploration, Preparation, and Feature Engineering")
    print("=" * 70)
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DATA_DIR):
        os.makedirs(OUTPUT_DATA_DIR)
        print(f"✓ Created directory for data I/O: '{OUTPUT_DATA_DIR}'")

    # 1.1 Load and Inspect
    df_original = load_and_inspect_data(file_path=RAW_DATA_FILE) 
    if df_original is None:
        print("\n❌ Halting Stage 1 pipeline due to data loading error.")
        return None

    # 1.2 Clean, Preprocess (Initial OHE)
    df_temp_processed = clean_and_preprocess_data(df_original.copy()) # Pass a copy

    # 1.3 Feature Engineering and Final Column Drops
    df_engineered = feature_engineering_and_drops(df_temp_processed, df_original.copy()) 
    
    # 1.4 Label Encode 'origin_route_pair'
    df_final_for_eda = apply_label_encoding(df_engineered)
    
    # 1.5 Exploratory Data Analysis on the fully processed data
    exploratory_data_analysis(df_final_for_eda)
    
    # --- Final Summary and Output ---
    print("\n" + "=" * 60)
    print("STAGE 1 FINAL SUMMARY & OUTPUT")
    print("=" * 60)
    final_df_stage1 = df_final_for_eda # Assign for clarity before saving

    if final_df_stage1 is not None:
        print(f"  Final dataset shape after all Stage 1 processing: {final_df_stage1.shape}")
        if df_original is not None: 
             print(f"  Original columns: {df_original.shape[1]}")
             print(f"  Final features (including target): {final_df_stage1.shape[1]}")
             print(f"  Net change in columns: {final_df_stage1.shape[1] - df_original.shape[1]}")
        
        final_cols = list(final_df_stage1.columns)
        print(f"\n  First 10 columns of final dataset: {final_cols[:10]}")
        if len(final_cols) > 15: 
            print(f"  Last 5 columns of final dataset: {final_cols[-5:]}")
        
        # Save the processed DataFrame
        try:
            final_df_stage1.to_pickle(PROCESSED_OUTPUT_FILE)
            print(f"\n✓ Stage 1 output successfully saved to: '{PROCESSED_OUTPUT_FILE}'")
        except Exception as e:
            print(f"\n❌ ERROR saving Stage 1 output to '{PROCESSED_OUTPUT_FILE}': {e}")
    else:
        print("  ❌ No final DataFrame produced from Stage 1.")


    print(f"\n✅ STAGE 1 (v5 - DTYPE FIXES) COMPLETED SUCCESSFULLY!")
    print(f"   All One-Hot Encoded columns are now explicitly integers.")
    print(f"   'origin_route_pair' has been Label Encoded.")
    print(f"   The dataset should be fully numerical and ready for Stage 2: Model Training.")
    
    return final_df_stage1

if __name__ == "__main__":
    processed_dataframe = main_stage1_pipeline()
    if processed_dataframe is not None:
        print("\n--- First 5 rows of the final processed DataFrame (output of Stage 1) ---")
        print(processed_dataframe.head())
        print("\n--- Data types of the final processed DataFrame (Stage 1 output) ---")
        print(processed_dataframe.dtypes.value_counts()) # To verify dtypes