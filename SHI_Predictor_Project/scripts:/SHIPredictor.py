# %%
# Loading in the libraries and files we are going to use
import pandas as pd
cdc_filepath = '/Users/ruchirpatel/Downloads/CDCPlaces.csv'
sdoh2020_filepath = '/Users/ruchirpatel/Downloads/SDoH2020Data_cleaned.csv'

# %%
# --- Load the CDCPlaces.csv file ---
print(f"Attempting to load: {cdc_filepath}")

# Read the CSV file into a pandas DataFrame
# If the file is not found, this line will raise a FileNotFoundError
df_cdc_places = pd.read_csv(cdc_filepath)
    
print(f"\nSuccessfully loaded '{cdc_filepath}'")
    
# Display the first 5 rows of the DataFrame
print("\nFirst 5 rows of df_cdc_places:")
print(df_cdc_places.head())
    
# Print the shape of the DataFrame (rows, columns)
print(f"\nShape of df_cdc_places (rows, columns): {df_cdc_places.shape}")

# %%
location_id_dtype = df_cdc_places['LocationID'].dtype
print(f"Data type of 'LocationID' column: {location_id_dtype}")

# %%
# --- Standardize LocationID to CountyFIPS ---
print("\n--- Standardizing LocationID to CountyFIPS ---")

# Initialize df_cdc_places_filtered to ensure it exists even if 'LocationID' is missing
df_cdc_places_filtered = pd.DataFrame() 

if 'LocationID' in df_cdc_places.columns:
    # Convert LocationID to string and pad with leading zeros to ensure 5 digits.
    # A new column 'CountyFIPS' is created.
    df_cdc_places['CountyFIPS'] = df_cdc_places['LocationID'].astype(str).str.zfill(5)

    # Filter for rows where CountyFIPS is a 5-digit numeric string.
    # This creates a new DataFrame `df_cdc_places_filtered`.
    # .copy() is used to avoid SettingWithCopyWarning.
    df_cdc_places_filtered = df_cdc_places[df_cdc_places['CountyFIPS'].str.match(r'^\d{5}$')].copy()

    if not df_cdc_places_filtered.empty:
        print(f"\nSuccessfully created and filtered 'CountyFIPS' column.")
        print(f"Number of unique CountyFIPS codes in filtered data: {df_cdc_places_filtered['CountyFIPS'].nunique()}")
        print("Sample of CountyFIPS codes (first 5 unique values from filtered data):")
        print(df_cdc_places_filtered['CountyFIPS'].unique()[:5])
        print(f"\nShape of df_cdc_places_filtered (rows, columns): {df_cdc_places_filtered.shape}")
        # Note: df_cdc_places still exists as the original DataFrame but with an added 'CountyFIPS' column.
        # df_cdc_places_filtered is the one we'll likely use going forward.
    else:
        print("Warning: No valid 5-digit CountyFIPS found after processing LocationID.")
else:
    print("Error: 'LocationID' column was not found, so CountyFIPS cannot be created.")


# %%
# --- Select Core SDoH Measures ---
print("\n--- Selecting Core SDoH Measures from CDC Places Data ---")

# Initialize df_cdc_selected_measures to ensure it exists
df_cdc_selected_measures = pd.DataFrame()

if not df_cdc_places_filtered.empty:
    # User-specified core SDoH measures from CDCPlaces.csv
    core_cdc_measures = [
        'Food insecurity in the past 12 months among adults',
        'Received food stamps in the past 12 months among adults',
        'Housing insecurity in the past 12 months among adults',
        'Lack of reliable transportation in the past 12 months among adults',
        'Feeling socially isolated among adults',
        'Lack of social and emotional support among adults',
        'Frequent mental distress among adults'
    ]
    
    # Filter the DataFrame to keep only rows where the 'Measure' column is one of the core_cdc_measures
    df_cdc_selected_measures = df_cdc_places_filtered[df_cdc_places_filtered['Measure'].isin(core_cdc_measures)].copy() # Use .copy()

    if not df_cdc_selected_measures.empty:
        print(f"\nSuccessfully selected core SDoH measures.")
        print(f"Shape of df_cdc_selected_measures (rows, columns): {df_cdc_selected_measures.shape}")
        print("Unique measures selected:")
        print(df_cdc_selected_measures['Measure'].unique())
        print("\nFirst 5 rows of df_cdc_selected_measures:")
        print(df_cdc_selected_measures.head())
    else:
        print("Warning: No rows matched the core SDoH measures. df_cdc_selected_measures is empty.")
        print("Please check the 'core_cdc_measures' list and the 'Measure' column in your data.")
else:
    print("Skipping measure selection because df_cdc_places_filtered is empty or not created.")


# %%
# --- Pivot Selected Measures ---
print("\n--- Pivoting Selected SDoH Measures ---")

# Initialize df_cdc_county_pivot to ensure it exists
df_cdc_county_pivot = pd.DataFrame()

if not df_cdc_selected_measures.empty:
    # Select only the necessary columns for pivoting to simplify
    # We need CountyFIPS, StateAbbr, Year as index, Measure for new columns, Data_Value for values
    df_cdc_to_pivot = df_cdc_selected_measures[['CountyFIPS', 'StateAbbr', 'Year', 'Measure', 'Data_Value']].copy() # Use .copy()
    
    # Ensure Data_Value is numeric
    df_cdc_to_pivot['Data_Value'] = pd.to_numeric(df_cdc_to_pivot['Data_Value'], errors='coerce')
    
    # Handle potential duplicates again before pivot, ensuring one value per CountyFIPS/Measure/Year.
    # This sort_values and drop_duplicates is crucial if multiple entries for the same measure in the same year exist.
    # We are keeping the 'first' after sorting (Year is part of the index in pivot, so this handles if a County/Measure appears multiple times in the source for the same Year)
    # However, the earlier drop_duplicates on (CountyFIPS, Measure) after sorting by Year (desc) should have handled this.
    # This is an extra precaution or can be adjusted if a different aggregation is needed.
    df_cdc_to_pivot.sort_values(['CountyFIPS', 'Measure', 'Year'], ascending=[True, True, False], inplace=True)
    df_cdc_to_pivot.drop_duplicates(subset=['CountyFIPS', 'Measure', 'Year'], keep='first', inplace=True)

    df_cdc_county_pivot = df_cdc_to_pivot.pivot_table(
        index=['CountyFIPS', 'StateAbbr', 'Year'], 
        columns='Measure',              # This will make each unique measure a new column
        values='Data_Value'             # These will be the cell values
    ).reset_index()                     # reset_index() makes the index columns (CountyFIPS, etc.) regular columns again
    
    # Clean column names that result from the pivot
    # (e.g., replace spaces with underscores, shorten long measure names)
    cleaned_column_names = {}
    for col in df_cdc_county_pivot.columns:
        new_col_name = str(col) # Ensure it's a string
        new_col_name = new_col_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').replace(',', '')
        new_col_name = new_col_name.replace('in_the_past_12_months_among_adults', '_adults_12mo')
        cleaned_column_names[col] = new_col_name
    
    df_cdc_county_pivot.rename(columns=cleaned_column_names, inplace=True)
            
    if not df_cdc_county_pivot.empty:
        print(f"\nSuccessfully pivoted the data.")
        print(f"Shape of df_cdc_county_pivot (rows, columns): {df_cdc_county_pivot.shape}")
        print("\nFirst 5 rows of df_cdc_county_pivot:")
        print(df_cdc_county_pivot.head())
        print("\nColumns in df_cdc_county_pivot:")
        print(df_cdc_county_pivot.columns.tolist())
    else:
        print("Warning: Pivoting resulted in an empty DataFrame (df_cdc_county_pivot).")
else:
    print("Skipping pivot operation because df_cdc_selected_measures is empty or not created.")



# %%
# --- Step 2: Prepare SDoH2020Data_cleaned.csv ---
print("\n--- Preparing SDoH2020Data_cleaned.csv ---")

# Initialize df_sdoh_selected to ensure it exists
df_sdoh_selected = pd.DataFrame()

# List of SDoH columns originally identified by keywords for SDoH2020Data_cleaned.csv
# (excluding COUNTYFIPS as it's handled separately)
keyword_identified_sdoh_columns = [
    'ACS_PCT_CHILDREN_GRANDPARENT', 'ACS_PCT_TRANSPORT', 'ACS_PCT_HH_1FAM_FOOD_STMP', 'ACS_PCT_HH_FOOD_STMP',
    'ACS_PCT_HH_FOOD_STMP_BLW_POV', 'ACS_PCT_HH_NO_FD_STMP_BLW_POV', 'ACS_MEDIAN_HOME_VALUE', 'ACS_MEDIAN_RENT', 'ACS_PCT_1UP_RENT_1ROOM',
    'ACS_PCT_HU_MOBILE_HOME', 'ACS_PCT_OWNER_HU', 'ACS_PCT_OWNER_HU_CHILD', 'ACS_PCT_RENTER_HU', 'ACS_PCT_RENTER_HU_ABOVE65',
    'ACS_PCT_RENTER_HU_CHILD', 'ACS_PCT_RENTER_HU_COST_30PCT', 'ACS_PCT_RENTER_HU_COST_50PCT', 'ACS_PCT_VACANT_HU', 'ACS_PCT_HU_NO_FUEL',
    'ACS_PCT_HU_UTILITY_GAS', 'ACS_PCT_HU_BOT_TANK_LP_GAS', 'ACS_PCT_HU_OIL', 'ACS_PCT_HU_WOOD', 'ACS_PCT_HU_COAL',
    'ACS_PCT_HU_OTHER', 'ACS_PCT_HU_ELEC', 'ACS_PCT_HU_SOLAR', 'ACS_MDN_OWNER_COST_MORTGAGE', 'ACS_MDN_OWNER_COST_NO_MORTG',
    'ACS_PCT_OWNER_HU_COST_30PCT', 'ACS_PCT_OWNER_HU_COST_50PCT', 'ACS_PCT_HU_BUILT_1979', 'ACS_PCT_HU_KITCHEN', 'ACS_PCT_HU_PLUMBING',
    'ACS_PCT_HU_NO_VEH', 'ACS_PCT_WORK_NO_CAR', 'ACS_PCT_PUBL_TRANSIT', 'ACS_PCT_MEDICARE_ONLY', 'ACS_PCT_TRICARE_VA',
    'ACS_PCT_TRICARE_VA_BELOW64', 'CDCW_TRANSPORT_DTH_RATE', 'MP_MEDICARE_ELIGIBLES', 'MP_MEDICARE_ADVTG_ENROLLED', 'LTC_AVG_PCT_MEDICARE',
    'PC_PCT_MEDICARE_APPRVD_FULL_AMT', 'PC_PCT_MCARE_MAY_ACPT_APPRVD_AMT'
]

try:
    df_sdoh_full = pd.read_csv(sdoh2020_filepath) # Using the filepath variable defined at the top
    print(f"\nSuccessfully loaded '{sdoh2020_filepath}'")

    # Find the actual COUNTYFIPS column name (case-insensitive)
    actual_countyfips_col_sdoh = None
    for col in df_sdoh_full.columns: 
        if col.upper() == 'COUNTYFIPS':
            actual_countyfips_col_sdoh = col
            break
    
    if not actual_countyfips_col_sdoh:
        # Fallback for common alternatives if 'COUNTYFIPS' isn't found
        common_fips_alternates = ['FIPS', 'GEOID', 'COUNTY_FIPS']
        for alt_name in common_fips_alternates:
            for col_df in df_sdoh_full.columns:
                if col_df.upper() == alt_name.upper():
                    actual_countyfips_col_sdoh = col_df
                    print(f"Found alternative FIPS column in SDoH data: '{actual_countyfips_col_sdoh}'")
                    break
            if actual_countyfips_col_sdoh:
                break

    if not actual_countyfips_col_sdoh:
        raise ValueError("'COUNTYFIPS' or a recognizable alternative not found in SDoH2020Data_cleaned.csv for merging.")

    # Filter keyword-identified columns to include only those with actual values
    final_sdoh_columns_to_select = [actual_countyfips_col_sdoh] # Start with the FIPS column
    
    for col_template in keyword_identified_sdoh_columns:
        # Find actual casing of the column in df_sdoh_full
        actual_col_casing = None
        for col_in_file in df_sdoh_full.columns:
            if col_in_file.upper() == col_template.upper():
                actual_col_casing = col_in_file
                break
        
        if actual_col_casing:
            # Check if the column has at least one non-NaN value
            if df_sdoh_full[actual_col_casing].notna().any():
                if actual_col_casing not in final_sdoh_columns_to_select: # Avoid adding FIPS again or duplicates
                    final_sdoh_columns_to_select.append(actual_col_casing)
            else:
                print(f"Info: Column '{actual_col_casing}' from keyword list is all NaNs and will be excluded.")
        # If actual_col_casing is None, it means the column template wasn't found in df_sdoh_full
        # This case should be rare if keyword_identified_sdoh_columns was derived correctly.

    df_sdoh_selected = df_sdoh_full[final_sdoh_columns_to_select].copy() # Use .copy()
    
    # Standardize the FIPS column to 5-digit string
    df_sdoh_selected[actual_countyfips_col_sdoh] = df_sdoh_selected[actual_countyfips_col_sdoh].astype(str).str.zfill(5)
    
    # Rename the FIPS column to 'CountyFIPS' for consistent merging, if it's not already named that
    if actual_countyfips_col_sdoh != 'CountyFIPS': 
        df_sdoh_selected.rename(columns={actual_countyfips_col_sdoh: 'CountyFIPS'}, inplace=True)
        
    print(f"\ndf_sdoh_selected prepared. It has {df_sdoh_selected.shape[1]} columns (including CountyFIPS) after filtering for non-empty columns.")
    print(f"Shape of df_sdoh_selected (rows, columns): {df_sdoh_selected.shape}")
    print("\nFirst 5 rows of df_sdoh_selected:")
    print(df_sdoh_selected.head())
    print("\nColumns in df_sdoh_selected:")
    print(df_sdoh_selected.columns.tolist())


except FileNotFoundError:
    print(f"\nError: The file was not found at '{sdoh2020_filepath}'")
    print("Please double-check that the path and filename are correct.")
except ValueError as ve: # Catch specific error for missing FIPS
    print(f"\nValueError: {ve}")
except Exception as e:
    print(f"\nAn error occurred while processing '{sdoh2020_filepath}': {e}")



# %%
# --- Step 3: Merge the two prepared DataFrames ---
print("\n--- Merging DataFrames ---")

# Initialize df_merged_county to ensure it exists
df_merged_county = pd.DataFrame()

if not df_cdc_county_pivot.empty and not df_sdoh_selected.empty:
    print(f"Attempting to merge df_cdc_county_pivot (shape: {df_cdc_county_pivot.shape}) with df_sdoh_selected (shape: {df_sdoh_selected.shape}) on 'CountyFIPS'.")
    
    # Perform the merge using 'CountyFIPS' as the key
    # A 'left' merge keeps all rows from df_cdc_county_pivot and matching rows from df_sdoh_selected
    df_merged_county = pd.merge(df_cdc_county_pivot, df_sdoh_selected, on='CountyFIPS', how='left')
    
    if not df_merged_county.empty:
        print(f"\nSuccessfully merged DataFrames.")
        print(f"Shape of merged DataFrame (df_merged_county): {df_merged_county.shape}")
        print("\nFirst 5 rows of df_merged_county:")
        print(df_merged_county.head())
        
        print("\nNaN counts per column in df_merged_county (only columns with NaNs):")
        nan_counts = df_merged_county.isnull().sum()
        print(nan_counts[nan_counts > 0].sort_values(ascending=False))
        if nan_counts.sum() == 0:
            print("No NaN values found in the merged DataFrame.")
        
        # Check for rows from CDC PLACES that didn't get any SDoH data from the SDoH2020 file
        # These are rows where all columns that *came from* df_sdoh_selected (excluding CountyFIPS) are NaN
        sdoh_cols_to_check_for_nan = [col for col in df_sdoh_selected.columns if col != 'CountyFIPS']
        # Ensure these columns actually exist in the merged df before checking
        sdoh_cols_in_merged = [col for col in sdoh_cols_to_check_for_nan if col in df_merged_county.columns]

        if sdoh_cols_in_merged: 
            unmerged_cdc_rows = df_merged_county[df_merged_county[sdoh_cols_in_merged].isnull().all(axis=1)].shape[0]
            if unmerged_cdc_rows > 0:
                print(f"\nWarning: {unmerged_cdc_rows} counties from CDC PLACES data resulted in all NaN values for the SDoH columns (from SDoH2020Data_cleaned.csv) after the merge (likely no FIPS match).")
    else:
        print("Warning: Merging resulted in an empty DataFrame (df_merged_county).")
        
elif df_cdc_county_pivot.empty:
    print("Merge skipped: df_cdc_county_pivot is empty or not prepared.")
elif df_sdoh_selected.empty:
    print("Merge skipped: df_sdoh_selected is empty or not prepared.")
else:
    print("Merge skipped: Both df_cdc_county_pivot and df_sdoh_selected might be empty or another issue occurred.")



# %%
from sklearn.preprocessing import MinMaxScaler # Added for normalization

# --- Step 4: Handle Missing Data (NaNs) and Construct SHI Sub-Indices ---
print("\n--- Handling NaNs and Constructing SHI Sub-Indices ---")

if not df_merged_county.empty:
    # Define lists of variables for each sub-index based on actual column names in df_merged_county
    # These names should match the cleaned names from df_cdc_county_pivot and selected names from df_sdoh_selected
    
    # Dynamically get the exact CDC PLACES column names present in df_merged_county
    # (These were cleaned with a suffix like '_adults_12mo' and double underscores possibly)
    cdc_cols_map = {
        'food_insecurity': 'Food_insecurity__adults_12mo', # Check for double underscore first
        'food_stamps': 'Received_food_stamps__adults_12mo',
        'housing_insecurity': 'Housing_insecurity__adults_12mo',
        'transport_insecurity': 'Lack_of_reliable_transportation__adults_12mo',
        'social_isolation': 'Feeling_socially_isolated_among_adults', # This one might not have the suffix
        'lack_social_support': 'Lack_of_social_and_emotional_support_among_adults', # This one too
        'mental_distress': 'Frequent_mental_distress_among_adults' # And this one
    }
    
    actual_cdc_cols = {}
    for key, expected_name_pattern in cdc_cols_map.items():
        found_col = None
        # Try exact match first (could be from pivot cleaning)
        if expected_name_pattern in df_merged_county.columns:
            found_col = expected_name_pattern
        else: # Try alternative cleaning (single underscore)
            alt_name = expected_name_pattern.replace("__", "_")
            if alt_name in df_merged_county.columns:
                found_col = alt_name
        
        if found_col:
            actual_cdc_cols[key] = found_col
        else:
            print(f"Warning: CDC column for '{key}' (expected pattern: '{expected_name_pattern}') not found in df_merged_county.")
            actual_cdc_cols[key] = None # Mark as None if neither version is found
            
    # Define variables for each sub-index
    food_insecurity_vars = [
        actual_cdc_cols.get('food_insecurity'), 
        actual_cdc_cols.get('food_stamps'),
        'ACS_PCT_HH_1FAM_FOOD_STMP', 
        'ACS_PCT_HH_FOOD_STMP', 
        'ACS_PCT_HH_FOOD_STMP_BLW_POV', 
        'ACS_PCT_HH_NO_FD_STMP_BLW_POV'
    ]
    housing_insecurity_vars = [
        actual_cdc_cols.get('housing_insecurity'),
        'ACS_PCT_RENTER_HU_COST_30PCT', 
        'ACS_PCT_RENTER_HU_COST_50PCT',
        'ACS_PCT_OWNER_HU_COST_30PCT', # Cost burden for owners
        'ACS_PCT_OWNER_HU_COST_50PCT', 
        'ACS_PCT_VACANT_HU', 
        'ACS_PCT_HU_NO_FUEL', 
        'ACS_PCT_HU_PLUMBING', # Assuming higher % means more lacking these facilities
        'ACS_PCT_HU_KITCHEN'  # Assuming higher % means more lacking these facilities
    ]
    transport_insecurity_vars = [
        actual_cdc_cols.get('transport_insecurity'),
        'ACS_PCT_HU_NO_VEH', 
        'ACS_PCT_WORK_NO_CAR', 
        'CDCW_TRANSPORT_DTH_RATE',
        'ACS_PCT_PUBL_TRANSIT' # Higher public transit can sometimes indicate lack of other options
    ]
    social_isolation_vars = [
        actual_cdc_cols.get('social_isolation'), 
        actual_cdc_cols.get('lack_social_support'), 
        actual_cdc_cols.get('mental_distress'),
        'ACS_PCT_CHILDREN_GRANDPARENT' # Grandparents raising grandchildren can indicate stress/isolation
    ]

    # Filter out None values (columns not found) and ensure column exists in df_merged_county
    food_insecurity_vars = [v for v in food_insecurity_vars if v and v in df_merged_county.columns]
    housing_insecurity_vars = [v for v in housing_insecurity_vars if v and v in df_merged_county.columns]
    transport_insecurity_vars = [v for v in transport_insecurity_vars if v and v in df_merged_county.columns]
    social_isolation_vars = [v for v in social_isolation_vars if v and v in df_merged_county.columns]

    all_shi_vars = list(set(food_insecurity_vars + housing_insecurity_vars + transport_insecurity_vars + social_isolation_vars))

    # Convert all SHI variables to numeric, coercing errors.
    # Then impute NaNs that might result from coercion or were already present.
    imputation_summary = {}
    print("\nConverting SHI variables to numeric and performing median imputation...")
    for col in all_shi_vars:
        if col in df_merged_county.columns:
            df_merged_county[col] = pd.to_numeric(df_merged_county[col], errors='coerce')
            if df_merged_county[col].isnull().any():
                median_val = df_merged_county[col].median()
                if pd.notna(median_val):
                    df_merged_county[col].fillna(median_val, inplace=True)
                    imputation_summary[col] = median_val
                else:
                    # If median is NaN (e.g., column is all NaNs after coercion), fill with 0 or handle as error
                    print(f"Warning: Median for '{col}' is NaN after coercion. Filling remaining NaNs with 0 for this column.")
                    df_merged_county[col].fillna(0, inplace=True) 
        else:
            print(f"Warning: Column '{col}' intended for SHI was not found in df_merged_county during imputation.")
            
    if imputation_summary:
        print("Medians used for imputation (for columns that had NaNs and a valid median):")
        for col, med_val in imputation_summary.items():
            print(f"  '{col}': {med_val:.2f}")
    print("NaNs handled for SHI variables.")

    # Initialize the scaler
    scaler = MinMaxScaler()

    # Function to calculate sub-index
    def calculate_sub_index(df, var_list, index_name):
        # Ensure only valid, existing, and numeric columns are used
        valid_vars_for_index = [v for v in var_list if v in df.columns and pd.api.types.is_numeric_dtype(df[v])]
        
        # Further ensure no all-NaN columns are processed by scaler
        vars_with_data = []
        for v_col in valid_vars_for_index:
            if not df[v_col].isnull().all():
                vars_with_data.append(v_col)
            else:
                print(f"Warning: Column {v_col} for {index_name} is all NaNs. It will be excluded from scaling and index calculation.")
        
        valid_vars_for_index = vars_with_data

        if not valid_vars_for_index:
            print(f"No valid numeric variables with data found for {index_name}. Skipping index calculation.")
            df[index_name] = pd.NA 
            return df

        # Normalize: higher value = higher risk/insecurity (this is assumed for these vars)
        # Create a copy for normalization to avoid SettingWithCopyWarning
        df_normalized_subset = df[valid_vars_for_index].copy()
        df_normalized_subset[valid_vars_for_index] = scaler.fit_transform(df_normalized_subset[valid_vars_for_index])
        
        df[index_name] = df_normalized_subset.mean(axis=1)
        print(f"{index_name} calculated using variables: {valid_vars_for_index}")
        return df

    # Calculate each sub-index
    df_merged_county = calculate_sub_index(df_merged_county, food_insecurity_vars, 'Food_Insecurity_Index')
    df_merged_county = calculate_sub_index(df_merged_county, housing_insecurity_vars, 'Housing_Insecurity_Index')
    df_merged_county = calculate_sub_index(df_merged_county, transport_insecurity_vars, 'Transportation_Barriers_Index')
    df_merged_county = calculate_sub_index(df_merged_county, social_isolation_vars, 'Social_Isolation_Index')

    # Display Results
    print("\n--- SHI Sub-Indices Added (First 5 Rows) ---")
    shi_index_cols = ['CountyFIPS', 'StateAbbr', 'Food_Insecurity_Index', 'Housing_Insecurity_Index', 'Transportation_Barriers_Index', 'Social_Isolation_Index']
    
    # Check if all expected index columns were created and add them to a list for display
    shi_index_cols_present = [col for col in shi_index_cols if col in df_merged_county.columns]
    
    if shi_index_cols_present:
        print(df_merged_county[shi_index_cols_present].head())
        print("\n--- Descriptive Statistics for SHI Sub-Indices ---")
        # Get only the index columns that were successfully created for describe()
        desc_stats_cols = [col for col in shi_index_cols_present if 'Index' in col]
        if desc_stats_cols:
            print(df_merged_county[desc_stats_cols].describe())
        else:
            print("No SHI index columns available for descriptive statistics.")
    else:
        print("No SHI index columns were created or found to display.")
else:
    print("\nSkipping NaN handling and SHI construction because df_merged_county is empty or not prepared.")


# %%
# --- Step 5: Create an Overall Social Health Index (SHI) ---
print("\n--- Creating Overall Social Health Index (SHI) ---")

# Define the sub-index columns to average
sub_index_columns = []
if 'Food_Insecurity_Index' in df_merged_county.columns:
    sub_index_columns.append('Food_Insecurity_Index')
if 'Housing_Insecurity_Index' in df_merged_county.columns:
    sub_index_columns.append('Housing_Insecurity_Index')
if 'Transportation_Barriers_Index' in df_merged_county.columns:
    sub_index_columns.append('Transportation_Barriers_Index')
if 'Social_Isolation_Index' in df_merged_county.columns:
    sub_index_columns.append('Social_Isolation_Index')

if sub_index_columns: # Check if there are any sub-indices to average
    # Calculate the Overall_SHI by taking the mean of the sub-indices for each county
    # Ensure all sub-index columns are numeric before averaging
    for col in sub_index_columns:
        df_merged_county[col] = pd.to_numeric(df_merged_county[col], errors='coerce')
    
    # Handle any NaNs that might have been introduced by coercion or if a sub-index was all NA
    for col in sub_index_columns:
        if df_merged_county[col].isnull().any():
            print(f"Warning: Column '{col}' has NaNs before calculating Overall_SHI. Imputing with its median.")
            col_median = df_merged_county[col].median()
            if pd.notna(col_median):
                df_merged_county[col].fillna(col_median, inplace=True)
            else: # If sub-index column itself is all NaNs, its median will be NaN. Fill with 0.
                print(f"Warning: Median for sub-index '{col}' is NaN. Filling with 0 for Overall_SHI calculation.")
                df_merged_county[col].fillna(0, inplace=True)


    df_merged_county['Overall_SHI'] = df_merged_county[sub_index_columns].mean(axis=1)
    print("Overall_SHI calculated successfully.")

    # Display the first 5 rows with the new Overall_SHI
    print("\nFirst 5 rows of df_merged_county with Overall_SHI:")
    display_cols_overall = ['CountyFIPS', 'StateAbbr'] + sub_index_columns + ['Overall_SHI']
    # Ensure all columns in display_cols_overall actually exist
    display_cols_overall = [col for col in display_cols_overall if col in df_merged_county.columns]
    print(df_merged_county[display_cols_overall].head())

    # Print descriptive statistics for the Overall_SHI
    print("\nDescriptive statistics for Overall_SHI:")
    if 'Overall_SHI' in df_merged_county.columns:
        print(df_merged_county['Overall_SHI'].describe())
    else:
        print("Overall_SHI column not created.")
        
else:
    print("No sub-index columns found to calculate Overall_SHI.")


# Display Results (from previous step, for context of where Overall_SHI is added)
print("\n--- SHI Sub-Indices Added (First 5 Rows) - For Context ---")
shi_index_cols = ['CountyFIPS', 'StateAbbr', 'Food_Insecurity_Index', 'Housing_Insecurity_Index', 'Transportation_Barriers_Index', 'Social_Isolation_Index']

# Check if all expected index columns were created and add them to a list for display
shi_index_cols_present = [col for col in shi_index_cols if col in df_merged_county.columns]

if shi_index_cols_present:
    print(df_merged_county[shi_index_cols_present].head())
    print("\n--- Descriptive Statistics for SHI Sub-Indices - For Context ---")
    # Get only the index columns that were successfully created for describe()
    desc_stats_cols = [col for col in shi_index_cols_present if 'Index' in col]
    if desc_stats_cols:
        print(df_merged_county[desc_stats_cols].describe())
    else:
        print("No SHI index columns available for descriptive statistics.")
else:
    print("No SHI index columns were created or found to display.")

# %%
import requests
import pandas as pd

url = "https://www.huduser.gov/hudapi/public/usps?type=2&query=all" # Your modified URL
token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiI2IiwianRpIjoiNWJmYzE0NjFhMDM5NmU0ZWM2YTI5ZjZjM2NkZTc1YzcxM2NkZGIzM2FlNmFiYjYwZWY4NDM1NjllZThmMzYxYWYwZjgxZGQ4OTU0ZTRmYzciLCJpYXQiOjE3NDc4NjA3OTYuOTY4NDIxLCJuYmYiOjE3NDc4NjA3OTYuOTY4NDI0LCJleHAiOjIwNjMzOTM1OTYuOTU4MjQ5LCJzdWIiOiI5ODg3MyIsInNjb3BlcyI6W119.KPl_64bQ1vhWf-p8hGvTMftzV_l3wRx6pM7I3B58i5oZYOGWVBs7hdt0STjVCFiuiu9foD2Ow5xVX6WpjeDrbw' # Replace with your token
headers = {"Authorization": "Bearer {0}".format(token)}

print(f"Attempting HUD API call to: {url}")
response = requests.get(url, headers=headers)

df_hud_crosswalk_raw = pd.DataFrame() # Initialize

if response.status_code == 200:
    try:
        data = response.json()
        if "data" in data and "results" in data["data"]:
            df_hud_crosswalk_raw = pd.DataFrame(data["data"]["results"])
            print("\nHUD API call successful. Data loaded into df_hud_crosswalk_raw.")
            print(f"Shape of df_hud_crosswalk_raw: {df_hud_crosswalk_raw.shape}")
            print("\nFirst 5 rows of df_hud_crosswalk_raw:")
            print(df_hud_crosswalk_raw.head())
            print("\nColumns in df_hud_crosswalk_raw:")
            print(df_hud_crosswalk_raw.columns.tolist())
        else:
            print("API call successful, but 'data' or 'results' key missing in JSON response.")
            print("Response content:", response.text[:500]) # Print part of the response
    except requests.exceptions.JSONDecodeError:
        print("API call successful, but response was not valid JSON.")
        print("Response content:", response.text[:500]) # Print part of the response
else:
    print(f"API call failed. Status code: {response.status_code}")
    print("Response content:", response.text[:500]) # Print part of the response

# %%
# --- Step 6: Map County-Level SHI to ZIP Codes ---
print("\n--- Mapping County-Level SHI to ZIP Codes ---")

# Initialize df_zip_shi to ensure it exists
df_zip_shi = pd.DataFrame()


if 'df_hud_crosswalk_raw' in locals() and isinstance(df_hud_crosswalk_raw, pd.DataFrame) and not df_hud_crosswalk_raw.empty:
    print(f"Processing HUD crosswalk data (shape: {df_hud_crosswalk_raw.shape})...")
    df_crosswalk_processed = df_hud_crosswalk_raw.copy()

    # Standardize ZIP code column
    if 'zip' in df_crosswalk_processed.columns:
        df_crosswalk_processed['ZIP'] = df_crosswalk_processed['zip'].astype(str).str.zfill(5)
    else:
        print("Error: 'zip' column not found in HUD crosswalk data.")
        df_crosswalk_processed = pd.DataFrame() # Make it empty to skip merge

    # For type=2 API, 'geoid' is the CountyFIPS. Standardize it.
    if 'geoid' in df_crosswalk_processed.columns:
        df_crosswalk_processed['CountyFIPS'] = df_crosswalk_processed['geoid'].astype(str).str.zfill(5)
    else:
        print("Error: 'geoid' column not found in HUD crosswalk data. Cannot determine CountyFIPS.")
        df_crosswalk_processed = pd.DataFrame() # Make it empty

    if not df_crosswalk_processed.empty and 'ZIP' in df_crosswalk_processed.columns and 'CountyFIPS' in df_crosswalk_processed.columns:
        # Ensure res_ratio is present for sorting, if not, skip this sorting step
        if 'res_ratio' in df_crosswalk_processed.columns:
            df_crosswalk_processed['res_ratio'] = pd.to_numeric(df_crosswalk_processed['res_ratio'], errors='coerce').fillna(0)
            # Sort by ZIP and res_ratio (descending) to pick the primary county for a ZIP
            df_crosswalk_processed.sort_values(['ZIP', 'res_ratio'], ascending=[True, False], inplace=True)
            # Keep the first occurrence for each ZIP (which has the highest res_ratio)
            df_zip_to_county = df_crosswalk_processed.drop_duplicates(subset=['ZIP'], keep='first')
        else:
            print("Warning: 'res_ratio' column not found in HUD crosswalk. Using first available CountyFIPS for each ZIP.")
            df_zip_to_county = df_crosswalk_processed.drop_duplicates(subset=['ZIP'], keep='first')


        # Select only necessary columns for the final crosswalk
        if 'ZIP' in df_zip_to_county.columns and 'CountyFIPS' in df_zip_to_county.columns:
            df_zip_to_county = df_zip_to_county[['ZIP', 'CountyFIPS']].copy()
            print(f"Processed ZIP to County crosswalk. Shape: {df_zip_to_county.shape}")

            # Merge with df_merged_county (which has the SHI scores)
            if not df_merged_county.empty and not df_zip_to_county.empty:
                print(f"Attempting to merge SHI data with ZIP code crosswalk...")
                df_zip_shi = pd.merge(df_zip_to_county, df_merged_county, on='CountyFIPS', how='left')
                
                print(f"\nSuccessfully merged SHI data with ZIP codes.")
                print(f"Shape of final ZIP-level SHI DataFrame (df_zip_shi): {df_zip_shi.shape}")
                print("\nFirst 5 rows of df_zip_shi:")
                print(df_zip_shi.head())

                # Display NaN counts for the SHI columns in the ZIP-level data
                print("\nNaN counts for SHI columns in df_zip_shi:")
                shi_cols_in_zip_df = [col for col in ['Overall_SHI', 'Food_Insecurity_Index', 'Housing_Insecurity_Index', 'Transportation_Barriers_Index', 'Social_Isolation_Index'] if col in df_zip_shi.columns]
                if shi_cols_in_zip_df:
                    print(df_zip_shi[shi_cols_in_zip_df].isnull().sum())
                else:
                    print("No SHI index columns found in df_zip_shi to check NaNs.")
            else:
                print("Skipping merge with ZIP codes: df_merged_county or df_zip_to_county is empty.")
        else:
            print("Error: 'ZIP' or 'CountyFIPS' missing after processing HUD crosswalk. Cannot merge.")
    else:
        print("Skipping merge with ZIP codes: HUD crosswalk data was not processed correctly (e.g. missing 'zip' or 'geoid' columns).")
else:
    print("\nSkipping Step 6: Mapping to ZIP Codes.")
    print("Reason: 'df_hud_crosswalk_raw' DataFrame not found or is empty.")
    print("Please ensure you have run your HUD API call code (and it was successful),")
    print("and the result is stored in a Pandas DataFrame named 'df_hud_crosswalk_raw'.")
    print("For example, ensure this code runs successfully before this step:")
    print("""
# import requests
# url = "https://www.huduser.gov/hudapi/public/usps?type=2&query=all" # Or your specific query
# token = 'YOUR_VALID_HUD_API_TOKEN' 
# headers = {"Authorization": "Bearer {0}".format(token)}
# response = requests.get(url, headers=headers)
# if response.status_code == 200:
#     try:
#         df_hud_crosswalk_raw = pd.DataFrame(response.json()["data"]["results"])
#         print("df_hud_crosswalk_raw created from API.")
#     except Exception as e:
#         print(f"Error processing API JSON response: {e}")
#         df_hud_crosswalk_raw = pd.DataFrame() # Ensure it's an empty DataFrame on error
# else:
#     print(f"HUD API call failed with status: {response.status_code}")
#     df_hud_crosswalk_raw = pd.DataFrame() # Ensure it's an empty DataFrame on error
    """)


# %% [markdown]
# The following pieces of code are there to explore the results. The first code block is there to give us the top 10 ZIP codes by an overall social health index value. 
# The second block of code is there where one can input any zip code, and it will calculate the indices for a specifc measure, as well as an overall SHI.

# %%
# --- Explore Results: Top N ZIP Codes by Overall_SHI ---
if 'df_zip_shi' in locals() and not df_zip_shi.empty and 'Overall_SHI' in df_zip_shi.columns:
    print("\n--- Top 10 ZIP Codes with Highest Overall_SHI ---")
    # Sort by Overall_SHI in descending order and show top 10
    top_overall_shi_zips = df_zip_shi.sort_values(by='Overall_SHI', ascending=False).head(10)
    # Display relevant columns
    display_cols = ['ZIP', 'CountyFIPS', 'StateAbbr', 'Overall_SHI', 'Food_Insecurity_Index', 'Housing_Insecurity_Index', 'Transportation_Barriers_Index', 'Social_Isolation_Index']
    # Ensure all display_cols exist before trying to print
    display_cols_present = [col for col in display_cols if col in top_overall_shi_zips.columns]
    print(top_overall_shi_zips[display_cols_present])
else:
    print("df_zip_shi DataFrame or 'Overall_SHI' column not found. Please ensure previous steps ran successfully.")

# %%
# --- Explore Results: View SHI for a Specific ZIP Code ---
if 'df_zip_shi' in locals() and not df_zip_shi.empty:
    target_zip = '60062' # Replace with a ZIP code you want to check (ensure it's a 5-digit string)
    
    zip_data = df_zip_shi[df_zip_shi['ZIP'] == target_zip]
    
    if not zip_data.empty:
        print(f"\n--- SHI Scores for ZIP Code: {target_zip} ---")
        display_cols = ['ZIP', 'CountyFIPS', 'StateAbbr', 'Overall_SHI', 'Food_Insecurity_Index', 'Housing_Insecurity_Index', 'Transportation_Barriers_Index', 'Social_Isolation_Index']
        display_cols_present = [col for col in display_cols if col in zip_data.columns]
        print(zip_data[display_cols_present])
    else:
        print(f"\nZIP Code {target_zip} not found in the data or has no SHI scores.")
else:
    print("df_zip_shi DataFrame not found.")

# %% [markdown]
# The code block underneath is to validate the model as well as give context to the scores we provided

# %%
# --- Step 7: Validate and Contextualize SHI Scores ---
print("\n--- Validating and Contextualizing SHI Scores ---")

# Initialize df_zip_shi_cleaned_with_category for use in the function later
df_zip_shi_cleaned_with_category = pd.DataFrame()

if not df_merged_county.empty and 'Overall_SHI' in df_merged_county.columns:
    print("\n7.1 Correlation of County-Level Overall_SHI with other SDoH Indicators")
    
    validation_vars = []
    potential_val_cols_map = {
        'ACS_PCT_LT_HS': None, 
        'ACS_PER_CAPITA_INC': None,
        'ACS_PCT_UNEMPLOY': None 
    }
    
    for template_name in potential_val_cols_map.keys():
        for actual_col_name in df_merged_county.columns:
            if actual_col_name.upper() == template_name.upper():
                potential_val_cols_map[template_name] = actual_col_name
                break
                
    for template_name, actual_val_col in potential_val_cols_map.items():
        if actual_val_col: 
            df_merged_county[actual_val_col] = pd.to_numeric(df_merged_county[actual_val_col], errors='coerce')
            if df_merged_county[actual_val_col].isnull().any():
                val_col_median = df_merged_county[actual_val_col].median()
                if pd.notna(val_col_median):
                    df_merged_county[actual_val_col].fillna(val_col_median, inplace=True)
                else: 
                    print(f"Warning: Median for validation variable '{actual_val_col}' is NaN. It will be excluded from correlation.")
                    continue 
            validation_vars.append(actual_val_col)
        else:
            print(f"Warning: Validation variable template '{template_name}' not found in df_merged_county.")
            
    if validation_vars:
        df_merged_county['Overall_SHI'] = pd.to_numeric(df_merged_county['Overall_SHI'], errors='coerce')
        if df_merged_county['Overall_SHI'].isnull().any():
            overall_shi_median = df_merged_county['Overall_SHI'].median()
            if pd.notna(overall_shi_median):
                 df_merged_county['Overall_SHI'].fillna(overall_shi_median, inplace=True)
            else: 
                 print("Warning: Overall_SHI became all NaNs. Correlation might fail or be meaningless.")

        correlation_data = df_merged_county[['Overall_SHI'] + validation_vars].corr()
        print("Correlation Matrix:")
        print(correlation_data[['Overall_SHI']].sort_values(by='Overall_SHI', ascending=False))
    else:
        print("No valid validation variables found for correlation analysis.")

    print("\nInternal Consistency: Correlation of Overall_SHI with Sub-Indices (County-Level)")
    sub_indices_for_corr = [col for col in ['Food_Insecurity_Index', 'Housing_Insecurity_Index', 'Transportation_Barriers_Index', 'Social_Isolation_Index'] if col in df_merged_county.columns]
    if sub_indices_for_corr:
        for sub_idx_col in sub_indices_for_corr:
            df_merged_county[sub_idx_col] = pd.to_numeric(df_merged_county[sub_idx_col], errors='coerce')
            if df_merged_county[sub_idx_col].isnull().any(): 
                sub_idx_median = df_merged_county[sub_idx_col].median()
                if pd.notna(sub_idx_median):
                    df_merged_county[sub_idx_col].fillna(sub_idx_median, inplace=True)
                else:
                    df_merged_county[sub_idx_col].fillna(0, inplace=True)

        internal_corr_data = df_merged_county[['Overall_SHI'] + sub_indices_for_corr].corr()
        print("Correlation Matrix (Overall SHI vs Sub-Indices):")
        print(internal_corr_data[['Overall_SHI']].sort_values(by='Overall_SHI', ascending=False))
    else:
        print("No sub-index columns found for internal consistency check.")
else:
    print("Skipping County-Level SHI validation as df_merged_county or Overall_SHI is not available.")


if 'df_zip_shi' in locals() and isinstance(df_zip_shi, pd.DataFrame) and not df_zip_shi.empty and 'Overall_SHI' in df_zip_shi.columns:
    print("\n7.2 Contextualizing ZIP-Level SHI Scores")
    
    df_zip_shi['Overall_SHI'] = pd.to_numeric(df_zip_shi['Overall_SHI'], errors='coerce')
    # Use .copy() when creating df_zip_shi_cleaned to avoid SettingWithCopyWarning
    df_zip_shi_cleaned = df_zip_shi.dropna(subset=['Overall_SHI']).copy() 

    if not df_zip_shi_cleaned.empty:
        print("\n7.2.A Top 10 ZIP Codes with Highest Overall_SHI:")
        top_overall_shi_zips = df_zip_shi_cleaned.sort_values(by='Overall_SHI', ascending=False).head(10)
        display_cols_zip = ['ZIP', 'CountyFIPS', 'StateAbbr', 'Overall_SHI', 'Food_Insecurity_Index', 'Housing_Insecurity_Index', 'Transportation_Barriers_Index', 'Social_Isolation_Index']
        display_cols_zip_present = [col for col in display_cols_zip if col in top_overall_shi_zips.columns]
        print(top_overall_shi_zips[display_cols_zip_present])

        sample_zip_to_show = None
        if not top_overall_shi_zips.empty and 'ZIP' in top_overall_shi_zips.columns:
             sample_zip_to_show = top_overall_shi_zips['ZIP'].iloc[0] 
        
        if sample_zip_to_show and sample_zip_to_show in df_zip_shi_cleaned['ZIP'].values: 
            target_zip_score_series = df_zip_shi_cleaned[df_zip_shi_cleaned['ZIP'] == sample_zip_to_show]['Overall_SHI']
            if not target_zip_score_series.empty:
                target_zip_score = target_zip_score_series.iloc[0]
                percentile = (df_zip_shi_cleaned['Overall_SHI'] < target_zip_score).mean() * 100
                print(f"\n7.2.B Percentile Example: ZIP Code {sample_zip_to_show} (Overall_SHI: {target_zip_score:.4f}) is in approximately the {percentile:.2f}th percentile.")
            else:
                print(f"\nCould not find SHI score for sample ZIP {sample_zip_to_show} to demonstrate percentile.")
        else:
            print("\nCould not demonstrate percentile for a sample ZIP.")

        # Make a copy for adding SHI_Category to avoid SettingWithCopyWarning
        df_zip_shi_cleaned_with_category = df_zip_shi_cleaned.copy() 
        if not df_zip_shi_cleaned_with_category.empty: 
            try:
                category_labels_func = ['Low Concern', 'Medium-Low Concern', 'Medium-High Concern', 'High Concern']
                df_zip_shi_cleaned_with_category['SHI_Category'] = pd.qcut(df_zip_shi_cleaned_with_category['Overall_SHI'], q=4, labels=category_labels_func, duplicates='drop')
                
                print("\n7.2.C SHI Score Categories (based on quartiles of Overall_SHI for ZIPs with scores):")
                print(df_zip_shi_cleaned_with_category['SHI_Category'].value_counts(normalize=True).sort_index())
                
                print("\nFirst 5 ZIP Codes with their SHI Category:")
                display_cols_cat = ['ZIP', 'Overall_SHI', 'SHI_Category']
                display_cols_cat_present = [col for col in display_cols_cat if col in df_zip_shi_cleaned_with_category.columns]
                print(df_zip_shi_cleaned_with_category[display_cols_cat_present].head())
            except Exception as e:
                print(f"Error during categorization: {e}.")
                # Add SHI_Category column with a default error message if qcut fails
                df_zip_shi_cleaned_with_category['SHI_Category'] = "Error in categorization"
        else:
            print("No data available for SHI categorization after cleaning Overall_SHI.")
            
        # --- 7.2.E: View Full SHI Details and Concern Level for a Specific (Hardcoded) ZIP Code ---
        # This section demonstrates the lookup for a fixed ZIP code.
        print("\n--- 7.2.E: Example Lookup - SHI Details and Concern Level for a Specific ZIP Code ---")
        
        example_target_zip = '60062' # Fixed example ZIP code
        example_target_zip_str = str(example_target_zip).strip().zfill(5)

        if 'SHI_Category' in df_zip_shi_cleaned_with_category.columns:
            zip_data_example = df_zip_shi_cleaned_with_category[df_zip_shi_cleaned_with_category['ZIP'] == example_target_zip_str]
            
            if not zip_data_example.empty:
                print(f"\n--- SHI Scores and Concern Level for ZIP Code: {example_target_zip_str} ---")
                display_cols_specific = ['ZIP', 'CountyFIPS', 'StateAbbr', 
                                         'Overall_SHI', 'SHI_Category', 
                                         'Food_Insecurity_Index', 'Housing_Insecurity_Index', 
                                         'Transportation_Barriers_Index', 'Social_Isolation_Index']
                display_cols_specific_present = [col for col in display_cols_specific if col in zip_data_example.columns]
                print(zip_data_example[display_cols_specific_present])
            else:
                original_zip_info_example = df_zip_shi[df_zip_shi['ZIP'] == example_target_zip_str]
                if not original_zip_info_example.empty and original_zip_info_example['Overall_SHI'].isnull().all():
                     print(f"\nZIP Code {example_target_zip_str} found, but its Overall SHI score is NaN. Cannot determine concern level.")
                else:
                    print(f"\nZIP Code {example_target_zip_str} not found in the dataset with categorized SHI scores.")
        else:
            print("\n'SHI_Category' column not found in df_zip_shi_cleaned_with_category. Cannot display concern level for example ZIP.")
            print("Please ensure Step 7.2.C (Categorization) ran successfully.")

    else:
        print("No ZIP codes with valid Overall_SHI scores found for contextualization after dropping NaNs.")
else:
    print("Skipping ZIP-Level SHI contextualization as df_zip_shi or Overall_SHI is not available or empty.")


# %%
# --- 7.2.F: Interactive Lookup for ZIP Code SHI Concern Level ---
print("\n--- 7.2.F: Interactive Lookup for ZIP Code SHI Concern Level ---")
# This section allows the user to input a ZIP code.
# It relies on df_zip_shi_cleaned_with_category being created and having the 'SHI_Category' column.

if 'df_zip_shi_cleaned_with_category' in locals() and \
   isinstance(df_zip_shi_cleaned_with_category, pd.DataFrame) and \
   not df_zip_shi_cleaned_with_category.empty and \
   'SHI_Category' in df_zip_shi_cleaned_with_category.columns:

    try:
        target_zip_input_str = input("Enter a 5-digit ZIP code to check (or type 'exit' to skip): ")
        if target_zip_input_str.lower() != 'exit':
            target_zip_interactive = str(target_zip_input_str).strip().zfill(5)

            if not target_zip_interactive.isdigit() or len(target_zip_interactive) != 5:
                print(f"Invalid ZIP code entered: '{target_zip_input_str}'. Please enter a 5-digit number.")
            else:
                zip_data_interactive = df_zip_shi_cleaned_with_category[df_zip_shi_cleaned_with_category['ZIP'] == target_zip_interactive]
                
                if not zip_data_interactive.empty:
                    print(f"\n--- SHI Scores and Concern Level for ZIP Code: {target_zip_interactive} ---")
                    display_cols_interactive = ['ZIP', 'CountyFIPS', 'StateAbbr', 
                                                'Overall_SHI', 'SHI_Category', 
                                                'Food_Insecurity_Index', 'Housing_Insecurity_Index', 
                                                'Transportation_Barriers_Index', 'Social_Isolation_Index']
                    display_cols_interactive_present = [col for col in display_cols_interactive if col in zip_data_interactive.columns]
                    print(zip_data_interactive[display_cols_interactive_present])
                else:
                    original_zip_info_interactive = df_zip_shi[df_zip_shi['ZIP'] == target_zip_interactive]
                    if not original_zip_info_interactive.empty and original_zip_info_interactive['Overall_SHI'].isnull().all():
                         print(f"\nZIP Code {target_zip_interactive} found, but its Overall SHI score is NaN. Cannot determine concern level.")
                    else:
                        print(f"\nZIP Code {target_zip_interactive} not found in the dataset with categorized SHI scores.")
        else:
            print("Skipped interactive ZIP code lookup.")
    except Exception as e:
        print(f"An error occurred during interactive lookup: {e}")
else:
    print("\nInteractive ZIP lookup cannot be performed: 'df_zip_shi_cleaned_with_category' not available or 'SHI_Category' column missing.")
    print("Please ensure all previous steps, especially 7.2.C (Categorization), have run successfully.")




