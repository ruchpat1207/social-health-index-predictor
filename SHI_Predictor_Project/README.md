# Social Health Index (SHI) Predictor Prototype

## Overview

This project presents a prototype model for calculating a Social Health Index (SHI) at the U.S. county and ZIP code levels. It utilizes publicly available Social Determinants of Health (SDoH) data to quantify and map social health challenges across different geographic areas. The primary goal is to identify areas with significant social health challenges based on key SDoH domains such as food insecurity, housing insecurity, transportation barriers, and social isolation.

This project was developed to demonstrate data integration, preprocessing, index construction, and results interpretation using Python and Pandas.

## Data Sources

The SHI predictor was developed using the following primary data sources:

1.  **CDC PLACES Data (`CDCPlaces.csv`):**
    * **Origin:** Centers for Disease Control and Prevention (CDC).
    * **Purpose:** Provided core SDoH-related health outcome and behavior measures at the county level.
    * **Access:** Can be found on the CDC PLACES website. (e.g., `https://www.cdc.gov/places/`)
    * **Note:** This file was used locally for development.
      
2.  **Consolidated SDoH Indicators (`SDoH2020Data_cleaned.csv`):**
    * **Origin:** A comprehensive dataset, likely derived primarily from the U.S. Census Bureau's American Community Survey (ACS) for 2020 and potentially other public sources. (Referred to as `SDOH2020.csv` in the notebook's original file path).
    * **Purpose:** Provided supplementary contextual SDoH variables related to demographics, economic status, housing characteristics, etc., at the county level.
    * **Access:** https://www.ahrq.gov/sdoh/data-analytics/sdoh-data.html
    * **Note:** This file was used locally for development.
    
3.  **HUD USPS Crosswalk Data (ZIP Code to County FIPS Mapping):**
    * **Origin:** U.S. Department of Housing and Urban Development (HUD) User Portal API – USPS Validation Service.
    * **Purpose:** Used to map county-level SHI scores to ZIP codes.
    * **API Endpoint Used:** `https://www.huduser.gov/hudapi/public/usps?type=2&query=all` (as explored in the notebook).
    * **Note:** Accessing this API requires an authenticated token from HUD. The notebook (`SHI_Predictor.ipynb`) contains the logic for making this API call. Users wishing to replicate this step will need to obtain their own HUD API token and insert it into the relevant section of the notebook.

## Methodology Summary

The development of the SHI predictor involved the following key steps:
1.  **Data Loading & Cleaning:** Ingesting `CDCPlaces.csv` and `SDoH2020Data_cleaned.csv`.
2.  **Geographic Identifier Standardization:** Ensuring County FIPS codes were consistently formatted as 5-digit strings.
3.  **SDoH Variable Selection:**
    * Selecting seven core SDoH measures from `CDCPlaces.csv`.
    * Selecting relevant supplementary variables from `SDoH2020Data_cleaned.csv` (after filtering for columns with actual data).
4.  **Data Transformation:** Pivoting the selected CDC PLACES data to a wide format.
5.  **DataFrame Merging:** Combining the processed CDC PLACES data with the selected SDoH2020 variables at the county level.
6.  **Missing Value Imputation:** Handling NaNs in SHI constituent variables using median imputation.
7.  **SHI Sub-Index Construction:**
    * Developing four sub-indices: Food Insecurity, Housing Insecurity, Transportation Barriers, and Social Isolation.
    * Normalizing constituent variables for each sub-index to a [0, 1] scale using Min-Max scaling.
    * Calculating sub-index scores by averaging their normalized components.
8.  **Overall SHI Calculation:** Computing a composite `Overall_SHI` by averaging the four sub-index scores.
9.  **ZIP Code Mapping:** Processing the HUD USPS crosswalk data and merging it with county-level SHI scores.
10. **Results Exploration & Contextualization:** Analyzing SHI score distributions, identifying high-risk ZIP codes, and categorizing ZIP codes by level of concern.

For a detailed technical breakdown of the methodology, data processing steps, and analytical decisions, please see the PDF report: **`Social Health Index (SHI) Predictor Prototype_ In-Depth Technical Report.pdf`**. *(You may want to create a `METHODOLOGY.md` file later for easier web viewing of this report).*

## Files in this Repository

* `SHIPredictor.ipynb`: The main Jupyter Notebook containing all Python code for data loading, processing, SHI calculation, and results exploration.
* `SHIPredictor.py`: A Python script version exported from the main Jupyter Notebook. This can be used to view the code or potentially run the analysis pipeline if all dependencies and data paths are correctly managed in a script-based environment.
* `Social Health Index (SHI) Predictor Prototype_ In-Depth Technical Report.pdf`: The detailed technical report for the project in PDF format.
* `SHIPredictor.html`: An HTML exported version of the Jupyter Notebook, providing an easily viewable static report of the analysis, code, and results.
* `README.md`: This file.

## How to Run the Code

### Prerequisites
* Python 3.x (e.g., Python 3.8 or newer recommended)
* Required Python libraries: `pandas`, `scikit-learn`, `requests`. 

### Setup
1.  **Download or Clone the Repository:**
    * You can download the files as a ZIP from GitHub or clone the repository if you have Git installed:
        ```bash
        git clone https://github.com/ruchpat1207/social-health-index-predictor.git
        ```
2.  **Obtain Data Files:**
    * As mentioned in "Data Sources," the large CSV files (`CDCPlaces.csv`, `SDoH2020Data_cleaned.csv`) are likely not included here. You will need to download them from their respective official sources.
    * Place these downloaded CSV files in the same directory as the `SHI_Predictor.ipynb` notebook, or update the filepaths at the beginning of the notebook to point to their correct location on your system.
3.  **HUD API Token (for ZIP Code Mapping):**
    * The Jupyter Notebook (`SHI_Predictor.ipynb`) includes code to fetch ZIP-to-County crosswalk data from the HUD API. To run this part successfully, you will need to:
        * Obtain a free API token from the [HUD User Portal](https://www.huduser.gov/portal/dataset/usps-api.html).
        * Replace the placeholder token string `YOUR_ACTUAL_VALID_HUD_API_TOKEN` in the relevant cell of the notebook with your own valid token.
4.  **Install Dependencies (if you create a `requirements.txt` later):**
    ```bash
    # pip install -r requirements.txt 
    ```
    For now, ensure you have pandas, scikit-learn, and requests installed in your Python environment.

### Execution
1.  **Primary Method (Jupyter Notebook):**
    * Launch your Jupyter environment (e.g., Jupyter Lab, Jupyter Notebook, or VS Code with the Jupyter extension).
    * Open the `SHI_Predictor.ipynb` notebook.
    * Ensure the filepaths for `CDCPlaces.csv` and `SDoH2020Data_cleaned.csv` (and your HUD API token) are correctly set within the notebook.
    * Run the notebook cells sequentially.
2.  **Alternative (Python Script):**
    * The `SHI_Predictor_export.py` file can be reviewed to see the code structure. Running it directly as a script might require modifications to handle file paths and any interactive elements (like `input()` prompts) differently than in a notebook.

## Key Outputs

The primary output of this project is the `df_zip_shi` Pandas DataFrame generated by the Jupyter Notebook. This DataFrame contains:
* `ZIP`: The 5-digit ZIP code.
* `CountyFIPS`: The primary County FIPS code associated with the ZIP code.
* `Overall_SHI`: The composite Social Health Index score (scaled 0-1, where higher indicates greater SDoH challenges).
* Sub-Indices: `Food_Insecurity_Index`, `Housing_Insecurity_Index`, `Transportation_Barriers_Index`, `Social_Isolation_Index`.
* `SHI_Category`: A qualitative assessment ('Low Concern' to 'High Concern') based on the `Overall_SHI`.

The notebook also demonstrates how to query for specific ZIP codes and view their detailed SHI profiles. The `SHI_Predictor_Results_Export.html` file provides a static view of the notebook's execution and outputs.

## Conceptual LLM Integration

A conceptual discussion on how Large Language Models (LLMs) could be leveraged in future iterations to streamline SDoH data integration (e.g., for automated variable annotation) is included within the detailed technical report (`Social Health Index (SHI) Predictor Prototype_ In-Depth Technical Report.pdf`).

---
*Developed by Ruchir Patel*
