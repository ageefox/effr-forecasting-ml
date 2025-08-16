# working with dataframes
import pandas as pd

# Load datasets
# loading the dataset
index_data = pd.read_csv("index 2.csv")
# loading the dataset
cpi_data = pd.read_csv("US CPI.csv")

# Convert date columns to datetime
index_data['date'] = pd.to_datetime(index_data[['Year', 'Month', 'Day']])
cpi_data['date'] = pd.to_datetime(cpi_data['Yearmon'], format='%d-%m-%Y')

# Merge CPI into index data
# combining datasets
merged_df = pd.merge(index_data, cpi_data[['date', 'CPI']], on='date', how='left')

# Interpolate macroeconomic features (forward fill where safe)
# filling in missing values
merged_df['Inflation Rate'] = merged_df['Inflation Rate'].interpolate(method='linear', limit_direction='forward')
# filling in missing values
merged_df['Unemployment Rate'] = merged_df['Unemployment Rate'].interpolate(method='linear', limit_direction='forward')
# filling in missing values
merged_df['CPI'] = merged_df['CPI'].interpolate(method='linear', limit_direction='forward')

# Drop rows where the target variable is missing
# combining datasets
merged_df = merged_df[merged_df['Effective Federal Funds Rate'].notna()].reset_index(drop=True)

# Forward-fill Real GDP (quarterly → monthly)
# combining datasets
if 'Real GDP (Percent Change)' in merged_df.columns:
# combining datasets
    merged_df['Real GDP (Percent Change)'] = merged_df['Real GDP (Percent Change)'].fillna(method='ffill')

    columns_to_drop = [
    'Federal Funds Target Rate',
    'Federal Funds Upper Target',
    'Federal Funds Lower Target',
    'Year', 'Month', 'Day' ]
    
# combining datasets
merged_df = merged_df.drop(columns=[col for col in columns_to_drop if col in merged_df.columns])

# Final cleaned dataset
# combining datasets
cleaned_df = merged_df

cleaned_df.to_csv("cleaned_effr_data.csv", index=False)

