import ast
import os
import pandas as pd
import glob
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

# Folder paths
input_folder = r'D:\Capstone projects\cardekho\Unstructured_files'
output_folder = r'D:\Capstone projects\cardekho\Structured_files'

# Ensure the output folder exists, if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to flatten nested dictionaries and lists
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, f'{new_key}_{i}', sep=sep).items())
                else:
                    items.append((f'{new_key}_{i}', item))
        else:
            items.append((new_key, v))
    return dict(items)

# Columns to be flattened
columns_to_flatten = ['new_car_detail', 'new_car_overview', 'new_car_feature', 'new_car_specs']

# Process each Excel file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.xlsx'):
        # Define file paths
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace('.xlsx', '.csv'))
        
        # Load the Excel file
        df = pd.read_excel(input_path, engine='openpyxl')
        
        # Process and flatten relevant columns
        structured_data = {}
        
        # Loop through each column that needs flattening
        for col in columns_to_flatten:
            if col in df.columns:
                # Process only if the column is not empty
                structured_column_data = []
                for value in df[col]:
                    if pd.notna(value):
                        try:
                            # Convert string representation of dictionaries to actual dictionaries
                            parsed_value = ast.literal_eval(value)
                            structured_column_data.append(flatten_dict(parsed_value))
                        except (ValueError, SyntaxError):
                            # Handle cases where the value cannot be parsed as a dictionary
                            structured_column_data.append({col: value})
                    else:
                        structured_column_data.append({col: None})
                
                structured_data[col] = pd.DataFrame(structured_column_data)

        # Keep the 'car_links' column as it is
        if 'car_links' in df.columns:
            structured_data['car_links'] = df[['car_links']].copy()

        # Merge all structured DataFrames into one
        structured_dfs = [structured_data[col] for col in structured_data]
        structured_cars_df = pd.concat(structured_dfs, axis=1)

        # Add a 'City' column based on the filename (e.g., 'chennai_cars.xlsx' -> 'Chennai')
        city_name = filename.replace('_cars.xlsx', '').capitalize()
        structured_cars_df['City'] = city_name

        # Save the structured DataFrame to a CSV file
        structured_cars_df.to_csv(output_path, index=False)

        # Display the merged DataFrame for each file (optional)
        print(f"Processed {filename}:")
        print(structured_cars_df.head())  # Shows the first few rows of the processed DataFrame


# Define the folder path where your CSV files are stored
folder_path = r'D:\Capstone projects\cardekho\Structured_files'

# Use glob to get a list of all CSV file paths in the folder
file_paths = glob.glob(os.path.join(folder_path, '*.csv'))

# Function to load and concatenate datasets
def load_and_concatenate_datasets(file_paths):
    dataframes = []
    for file_path in file_paths:
        # Extract city name from the file path
        city_name = os.path.basename(file_path).replace('_cars.csv', '')
        
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Add a 'City' column to the DataFrame
        df['City'] = city_name.capitalize()
        
        # Append the DataFrame to the list
        dataframes.append(df)
    
    # Concatenate all DataFrames
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

# Load and concatenate the data
combined_df = load_and_concatenate_datasets(file_paths)

# Define the output directory and file name for the combined dataset
output_dir = r'D:\Capstone projects\cardekho'
output_file = 'car_dekho_Structured.csv'
output_path = os.path.join(output_dir, output_file)

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the combined DataFrame to a CSV file
combined_df.to_csv(output_path, index=False)

print(f"All datasets concatenated and saved to {output_path}")

import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the dataset
df = pd.read_csv(r'D:\Capstone projects\cardekho\car_dekho_Structured.csv', low_memory=False)

# Drop columns with more than 50% missing values
threshold = len(df) * 0.5
df.dropna(thresh=threshold, axis=1, inplace=True)

# Helper function to convert price from various formats to float
def convert_price(price):
    try:
        price_str = str(price).replace('₹', '').replace(',', '').strip()
        if 'Lakh' in price_str:
            return float(price_str.replace('Lakh', '').strip()) * 100000
        return float(price_str)
    except ValueError:
        return np.nan

# Apply conversion function to the 'price' column
df['price'] = df['price'].apply(convert_price)

# Clean the 'km' column by removing commas and converting to float
df['km'] = df['km'].str.replace('Kms', '').str.replace(',', '').astype(float)

# Fill missing values for numerical columns with the median
df.fillna({
    'price': df['price'].median(),
    'ownerNo': df['ownerNo'].median(),
    'km': df['km'].median()
}, inplace=True)

# Drop the redundant 'owner' column
df.drop(columns=['owner'], inplace=True)

# Clean and extract mileage and rename the column
def clean_mileage(mileage):
    try:
        mileage_str = str(mileage).replace('kmpl', '').replace('km/kg', '').strip()
        mileage_float = float(mileage_str)
        return mileage_float if mileage_float < 100 else np.nan  # Assuming mileage < 100
    except ValueError:
        return np.nan

df['mileage'] = df['top_0_value.2'].apply(clean_mileage)

# Clean and extract seats from 'top_3_value' column and rename the column
def clean_seats(seats):
    try:
        seats_int = int(str(seats).replace('Seats', '').strip())
        return seats_int if seats_int < 10 else np.nan  # Assuming seats < 10
    except ValueError:
        return np.nan

df['Seats'] = df['top_3_value'].apply(clean_seats)

# Drop the original columns 'top_0_value.2' and 'top_3_value'
df.drop(['top_0_value.2', 'top_3_value'], axis=1, inplace=True)

# Save the dataset after cleaning, without encoding or scaling
cleaned_output_path = r'D:\Capstone projects\cardekho\car_dekho_raw.csv'
df.to_csv(cleaned_output_path, index=False)
print(f"Data cleaning complete. Cleaned dataset saved as '{cleaned_output_path}'.")

# Now continue with encoding and scaling for the second dataset
label_encoders = {}

def label_encode(df, columns):
    global label_encoders
    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    return df

categorical_columns = ['ft', 'bt', 'transmission', 'oem', 'model', 'variantName', 'City']
df_encoded = label_encode(df.copy(), categorical_columns)

# Identify constant columns
constant_columns = [col for col in df_encoded.columns if df_encoded[col].nunique() <= 1]

# Drop constant columns
df_encoded = df_encoded.drop(columns=constant_columns)

# Identify columns that contain URLs
url_columns = [col for col in df_encoded.columns if df_encoded[col].astype(str).str.contains('https://images10').any()]

# Drop URL columns
df_encoded = df_encoded.drop(columns=url_columns)

# Normalizing numerical features using Min-Max Scaling
scalers = {}

def min_max_scaling(df, columns):
    global scalers
    for col in columns:
        if col in df.columns:
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])
            scalers[col] = scaler
    return df

df_encoded = min_max_scaling(df_encoded, ['km', 'modelYear', 'ownerNo', 'mileage', 'Seats'])

# Removing outliers using IQR for 'price'
Q1 = df_encoded['price'].quantile(0.25)
Q3 = df_encoded['price'].quantile(0.75)
IQR = Q3 - Q1
df_encoded = df_encoded[(df_encoded['price'] >= (Q1 - 1.5 * IQR)) & (df_encoded['price'] <= (Q3 + 1.5 * IQR))]

# Save the cleaned and transformed data
transformed_output_path = r'D:\Capstone projects\cardekho\car_dekho_cleaned_dataset.csv'
df_encoded.to_csv(transformed_output_path, index=False)

# Save preprocessing steps
label_encoders_path = r'D:\Capstone projects\cardekho\label_encoders.pkl'
scalers_path = r'D:\Capstone projects\cardekho\scalers.pkl'
joblib.dump(label_encoders, label_encoders_path)
joblib.dump(scalers, scalers_path)

print(f"Data cleaning and transformations complete. Encoded and scaled dataset saved as '{transformed_output_path}'.")
print("Preprocessing steps saved.")

