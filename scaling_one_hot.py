import pandas as pd

# Load the dataset from the Excel file
data = pd.read_excel("intrusion.xlsx")
print("Original DataFrame:")
print(data.head())

# Assuming 'prediction' is the column name you want to convert to binary code
feature_column_name = 'predicition'  # Corrected the column name spelling

# Strip any leading or trailing whitespaces from the 'prediction' column
data[feature_column_name] = data[feature_column_name].str.strip()

# Define a replacement dictionary for binary coding
replacement_dict = {'Normal': 0, 'Attack': 1}

# Replace categorical labels with binary codes
data[feature_column_name] = data[feature_column_name].replace(replacement_dict)

# Convert the column to integer type
data[feature_column_name] = data[feature_column_name].astype(int)

# Display the DataFrame with binary-coded features
print("\nDataFrame with Binary Coded Features:")
print(data.head())
