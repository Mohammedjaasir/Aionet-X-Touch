import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("intrusion.csv")

# Split the Source IP address into octets
data[['Source_Octet1', 'Source_Octet2', 'Source_Octet3', 'Source_Octet4']] = data['Source IP'].str.split('.', expand=True)

# Initialize label encoder
label_encoder = LabelEncoder()

# Convert each octet from string to float
for column in ['Source_Octet1', 'Source_Octet2', 'Source_Octet3', 'Source_Octet4']:
    data[column] = label_encoder.fit_transform(data[column])

# Now each octet of the Source IP is converted from string to float
print(data)
