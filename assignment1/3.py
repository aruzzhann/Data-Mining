# 1. Normalize numerical features using Min-Max scaling
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

data = pd.DataFrame({
    'price': [2000, 4000, 6000, 8000, 10000]
})

scaler = MinMaxScaler()

data[['price']] = scaler.fit_transform(data[['price']])
print("Normalized Data:")
print(data)

# -----------------------------------------------------------------------------

# 2. Encode categorical variables using OneHotEncoder from sklearn
from sklearn.preprocessing import OneHotEncoder

data = pd.DataFrame({
    'Category': ['X', 'Y', 'Z', 'X'],
    'Value': [10, 20, 30, 40]
})

encoder = OneHotEncoder(sparse=False)

encoded_categories = encoder.fit_transform(data[['Category']])

encoded_df = pd.DataFrame(encoded_categories, columns=encoder.get_feature_names_out(['Category']))

data_encoded = pd.concat([data[['Value']], encoded_df], axis=1)
print("\nOne-Hot Encoded Data:")
print(data_encoded)

# -----------------------------------------------------------------------------

# 3. Bin continuous variables into discrete intervals using pd.cut()
import pandas as pd

data = pd.DataFrame({
    'Height': [150, 160, 170, 180, 190, 200, 210]
})

data['Height_Group'] = pd.cut(data['Height'], bins=3, labels=['Short', 'Medium', 'Tall'])
print("\nBinned Data:")
print(data)
