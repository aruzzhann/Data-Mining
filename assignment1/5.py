import pandas as pd

# -----------------------------------------------------------------------------
# 1. Remove duplicate rows
data = {
    'Employee': ['Aruzhan', 'Nargiz', 'Olzhas', 'Aruzhan', 'Nargiz', 'Aruzhan'],
    'Department': ['HR', 'IT', 'IT', 'HR', 'IT', 'HR']
}

df = pd.DataFrame(data)

print("Duplicate Rows:")
print(df[df.duplicated()])

# Remove duplicates and keep the first occurrence
df_no_duplicates = df.drop_duplicates()
print("\nCleaned DataFrame without duplicates:")
print(df_no_duplicates)

# -----------------------------------------------------------------------------

# 2. Detect and remove outliers using Z-score
from scipy import stats

data = {
    'Salary': [3500, 4200, 3800, 9000, 4100, 4600, 5000]
}

df = pd.DataFrame(data)

z_scores = stats.zscore(df['Salary'])
abs_z_scores = abs(z_scores)

df_no_outliers = df[abs_z_scores < 3]

print("\nDataFrame without outliers:")
print(df_no_outliers)

# -----------------------------------------------------------------------------

# 3. Correct inconsistencies in categorical data
data = {
    'City': ['new york', 'London', 'New York', 'LONDON', 'Paris'],
    'Country': ['usa', 'UK', 'USA', 'uk', 'france']
}

df = pd.DataFrame(data)

df['City'] = df['City'].str.title()
df['Country'] = df['Country'].str.upper()

print("\nStandardized DataFrame:")
print(df)
