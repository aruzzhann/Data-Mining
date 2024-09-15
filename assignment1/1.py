import pandas as pd
df = pd.read_csv('Apple_price_to_clean.csv')

print(df.head()) #first 5
print(df.tail()) #last 5
print(df.info()) #dataset, data types, number non-null, memory usage
print(df.isnull())
print(df.dtypes)