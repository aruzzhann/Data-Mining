import pandas as pd

# 1.1 Calculate speed from distance and time
df = pd.DataFrame({
    'Distance': [100, 150, 200],  # in kilometers
    'Time': [1.5, 2, 2.5]        # in hours
})

# Calculate speed (km/h)
df['Speed'] = df['Distance'] / df['Time']
print("Data with Speed Feature:")
print(df)

# -----------------------------------------------------------------------------

# 1.2 Create polynomial features from existing data
from sklearn.preprocessing import PolynomialFeatures

df = pd.DataFrame({
    'Hours_Studied': [1, 2, 3],
    'Grade_Obtained': [50, 70, 90]
})

poly = PolynomialFeatures(degree=2, include_bias=False)
new_features = poly.fit_transform(df)

new_features_df = pd.DataFrame(new_features, columns=poly.get_feature_names_out(df.columns))
print("\nPolynomial Features DataFrame:")
print(new_features_df)

# -----------------------------------------------------------------------------

# 2. Extract date-based features
df = pd.DataFrame({
    'Event_Date': ['2022-12-31', '2023-06-15', '2024-02-01']
})

df['Event_Date'] = pd.to_datetime(df['Event_Date'])

df['Year'] = df['Event_Date'].dt.year
df['Month'] = df['Event_Date'].dt.month
df['Day'] = df['Event_Date'].dt.day
df['Quarter'] = df['Event_Date'].dt.quarter

print("\nData with Date-Based Features:")
print(df)

# -----------------------------------------------------------------------------

# 3. Engineer features using domain knowledge (e.g., working hours)
df = pd.DataFrame({
    'Start_Work': ['08:00', '14:00', '20:00'],
    'End_Work': ['17:00', '22:00', '06:00']
})

df['Start_Work'] = pd.to_datetime(df['Start_Work'], format='%H:%M').dt.time
df['End_Work'] = pd.to_datetime(df['End_Work'], format='%H:%M').dt.time

df['Work_Duration'] = pd.to_datetime(df['End_Work'].astype(str), format='%H:%M') - pd.to_datetime(df['Start_Work'].astype(str), format='%H:%M')
df['Work_Duration'] = df['Work_Duration'].dt.total_seconds() / 3600  # Convert duration to hours

def shift_type(hour):
    if 6 <= hour < 14:
        return 'Morning Shift'
    elif 14 <= hour < 22:
        return 'Afternoon Shift'
    else:
        return 'Night Shift'

df['Start_Shift_Type'] = df['Start_Work'].apply(lambda x: shift_type(x.hour))
df['End_Shift_Type'] = df['End_Work'].apply(lambda x: shift_type(x.hour))

print("\nData with Work Duration and Shift Type:")
print(df)
