import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = {
    'Age': [22, 25, 47, 52, 46, 56, 30, 34, 40, 50],
    'Income': [35000, 40000, 60000, 80000, 52000, 90000, 45000, 48000, 58000, 75000],
    'Purchased': [0, 0, 1, 1, 1, 1, 0, 0, 1, 1]  # 0 = Not Purchased, 1 = Purchased
}

df = pd.DataFrame(data)

X = df.drop('Purchased', axis=1)  # Features: 'Age' and 'Income'
y = df['Purchased']  # Target: 'Purchased'

# Train-test split with 75-25 ratio
X_train_75, X_test_25, y_train_75, y_test_25 = train_test_split(X, y, test_size=0.25, random_state=42)

# Train-test split with 85-15 ratio
X_train_85, X_test_15, y_train_85, y_test_15 = train_test_split(X, y, test_size=0.15, random_state=42)

model = DecisionTreeClassifier()

model.fit(X_train_75, y_train_75)
predictions_75 = model.predict(X_test_25)
accuracy_75 = accuracy_score(y_test_25, predictions_75)

model.fit(X_train_85, y_train_85)
predictions_85 = model.predict(X_test_15)
accuracy_85 = accuracy_score(y_test_15, predictions_85)

print(f"Accuracy with 75-25 split: {accuracy_75:.2f}")
print(f"Accuracy with 85-15 split: {accuracy_85:.2f}")