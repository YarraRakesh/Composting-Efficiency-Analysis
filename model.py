import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("dataset.csv")

# Encode categorical column
le = LabelEncoder()
df['Type_Waste'] = le.fit_transform(df['Type_Waste'])

# Features and target
X = df.drop('Decomposition_Time', axis=1)
y = df['Decomposition_Time']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
print("Model trained and saved!")
