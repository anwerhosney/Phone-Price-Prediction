import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle as pk

# Sample data creation
data = {
    'brand': ['Samsung', 'Apple', 'Xiaomi', 'Oppo', 'Vivo'] * 20,
    'battery': [3000, 3100, 3200, 3300, 3400] * 20,
    'camera': [12, 13, 14, 15, 16] * 20,
    'price': [20000, 25000, 15000, 18000, 22000] * 20
}
df = pd.DataFrame(data)

# Preprocessing
df['battery'] = df['battery'] / 1000
df['camera'] = df['camera'] / 10

# One-hot encode the categorical 'brand' feature
df = pd.get_dummies(df, columns=['brand'])

# Define the features and target
X = df.drop('price', axis=1)
y = df['price']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save the model and scaler
with open('Price_Predictor.pkl', 'wb') as model_file:
    pk.dump(model, model_file)

with open('Scaler.pkl', 'wb') as scaler_file:
    pk.dump(scaler, scaler_file)

print("Model and scaler saved successfully.")
