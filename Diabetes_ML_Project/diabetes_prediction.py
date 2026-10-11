import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv("diabetes.csv")

print("Dataset loaded successfully!")
print("Number of records:", len(data))

# Input and output
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nDiabetes Prediction using Machine Learning")
print("------------------------------------------")
print("Accuracy:", accuracy)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Patient prediction
print("\nEnter Patient Details")

patient = [[
    float(input("Pregnancies: ")),
    float(input("Glucose: ")),
    float(input("Blood Pressure: ")),
    float(input("Skin Thickness: ")),
    float(input("Insulin: ")),
    float(input("BMI: ")),
    float(input("Diabetes Pedigree Function: ")),
    float(input("Age: "))
]]

patient = scaler.transform(patient)

prediction = model.predict(patient)

if prediction[0] == 1:
    print("\nPrediction: Diabetes")
else:
    print("\nPrediction: No Diabetes")