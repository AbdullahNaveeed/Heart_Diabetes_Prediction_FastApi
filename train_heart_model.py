import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib


df = pd.read_excel("Heart_datasets.xlsx")
df.dropna(inplace=True)


X = df.drop("target", axis=1)
y = df["target"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)


model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

with open("heart_accuracy.txt", "w") as f:
    f.write(str(round(accuracy * 100, 2)))


joblib.dump(model, "heart_model.pkl")
joblib.dump(scaler, "heart_scaler.pkl")

print("Heart model and scaler saved successfully.")
