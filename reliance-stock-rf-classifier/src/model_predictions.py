import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from model_training import time_series_split

# 1. Load the saved model
model = joblib.load('E:/randomForestClassifier/reliance-stock-rf-classifier/models/rf_model.pkl')

#load the data
X = joblib.load('E:/randomForestClassifier/reliance-stock-rf-classifier/data/X.pkl')
y = joblib.load('E:/randomForestClassifier/reliance-stock-rf-classifier/data/y.pkl')
X_train, X_test, y_train, y_test = time_series_split(X, y, test_size=0.2)
print(y.value_counts())

# 2. Predict
y_pred = model.predict(X_test)

# 3. Plot
plt.figure(figsize=(12, 6))
plt.plot(y_test.reset_index(drop=True), label='Actual', marker='o')
plt.plot(y_pred, label='Predicted', marker='x')
plt.title("Actual vs Predicted Price Direction")
plt.xlabel("Time Index")
plt.ylabel("Class (0 = Down, 1 = Up)")
plt.legend()
plt.grid(True)
plt.show()

# 4. Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
