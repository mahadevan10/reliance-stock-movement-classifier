import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, confusion_matrix
import joblib



# === STEP 1: Load preprocessed features ===
input_dir = r'E:\randomForestClassifier\reliance-stock-rf-classifier\data'

with open(os.path.join(input_dir, 'X.pkl'), 'rb') as f:
    X = pickle.load(f)

with open(os.path.join(input_dir, 'y.pkl'), 'rb') as f:
    y = pickle.load(f)

X.fillna(method='ffill', inplace=True)  # forward fill

#train test split
def time_series_split(X, y, test_size=0.2):
   
    split_index = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = time_series_split(X, y, test_size=0.2)


#debugging
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.ffill(inplace=True)



# Optional sanity check
print("NaNs:", X.isna().sum().sum())
print("Infs:", np.isinf(X).sum().sum())
print("Max value:", X.max().max())

# === EXPERIMENTAL STEP 2: Define model and CV strategy ===
# 🎯 Custom scoring function: rewards TP and TN
def balanced_recall_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]

    # Avoid division by zero
    recall_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
    recall_1 = TP / (TP + FN) if (TP + FN) > 0 else 0

    return (recall_0 + recall_1) / 2

# Convert to scorer
scorer = make_scorer(balanced_recall_score)

# 🔧 Parameters to tune
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 4],
    'class_weight': [None, 'balanced']
}

# ⚙️ GridSearchCV setup
rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring=scorer,
    cv=3,
    verbose=2,
    n_jobs=1
)

# 🧠 Train the best model
grid.fit(X_train, y_train)

# 💾 Save best model
rf_model = grid.best_estimator_
joblib.dump(rf_model, 'E:/randomForestClassifier/reliance-stock-rf-classifier/models/rf_model_tuned.pkl')

# 📊 Show best results
print("✅ Best Params:", grid.best_params_)
print("📈 Best Balanced Recall Score:", grid.best_score_)

# === STEP 2: Define model and CV strategy ===
#rf_model = RandomForestClassifier(
#    n_estimators=100,
#    max_depth=5,
#    min_samples_leaf=3,
#    random_state=42
#)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# === STEP 3: Cross-validation ===
cv_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='accuracy')
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

# === STEP 4: Train on training dataset ===
rf_model.fit(X_train, y_train)

# === STEP 5: Save trained model ===
model_output_path = r'E:\randomForestClassifier\reliance-stock-rf-classifier\models\rf_model.pkl'

os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

with open(model_output_path, 'wb') as f:
    pickle.dump(rf_model, f)

print(f"Model saved to: {model_output_path}")
