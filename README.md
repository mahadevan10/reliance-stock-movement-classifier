# 📈 Reliance Stock Price Direction Predictor using Random Forest

A machine learning project that predicts whether **Reliance stock price** will go **up or down the next day** using recent historical OHLCV data and engineered technical indicators. The model is trained using **Random Forest Classifier** with **GridSearchCV** optimization and a custom scoring function for balanced recall.

---

## 🚀 Overview

This project attempts to model stock price *direction* rather than exact value, framing the problem as **binary classification**:

- `0` → Price goes **down**
- `1` → Price goes **up**

Despite challenges due to class imbalance and market noise, this model uses several handcrafted features and hyperparameter tuning to improve performance.

---

## 🔧 Features Used

The following features were extracted from OHLCV data for Reliance:

- Raw data:
  - `Open`, `High`, `Low`, `Close`, `Volume`
- Lag features:
  - `Close_t-1`, `Volume_t-1`
- Technical indicators:
  - `Close_MA_5`, `Close_MA_10`, `Close_STD_5`
  - `Daily_Return`, `Volume_Change`
- Date-based:
  - `Day_of_Week`

---

## 📊 Model Details

- **Model**: `RandomForestClassifier` from `sklearn`
- **Tuning**: `GridSearchCV` (3-fold)
- **Scoring**: Custom `balanced_recall_score()` to reward correct predictions on both classes
- **Hyperparameters tuned**:
  - `n_estimators`
  - `max_depth`
  - `min_samples_leaf`
  - `min_samples_split`
  - `class_weight`

---

## 📁 Folder Structure

