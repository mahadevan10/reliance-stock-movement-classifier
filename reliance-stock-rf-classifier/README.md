# Reliance Stock Random Forest Classifier

This project aims to develop a Random Forest Classifier model to predict whether the stock price of Reliance will close above or below today's closing price. The model utilizes historical stock price data and various features derived from it to make predictions.

## Project Structure

```
reliance-stock-rf-classifier
├── src
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
├── notebooks
│   └── exploratory_analysis.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Files Description

- **src/data_preprocessing.py**: Contains functions for loading and cleaning the stock price data, handling missing values, and formatting the data for further analysis.

- **src/feature_engineering.py**: Includes functions for creating new features from the stock price data, such as moving averages and volatility indicators.

- **src/model_training.py**: Defines the `RandomForestModel` class with methods for training the Random Forest Classifier and saving the trained model to disk.

- **src/model_evaluation.py**: Contains functions for evaluating the model's performance using metrics like accuracy, precision, recall, and F1 score.

- **src/utils.py**: Provides utility functions for data visualization and manipulation used across the project.

- **notebooks/exploratory_analysis.ipynb**: A Jupyter notebook for exploratory data analysis (EDA) on the stock price data, visualizing trends, and understanding data distribution.

- **requirements.txt**: Lists the Python dependencies required for the project, including pandas, scikit-learn, and matplotlib.

- **.gitignore**: Specifies files and directories to be ignored by Git, such as virtual environment folders and temporary files.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd reliance-stock-rf-classifier
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the exploratory analysis notebook to understand the data:
   ```
   jupyter notebook notebooks/exploratory_analysis.ipynb
   ```

## Usage

- Use the `data_preprocessing.py` script to load and clean your stock price data.
- Apply feature engineering techniques using `feature_engineering.py` to create relevant features.
- Train the model using `model_training.py` and evaluate its performance with `model_evaluation.py`.

## License

This project is licensed under the MIT License.