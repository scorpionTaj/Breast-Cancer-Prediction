# Breast Cancer Prediction

## Description

This project implements a machine learning pipeline to predict breast cancer diagnosis (malignant or benign) using the Breast Cancer Wisconsin dataset. The analysis includes data preprocessing, exploratory data analysis (EDA), feature selection, model training with multiple algorithms, evaluation, and visualization of results.

## Dataset

The dataset (`breast_cancer.csv`) is derived from digitized images of fine needle aspirates of breast masses. It contains 569 instances with 32 features, including:

- ID
- Diagnosis (M = malignant, B = benign)
- 30 real-valued features computed from the images

## Features

- **Data Preprocessing**: Handling missing values, encoding categorical variables, feature scaling
- **Exploratory Data Analysis (EDA)**: Distribution plots, correlation heatmap, feature selection to remove multicollinearity
- **Machine Learning Models**:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM) with hyperparameter tuning
  - Decision Tree Classifier with hyperparameter tuning
  - Random Forest Classifier
  - Gradient Boosting Classifier with hyperparameter tuning
  - XGBoost Classifier
- **Model Evaluation**: Accuracy, confusion matrix, classification report, ROC curves
- **Visualization**: ROC curves, performance comparison plots

## Files

- `Project.ipynb`: Jupyter notebook containing the complete analysis and code
- `breast_cancer.csv`: The input dataset
- `brest_cancer.pkl`: Pickled trained SVM model (best performing model)
- `roc_breast_cancer.jpeg`: ROC curve comparison for all models
- `PE_breast_cancer.jpeg`: Performance evaluation plot (Accuracy vs ROC)
- `requirements.txt`: Python package dependencies

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/scorpionTaj/Breast-Cancer-Prediction
   cd Breast-Cancer-Prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Open the Jupyter notebook:

   ```bash
   jupyter notebook Project.ipynb
   ```

2. Run all cells in sequence to reproduce the analysis.

3. The trained model is saved as `brest_cancer.pkl` and can be loaded for predictions:
   ```python
   import pickle
   model = pickle.load(open("brest_cancer.pkl", "rb"))
   prediction = model.predict(new_data)
   ```

## Results

The project compares multiple machine learning models and saves the best performing model (SVM) with hyperparameter tuning. Key visualizations include:

- ROC curves for all models
- Performance metrics comparison (Accuracy and ROC AUC)

## Dependencies

The required Python packages are listed in `requirements.txt`. The main dependencies include:

- pandas: Data manipulation and analysis
- numpy: Numerical computing
- seaborn: Statistical data visualization
- matplotlib: Plotting library
- scikit-learn: Machine learning algorithms
- missingno: Missing data visualization
- xgboost: Gradient boosting framework
- jupyter: Jupyter notebook environment

## License

This project is for educational purposes. Please refer to the dataset source for licensing information.
