# Titanic Dataset - Exploratory Data Analysis and Logistic Regression

This project explores the famous Titanic dataset from Kaggle, which is a common starting point for machine learning. The goal of this project is to predict whether a passenger survived or not using Logistic Regression, after performing comprehensive data cleaning and exploratory data analysis (EDA).

## Project Overview

- **Dataset**: The Titanic dataset (`titanic_train.csv`).
- **Goal**: Predict passenger survival based on features like class, age, sex, and more.
- **Methods**:
  - Data Cleaning and Imputation.
  - Exploratory Data Analysis (EDA).
  - Feature Engineering.
  - Logistic Regression for classification.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Exploratory Data Analysis](#exploratory-data-analysis)
3. [Data Cleaning](#data-cleaning)
4. [Feature Engineering](#feature-engineering)
5. [Logistic Regression Model](#logistic-regression-model)
6. [Results](#results)
7. [Dependencies](#dependencies)
8. [Acknowledgments](#acknowledgments)

---

## Getting Started

### Import Libraries
The following Python libraries were used in the project:

python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

### Load Dataset
The dataset is loaded using Pandas:
train = pd.read_csv('titanic_train.csv')
### Exploratory Data Analysis
- Visualized missing data using Seaborn heatmaps.
- Explored the distribution of age, fare, survival rate, and other features.
- Plotted correlations between survival and categorical variables such as sex, passenger class, and siblings/spouses aboard.
### Sample Plots
Survival count by sex:

### Age distribution:
sns.histplot(train['Age'].dropna(), kde=True, color='darkred', bins=40)

### Data Cleaning
#### Missing Values:
- Age: Imputed using the mean age by Pclass.
- Cabin: Dropped due to excessive missing values.
- Embarked: Dropped rows with missing values.
#### Final Dataset: Cleaned and ready for machine learning.
### Feature Engineering
- Converted categorical variables (Sex, Embarked) into dummy variables using pd.get_dummies.
- Dropped redundant columns like Name and Ticket.
### Logistic Regression Model
1. Train-Test Split:
2. Model Training:
3. Predictions: Predictions were generated using the test data.

### Results
- Accuracy: Achieved through Logistic Regression.
- Key Insights:
- Survival rates are significantly higher for females.
- First-class passengers had better survival chances.

### Dependencies
The project requires the following Python packages:
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn
### Install dependencies via pip:
pip install pandas numpy matplotlib seaborn scikit-learn

### Acknowledgments
- Dataset: Kaggle Titanic Dataset
- Tutorials and References: Inspiration from various online resources for implementing Logistic Regression and EDA.
